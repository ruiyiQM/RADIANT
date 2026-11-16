"""Device-state equivalence, transfer guards and resident restart regressions."""

from copy import deepcopy
from importlib.util import find_spec
from itertools import product
from pathlib import Path
import sys
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from radiant.backends.gpu_resident import GPUResidentProvider
from radiant.backends.gpu_pyscf import GPU4PySCFProvider
from radiant.backends.pyscf import PySCFProvider
from radiant.backends.pyscf_nac import ZVectorNAC, _derivative_integrals
from radiant.core.arrays import to_host
from radiant.core.execution import LocalExecution
from radiant.core.propagation import CNOptions, crank_nicolson
from radiant.core.state import OrbitalState
from radiant.core.transport import lowdin, transport
from radiant.restart import load_checkpoint, save_checkpoint
from test_gpu_provider import complex_state, mean_field
from test_restart import make_run


class ResidentContractTests(unittest.TestCase):
    def test_incremental_potential_owns_density_and_failed_build_keeps_cache(self):
        provider=GPUResidentProvider(mean_field())
        from unittest.mock import Mock
        first=SimpleNamespace(vj=np.eye(2),ecoul=1.)
        second=SimpleNamespace(vj=2*np.eye(2),ecoul=2.)
        native=SimpleNamespace(direct_scf=True,get_veff=Mock(side_effect=[first,second,RuntimeError('J failure')]))
        provider._gpu_mf=native
        dm=np.eye(2)
        self.assertIs(provider._potential(dm,incremental=True),first)
        dm[0,0]=2
        self.assertIs(provider._potential(dm,incremental=True),second)
        a,b=native.get_veff.call_args_list
        self.assertNotIn('dm_last',a.kwargs)
        assert_allclose(b.kwargs['dm_last'],np.eye(2))
        self.assertIs(b.kwargs['vhf_last'].vj,first.vj)
        self.assertFalse(hasattr(b.kwargs['vhf_last'],'ecoul'))
        with self.assertRaisesRegex(RuntimeError,'J failure'):
            provider._potential(dm+1,incremental=True)
        self.assertIs(native._radiant_potential.vj,second.vj)
        assert_allclose(native._radiant_density,dm)
        self.assertIsNone(deepcopy(provider)._gpu_mf)
        native.direct_scf=False;native.get_veff=Mock(return_value=first)
        provider._potential(dm,incremental=True)
        self.assertNotIn('dm_last',native.get_veff.call_args.kwargs)
        native.direct_scf=True;native.get_veff.reset_mock()
        provider._potential(dm,incremental=False)
        for _ in range(40):provider._potential(dm,incremental=True)
        full_builds=sum('dm_last' not in call.kwargs for call in native.get_veff.call_args_list)
        self.assertGreaterEqual(full_builds,3)

    def test_high_angular_momentum_cannot_fall_back_to_cpu(self):
        from pyscf import gto
        mf=mean_field();mf.mol._bas[0,gto.ANG_OF]=5
        with patch.dict(sys.modules,{'cupy':None,'gpu4pyscf':None}):
            with self.assertRaisesRegex(ValueError,'angular momentum'):
                GPUResidentProvider(mf)

    def test_failed_grid_export_preserves_complete_host_cache(self):
        provider = GPUResidentProvider(mean_field())
        grids = provider.mf.grids
        old_coords, old_weights = grids.coords, grids.weights
        provider._gpu_mf = SimpleNamespace(grids=SimpleNamespace(coords=np.zeros((2, 3)), weights=np.ones(2)))
        with patch('radiant.backends.gpu_resident.to_host', side_effect=[np.ones((2, 3)), RuntimeError('transfer failure')]):
            with self.assertRaisesRegex(RuntimeError, 'transfer failure'):
                provider.capture_grids()
        self.assertIs(grids.coords, old_coords)
        self.assertIs(grids.weights, old_weights)

    def test_construction_does_not_load_optional_gpu_packages(self):
        mf=mean_field()
        with patch.dict(sys.modules,{'cupy':None,'gpu4pyscf':None}):
            provider=GPUResidentProvider(mf)
            self.assertTrue(provider.gpu_forces)
            self.assertIsNone(provider._gpu_mf)
            with self.assertRaises(ImportError):provider.initial_state()


@unittest.skipUnless(find_spec('cupy') and find_spec('gpu4pyscf'),'Requires a real GPU')
class ResidentHardwareTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('RADIANT_TEST_DEVICE_J_PATCH')=='1',
                         'Requires the bundled isolated GPU4PySCF device-J patch')
    def test_patched_coulomb_transform_receives_device_tensor(self):
        import cupy as cp
        from gpu4pyscf.scf import j_engine
        original=j_engine._Rt_to_dm
        calls=[]
        def device_transform(mol,tensor,*args,**kwargs):
            self.assertIsInstance(tensor,cp.ndarray)
            calls.append(tensor.shape)
            return original(mol,tensor,*args,**kwargs)
        for unrestricted in (False,True):
            mf=mean_field(unrestricted);provider=GPUResidentProvider(mf)
            state=provider.initial_state()
            with patch.object(j_engine,'_Rt_to_dm',device_transform):
                actual=provider.fock(state.density())
            expected=PySCFProvider(mf).fock(to_host(state.density()))
            assert_allclose(to_host(actual),expected,atol=2e-9,rtol=0)
        self.assertEqual(len(calls),2)

    def test_incremental_fock_matches_full_rebuild_and_geometry_reset(self):
        import cupy as cp
        for unrestricted in (False,True):
            provider=GPUResidentProvider(mean_field(unrestricted))
            initial=provider.initial_state().density()
            other=complex_state(PySCFProvider(provider.mf)).density()
            other=cp.asarray(other)
            for weight in (.1,.01,.001,.0001,0.,.05)*7:
                density=(1-weight)*initial+weight*other
                actual=provider.fock(density)
                expected=provider.hcore+provider._gpu_mean_field().get_veff(dm=density.real)
                assert_allclose(to_host(actual),to_host(expected),atol=2e-9,rtol=0)
            native=provider._gpu_mf
            moved=provider.positions;moved[0,0]+=.001
            provider.prepare_geometry(moved)
            self.assertIsNot(provider._gpu_mf,native)
            self.assertIsNone(provider._gpu_mf._radiant_potential)
            actual=provider.fock(initial)
            expected=provider.hcore+provider._gpu_mean_field().get_veff(dm=initial.real)
            assert_allclose(to_host(actual),to_host(expected),atol=2e-9,rtol=0)

    def test_response_ao_derivatives_and_complex_nac_match_cpu(self):
        import cupy as cp
        for unrestricted, xc in product((False, True), ('lda,vwn', 'pbe')):
            with self.subTest(unrestricted=unrestricted, xc=xc):
                mf = mean_field(unrestricted, xc)
                runs = [make_run(mf, backend=kind).execution for kind in (PySCFProvider, GPUResidentProvider)]
                cpu, gpu = (r.providers[0] for r in runs)
                for a, b in zip(_derivative_integrals(mf.mol), gpu.derivative_integrals()):
                    assert_allclose(to_host(b), a, atol=2e-10, rtol=0)
                blocks = mf.mo_coeff if unrestricted else (mf.mo_coeff,)
                occupations = mf.mo_occ if unrestricted else (mf.mo_occ,)
                mo = [cp.asarray(c[:, f > 0]) for c, f in zip(blocks, occupations)]
                occ = [cp.asarray(f[f > 0]) for f in occupations]
                if not unrestricted:
                    mo, occ = mo[0], occ[0]
                model = gpu.response_model(mo, occ)
                actual = gpu.response_h1(model, mo, occ)
                expected = mf.Hessian().make_h1(mf.mo_coeff, mf.mo_occ)
                for a, b in zip(actual if unrestricted else (actual,), expected if unrestricted else (expected,)):
                    assert_allclose(to_host(a), b, atol=3e-7, rtol=0)
                for step in range(2):
                    if step:
                        for run in runs:
                            end = run.positions; end[0, 0] += .001
                            run.advance(end, .02, forces=False)
                    reference = ZVectorNAC(tolerance=1e-10).compute(runs[0])
                    exported = []; asnumpy = cp.asnumpy
                    def small_only(value, *args, **kwargs):
                        exported.append(value.shape)
                        self.assertEqual(value.shape, gpu.positions.shape, 'NAC exported an electronic matrix')
                        return asnumpy(value, *args, **kwargs)
                    with patch.object(cp, 'asnumpy', small_only):
                        result = ZVectorNAC(tolerance=1e-10).compute(runs[1])
                    assert_allclose(result.raw, reference.raw, atol=3e-7, rtol=0)
                    assert_allclose(result.real_norm, reference.real_norm, atol=3e-7, rtol=0)
                    self.assertEqual(len(exported), 2)

    def test_device_fock_and_full_fixed_x_force_match_cpu_and_energy_differences(self):
        import cupy as cp
        for unrestricted,xc in product((False,True),('lda,vwn','pbe')):
            with self.subTest(unrestricted=unrestricted,xc=xc):
                mf=mean_field(unrestricted,xc)
                cpu,gpu=PySCFProvider(mf),GPUResidentProvider(mf)
                original=complex_state(cpu)
                device=OrbitalState(tuple(cp.asarray(c) for c in original.coefficients),original.occupations)
                old_cpu,old_gpu=cpu.overlap,gpu.overlap
                for shift in (0.,.01):
                    coords=mf.mol.atom_coords();coords[0,0]+=shift
                    cpu.prepare_geometry(coords);gpu.prepare_geometry(coords)
                    state=transport(original,old_cpu,cpu.overlap)
                    resident=transport(device,old_gpu,gpu.overlap)
                    self.assertTrue(all(isinstance(c,cp.ndarray) for c in resident.coefficients))
                    assert_allclose(cp.asnumpy(gpu.overlap),cpu.overlap,atol=2e-11,rtol=0)
                    assert_allclose(cp.asnumpy(gpu.fock(resident.density())),cpu.fock(state.density()),atol=2e-8,rtol=0)
                    with patch.object(OrbitalState,'density',autospec=True,side_effect=OrbitalState.density) as build, \
                            patch.object(gpu,'_density',wraps=gpu._density) as validate:
                        energy,force=gpu.evaluate(resident)
                    self.assertEqual(build.call_count,1)
                    self.assertEqual(validate.call_count,1)
                    self.assertAlmostEqual(energy,cpu.energy(state),delta=2e-8)
                    assert_allclose(force,cpu.forces(state),atol=2e-8,rtol=0)
                    values=[]
                    for sign in (1,-1):
                        candidate=deepcopy(gpu);moved=coords.copy();moved[0,2]+=sign*1e-4
                        candidate.prepare_geometry(moved)
                        values.append(candidate.energy(transport(resident,gpu.overlap,candidate.overlap)))
                    self.assertAlmostEqual(force[0,2],-(values[0]-values[1])/2e-4,delta=3e-7)

    def test_all_propagators_keep_large_matrices_on_device(self):
        import cupy as cp
        rng=np.random.default_rng(19);n=5
        a=rng.normal(size=(n,n));s=a@a.T+np.eye(n)
        inverse=lowdin(s)[0]
        h=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));h=(h+h.conj().T)*.1
        q=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
        cases=[([2.,2.],),([2.,1.,1.],),([1.,1.],[1.]),([1.],[])]
        for occupations,variant in product(cases,('cn','cn_accelerated','pt_cn','pt_im')):
            state=OrbitalState(tuple(inverse@q[:,:len(f)] for f in occupations),occupations)
            def functional(xp,ham,density):
                total=density if density.ndim==2 else density.sum(axis=0)
                value=ham+.03*total
                return value if density.ndim==2 else xp.stack((value,value))
            options=CNOptions(variant,1e-10,120)
            expected,_=crank_nicolson(state,s,lambda d,t:functional(np,h,d),0.,.1,options=options)
            resident=OrbitalState(tuple(cp.asarray(c) for c in state.coefficients),state.occupations)
            ds,dh=cp.asarray(s),cp.asarray(h)
            uploads=[];original_asarray=cp.asarray
            def guarded(value,*args,**kwargs):
                if isinstance(value,np.ndarray):
                    uploads.append(value.shape)
                    self.assertLess(value.ndim,2,'Host matrix entered the electronic iteration')
                return original_asarray(value,*args,**kwargs)
            with patch.object(cp,'asnumpy',side_effect=AssertionError('Device matrix exported during iteration')), \
                    patch.object(cp,'asarray',guarded):
                actual,_=crank_nicolson(resident,ds,lambda d,t:functional(cp,dh,d),0.,.1,options=options)
            assert_allclose(cp.asnumpy(actual.density()),expected.density(),atol=2e-8,rtol=0)
            self.assertTrue(all(isinstance(c,cp.ndarray) for c in actual.coefficients))

    def test_moving_pair_rollback_and_restart_preserve_device_states(self):
        import cupy as cp
        for unrestricted in (False,True):
            mf=mean_field(unrestricted)
            full=make_run(mf,backend=GPUResidentProvider)
            split=make_run(mf,backend=GPUResidentProvider)
            full.run(4,.02)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'state.npz'
                split.run(2,.02,checkpoint=path)
                resumed=load_checkpoint(path,[GPUResidentProvider(mf),GPUResidentProvider(mf)])
                self.assertTrue(all(isinstance(c,cp.ndarray) for s in resumed.execution.states for c in s.coefficients))
                before=resumed.execution.snapshot()
                for failing in ('fock','_forces'):
                    with patch.object(GPUResidentProvider,failing,side_effect=RuntimeError('device failure')):
                        with self.assertRaisesRegex(RuntimeError,'device failure'):resumed.run(1,.02)
                    self.assertEqual(resumed.execution.time,before.time)
                    self.assertIsNone(resumed.execution.providers[0]._gpu_mf)
                    for a,b in zip(resumed.execution.states,before.states):
                        assert_allclose(to_host(a.density()),to_host(b.density()),atol=2e-14,rtol=0)
                resumed.run(2,.02,checkpoint=path)
                assert_allclose(resumed.nuclei.positions,full.nuclei.positions,atol=2e-9,rtol=0)
                assert_allclose(resumed.nuclei.momenta,full.nuclei.momenta,atol=2e-9,rtol=0)
                assert_allclose(resumed.method.coefficients,full.method.coefficients,atol=2e-9,rtol=0)
                for a,b in zip(resumed.execution.states,full.execution.states):
                    assert_allclose(to_host(a.density()),to_host(b.density()),atol=2e-9,rtol=0)
                legacy = make_run(mf, backend=GPU4PySCFProvider)
                legacy.run(2, .02, checkpoint=path)
                migrated = load_checkpoint(path, [GPUResidentProvider(mf), GPUResidentProvider(mf)])
                self.assertTrue(all(p.gpu_forces for p in migrated.execution.providers))
                migrated.run(2, .02)
                assert_allclose(migrated.nuclei.positions, full.nuclei.positions, atol=2e-9, rtol=0)
                for a, b in zip(migrated.execution.states, full.execution.states):
                    assert_allclose(to_host(a.density()), to_host(b.density()), atol=2e-9, rtol=0)

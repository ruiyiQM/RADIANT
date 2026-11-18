"""DF-J energy derivatives, model-safe restart and optional device equivalence."""

from copy import deepcopy
from importlib.util import find_spec
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from radiant.backends.pyscf import PySCFProvider
from radiant.backends.gpu_resident import GPUResidentProvider
from radiant.backends.gpu_pyscf import GPU4PySCFProvider
from radiant.backends.pyscf_nac import ZVectorNAC
from radiant.core.arrays import to_host
from radiant.core.state import OrbitalState
from radiant.core.transport import transport
from radiant.restart import load_checkpoint
from test_gpu_provider import mean_field, complex_state
from test_restart import make_run


def fitted_field(unrestricted=False, xc='pbe'):
    mf = mean_field(unrestricted, xc).density_fit(auxbasis='weigend', only_dfj=True)
    mf.kernel()
    assert mf.converged
    return mf


class DensityFittingTests(unittest.TestCase):
    def test_complex_fixed_x_force_matches_df_energy_finite_differences(self):
        for unrestricted, xc in product((False, True), ('lda,vwn', 'pbe')):
            with self.subTest(unrestricted=unrestricted, xc=xc):
                provider = PySCFProvider(fitted_field(unrestricted, xc))
                state = complex_state(provider)
                density = state.density().real
                expected = provider.mf.get_hcore() + provider.mf.get_veff(dm=density)
                assert_allclose(provider.fock(state.density()), expected, atol=2e-12, rtol=0)
                _, force = provider.evaluate(state)
                for axis in (0, 2):
                    energies = []
                    for sign in (1, -1):
                        moved = deepcopy(provider)
                        positions = moved.positions; positions[0, axis] += sign*1e-4
                        moved.prepare_geometry(positions)
                        energies.append(moved.energy(transport(state, provider.overlap, moved.overlap)))
                    self.assertAlmostEqual(force[0, axis], -(energies[0]-energies[1])/2e-4, delta=3e-7)

    def test_restart_preserves_df_model_and_rejects_auxiliary_basis_changes(self):
        for unrestricted in (False, True):
            mf = fitted_field(unrestricted)
            full, split = make_run(mf), make_run(mf)
            full.run(4, .02)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'df.npz'
                split.run(2, .02, checkpoint=path)
                with np.load(path, allow_pickle=False) as data:
                    model = json.loads(str(data['metadata']))['models'][0]
                    self.assertEqual(model['density_fitting'], {'auxbasis':'weigend', 'only_dfj':True})
                with patch.object(type(mf), 'kernel', side_effect=AssertionError('SCF during load')):
                    resumed = load_checkpoint(path, split.execution.providers)
                resumed.run(2, .02)
                assert_allclose(resumed.nuclei.positions, full.nuclei.positions, atol=2e-12, rtol=0)
                assert_allclose(resumed.nuclei.momenta, full.nuclei.momenta, atol=2e-12, rtol=0)
                for a, b in zip(resumed.execution.states, full.execution.states):
                    assert_allclose(a.density(), b.density(), atol=2e-12, rtol=0)
                other = deepcopy(mf); other.with_df.auxbasis = 'def2-universal-jkfit'
                for mismatched in (other, mf.undo_df()):
                    with self.assertRaisesRegex(ValueError, 'settings do not match'):
                        load_checkpoint(path, [PySCFProvider(mismatched) for _ in range(2)])


@unittest.skipUnless(find_spec('cupy') and find_spec('gpu4pyscf'), 'Requires a real GPU')
class DensityFittingHardwareTests(unittest.TestCase):
    def test_device_df_force_response_and_nac_match_cpu(self):
        import cupy as cp
        for unrestricted, xc in product((False, True), ('lda,vwn', 'pbe')):
            with self.subTest(unrestricted=unrestricted, xc=xc):
                mf = fitted_field(unrestricted, xc)
                cpu, gpu = PySCFProvider(mf), GPUResidentProvider(mf)
                state = complex_state(cpu)
                device = OrbitalState(tuple(cp.asarray(c) for c in state.coefficients), state.occupations)
                for gpu_forces in (False, True):
                    legacy = GPU4PySCFProvider(mf, gpu_forces=gpu_forces)
                    assert_allclose(legacy.fock(state.density()), cpu.fock(state.density()), atol=2e-8, rtol=0)
                    energy, force = legacy.evaluate(state)
                    expected_energy, expected_force = cpu.evaluate(state)
                    self.assertAlmostEqual(energy, expected_energy, delta=2e-8)
                    assert_allclose(force, expected_force, atol=3e-7, rtol=0)
                for shift in (0., .003):
                    old_cpu, old_gpu = cpu.overlap, gpu.overlap
                    positions = cpu.positions; positions[0, 0] += shift
                    cpu.prepare_geometry(positions); gpu.prepare_geometry(positions)
                    state = transport(state, old_cpu, cpu.overlap)
                    device = transport(device, old_gpu, gpu.overlap)
                    assert_allclose(to_host(gpu.fock(device.density())), cpu.fock(state.density()), atol=2e-8, rtol=0)
                    self.assertFalse(gpu._gpu_mf.direct_scf)
                    ce, cf = cpu.evaluate(state); ge, gf = gpu.evaluate(device)
                    self.assertAlmostEqual(ge, ce, delta=2e-8)
                    assert_allclose(gf, cf, atol=3e-7, rtol=0)
                runs = [make_run(mf, backend=kind).execution for kind in (PySCFProvider, GPUResidentProvider)]
                for step in range(2):
                    if step:
                        for run in runs:
                            positions = run.positions; positions[0, 0] += .001
                            run.advance(positions, .02, forces=False)
                    expected = ZVectorNAC(tolerance=1e-10).compute(runs[0])
                    actual = ZVectorNAC(tolerance=1e-10).compute(runs[1])
                    assert_allclose(actual.raw, expected.raw, atol=3e-7, rtol=0)

    def test_device_df_geometry_failure_and_restart_preserve_owned_caches(self):
        for unrestricted in (False, True):
            mf = fitted_field(unrestricted)
            provider = GPUResidentProvider(mf)
            state = provider.initial_state(); expected = to_host(provider.fock(state.density()))
            native = provider._gpu_mf; integrals = native.with_df.intopt
            positions = provider.positions; positions[0, 0] += .001
            with patch('radiant.backends.gpu_resident.lowdin', side_effect=RuntimeError('metric failure')):
                with self.assertRaisesRegex(RuntimeError, 'metric failure'):
                    provider.prepare_geometry(positions)
            self.assertIs(provider._gpu_mf, native)
            self.assertIs(native.with_df.intopt, integrals)
            assert_allclose(native.with_df.mol.atom_coords(), mf.mol.atom_coords(), atol=0, rtol=0)
            assert_allclose(to_host(provider.fock(state.density())), expected, atol=2e-10, rtol=0)
            full, split = (make_run(mf, backend=GPUResidentProvider) for _ in range(2))
            full.run(4, .02)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'device-df.npz'; split.run(2, .02, checkpoint=path)
                resumed = load_checkpoint(path, split.execution.providers)
                resumed.run(2, .02)
                assert_allclose(resumed.nuclei.momenta, full.nuclei.momenta, atol=2e-9, rtol=0)
                for a, b in zip(resumed.execution.states, full.execution.states):
                    assert_allclose(to_host(a.density()), to_host(b.density()), atol=2e-9, rtol=0)

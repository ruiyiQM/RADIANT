"""GPU-provider contracts and optional real-device CPU/GPU regressions."""

from copy import deepcopy
from importlib.util import find_spec
from itertools import product
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from pyscf import dft, gto, scf

from radiant.backends.gpu_pyscf import GPU4PySCFProvider
from radiant.backends.pyscf import PySCFProvider
from radiant.backends.pyscf_nac import ZVectorNAC
from radiant.core.execution import LocalExecution
from radiant.core.propagation import CNOptions
from radiant.core.state import OrbitalState
from radiant.core.transport import lowdin, transport
from radiant.driver import follow_path
from radiant.methods.mash import MASH
from test_pt_mixed import lih_mean_field


def mean_field(unrestricted=False, xc="pbe"):
    mol = gto.M(atom="He 0 0 0; H .2 0 1.4", basis="6-31g", unit="Bohr",
                charge=0 if unrestricted else 1, spin=int(unrestricted), verbose=0)
    mf = (dft.UKS if unrestricted else dft.RKS)(mol, xc=xc)
    mf.grids.level = 1
    mf.grids.prune = None
    mf.conv_tol = 1e-12
    mf.chkfile = None
    mf.kernel()
    assert mf.converged
    return mf


def complex_state(provider):
    rng = np.random.default_rng(42)
    inverse, _ = lowdin(provider.overlap)
    n = len(inverse)
    state = provider.initial_state()
    coefficients = []
    for c in state.coefficients:
        rotation = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
        coefficients.append(inverse @ rotation[:, :c.shape[1]])
    return OrbitalState(tuple(coefficients), state.occupations)


class GPUProviderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mean_fields = [mean_field(unrestricted) for unrestricted in (False, True)]

    def test_supported_models_do_not_load_optional_gpu_packages(self):
        with patch.dict(sys.modules, {"cupy": None, "gpu4pyscf": None}):
            for mf in self.mean_fields:
                provider = GPU4PySCFProvider(mf)
                self.assertIsNone(provider._gpu_mf)
                assert_allclose(provider.initial_state().density(), mf.make_rdm1(), atol=1e-14)
                with self.assertRaises(ImportError):
                    provider.fock(provider.initial_state().density())

    def test_unsupported_models_and_invalid_inputs_fail_before_gpu_use(self):
        mf = self.mean_fields[0]
        hybrid = deepcopy(mf)
        hybrid.xc = "b3lyp"
        meta = deepcopy(mf)
        meta.xc = "tpss"
        hf = scf.RHF(mf.mol)
        hf.converged = True
        with patch.dict(sys.modules, {"cupy": None, "gpu4pyscf": None}):
            for candidate in (hybrid, meta, hf, hybrid.density_fit()):
                with self.assertRaises(ValueError):
                    GPU4PySCFProvider(candidate)
            for mode in (None, 1, 'gpu'):
                with self.assertRaises(ValueError):
                    GPU4PySCFProvider(mf, gpu_forces=mode)
            provider = GPU4PySCFProvider(mf)
            density = provider.initial_state().density()
            for value, time in ((density, np.nan), (density[:1], 0.),
                                (np.full_like(density, np.nan), 0.),
                                (density + np.triu(np.ones_like(density), 1), 0.)):
                with self.assertRaises(ValueError):
                    provider.fock(value, time)

    def test_geometry_and_snapshot_discard_device_caches(self):
        transfers = SimpleNamespace(asarray=np.asarray, asnumpy=np.asarray)
        for mf in self.mean_fields:
            provider = GPU4PySCFProvider(mf)
            cpu = PySCFProvider(mf)
            state = complex_state(cpu)
            with patch.dict(sys.modules, {"cupy": transfers}), patch.object(
                    type(mf), "to_gpu", autospec=True, side_effect=lambda value: deepcopy(value)) as convert:
                expected = cpu.fock(state.density())
                assert_allclose(provider.fock(state.density()), expected, atol=1e-12, rtol=0)
                self.assertAlmostEqual(provider.energy(state), cpu.energy(state), places=12)
                with patch.object(provider, 'forces', side_effect=lambda state, **kw:
                                  PySCFProvider.forces(provider, state, **kw)), patch.object(
                        provider._gpu_mf, 'get_veff', wraps=provider._gpu_mf.get_veff) as potential:
                    energy, force = provider.evaluate(state)
                    self.assertEqual(potential.call_count, 1)
                self.assertAlmostEqual(energy, cpu.energy(state), places=12)
                assert_allclose(force, cpu.forces(state), atol=1e-12, rtol=0)
                self.assertEqual(convert.call_count, 1)
                cached = provider._gpu_mf
                with patch.object(cached, "__deepcopy__", side_effect=AssertionError("device cache copied"), create=True):
                    saved = deepcopy(provider)
                self.assertIsNone(saved._gpu_mf)
                self.assertIs(provider._gpu_mf, cached)
                provider.prepare_geometry(provider.positions)
                self.assertIs(provider._gpu_mf, cached)
                with self.assertRaises(ValueError):
                    provider.prepare_geometry([[0., 0., 0.]])
                self.assertIs(provider._gpu_mf, cached)
                old_overlap = provider.overlap
                moved = provider.positions.copy()
                moved[0, 0] += .01
                provider.prepare_geometry(moved)
                self.assertIsNone(provider._gpu_mf)
                cpu.prepare_geometry(moved)
                state = transport(state, old_overlap, provider.overlap)
                assert_allclose(provider.fock(state.density()), cpu.fock(state.density()), atol=1e-12, rtol=0)
                assert_allclose(saved.positions, mf.mol.atom_coords(), atol=0, rtol=0)

    def test_device_failures_are_not_replaced_by_cpu_results(self):
        provider = GPU4PySCFProvider(self.mean_fields[0])
        with patch.dict(sys.modules, {"cupy": SimpleNamespace(asarray=np.asarray, asnumpy=np.asarray)}), \
                patch.object(provider, "_gpu_mean_field", side_effect=RuntimeError("device failure")):
            with self.assertRaisesRegex(RuntimeError, "device failure"):
                provider.fock(provider.initial_state().density())


@unittest.skipUnless(find_spec("cupy") and find_spec("gpu4pyscf"),
                     "Real-device tests require CuPy and GPU4PySCF")
class GPUProviderHardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mean_fields = [mean_field(unrestricted, xc)
                           for unrestricted in (False, True) for xc in ("lda,vwn", "pbe")]

    def test_gradient_device_failure_rolls_back_without_cpu_fallback(self):
        mf = self.mean_fields[0]
        execution = LocalExecution([GPU4PySCFProvider(mf, gpu_forces=True)])
        execution.observe()
        before = execution.snapshot()
        moved = execution.positions.copy(); moved[0, 0] += .001
        gradient_class = type(execution.providers[0]._gpu_mf.nuc_grad_method())
        with patch.object(gradient_class, 'energy_ee', side_effect=RuntimeError('device gradient failure')), \
                patch.object(PySCFProvider, 'forces', side_effect=AssertionError('CPU fallback')):
            with self.assertRaisesRegex(RuntimeError, 'device gradient'):
                execution.advance(moved, .02)
        self.assertEqual(execution.time, before.time)
        assert_allclose(execution.positions, before.providers[0].positions, atol=0, rtol=0)
        assert_allclose(execution.states[0].density(), before.states[0].density(), atol=0, rtol=0)

    def test_complex_fock_energy_and_moving_fixed_x_forces(self):
        import cupy
        for mf in self.mean_fields:
            with self.subTest(kind=type(mf).__name__, xc=mf.xc):
                cpu, gpu = PySCFProvider(mf), GPU4PySCFProvider(mf, gpu_forces=True)
                state = complex_state(cpu)
                self.assertGreater(np.max(abs(state.density().imag)), 1e-3)
                old_overlap = cpu.overlap
                for displacement in (0., .01):
                    positions = mf.mol.atom_coords()
                    positions[0, 0] += displacement
                    cpu.prepare_geometry(positions)
                    gpu.prepare_geometry(positions)
                    moved = transport(state, old_overlap, cpu.overlap)
                    assert_allclose(gpu.fock(moved.density()), cpu.fock(moved.density()), atol=2e-8, rtol=0)
                    self.assertAlmostEqual(gpu.energy(moved), cpu.energy(moved), delta=2e-8)
                    assert_allclose(gpu.forces(moved), cpu.forces(moved), atol=2e-8, rtol=0)
                    energy, force = gpu.evaluate(moved)
                    self.assertAlmostEqual(energy, cpu.energy(moved), delta=2e-8)
                    assert_allclose(force, cpu.forces(moved), atol=2e-8, rtol=0)
                    assert_allclose(cupy.asnumpy(gpu._gpu_mf.grids.coords), cpu.mf.grids.coords, atol=0, rtol=0)
                    energies = []
                    for sign in (1, -1):
                        trial = deepcopy(gpu)
                        shifted = positions.copy()
                        shifted[0, 2] += sign * 1e-4
                        trial.prepare_geometry(shifted)
                        energies.append(trial.energy(transport(moved, cpu.overlap, trial.overlap)))
                    self.assertAlmostEqual(gpu.forces(moved)[0, 2],
                                           -(energies[0] - energies[1]) / 2e-4, delta=3e-7)

    def test_pt_propagation_and_failed_step_restore(self):
        for mf in self.mean_fields:
            for variant in ("pt_cn", "pt_im"):
                with self.subTest(kind=type(mf).__name__, xc=mf.xc, variant=variant):
                    state = complex_state(PySCFProvider(mf))
                    options = CNOptions(variant, tolerance=1e-10)
                    cpu = LocalExecution([PySCFProvider(mf)], [state], alphas=[5.], options=options)
                    gpu = LocalExecution([GPU4PySCFProvider(mf)], [state], alphas=[5.], options=options)
                    gpu.observe(forces=False)
                    before = gpu.snapshot()
                    self.assertIsNone(before.providers[0]._gpu_mf)
                    end = gpu.positions.copy()
                    end[0, 0] += .002
                    with patch.object(GPU4PySCFProvider, "fock", side_effect=RuntimeError("device failure")):
                        with self.assertRaisesRegex(RuntimeError, "device failure"):
                            gpu.advance(end, .02, forces=False)
                    self.assertEqual(gpu.time, 0.)
                    assert_allclose(gpu.positions, before.providers[0].positions, atol=0, rtol=0)
                    assert_allclose(gpu.states[0].density(), state.density(), atol=0, rtol=0)
                    expected = cpu.advance(end, .02, forces=False)
                    for _ in range(2):
                        gpu.restore(before)
                        actual = gpu.advance(end, .02, forces=False)
                        assert_allclose(gpu.states[0].density(), cpu.states[0].density(), atol=2e-8, rtol=0)
                        assert_allclose(actual.observation.energies, expected.observation.energies, atol=2e-8, rtol=0)
                        self.assertLess(gpu.states[0].orthogonality_error(gpu.providers[0].overlap), 1e-10)
                        self.assertEqual(gpu.time, cpu.time)

    def test_two_state_dto_and_spin(self):
        for mf, variant in product((*self.mean_fields, lih_mean_field()),
                                   ("cn_accelerated", "pt_cn", "pt_im")):
            with self.subTest(kind=type(mf).__name__, xc=mf.xc, variant=variant):
                ground = PySCFProvider(mf).initial_state()
                if isinstance(mf, dft.uks.UKS):
                    excited = OrbitalState((ground.coefficients[0], mf.mo_coeff[1][:, 1:2]),
                                           ground.occupations)
                else:
                    nocc = len(ground.occupations[0])
                    excited = OrbitalState((mf.mo_coeff[:, :nocc + 1],),
                                           (np.r_[np.full(nocc - 1, 2.), 1., 1.],))
                runs = [LocalExecution([kind(mf), kind(mf)], [ground, excited], alphas=[5., 5.],
                                       options=CNOptions(variant, tolerance=1e-10))
                        for kind in (PySCFProvider, GPU4PySCFProvider)]
                methods = [MASH([np.sqrt(.5), 0., -np.sqrt(.5)]) for _ in runs]
                path = np.repeat(runs[0].positions[None], 3, axis=0)
                path[:, 0, 0] += np.arange(3) * 1e-4
                frames = [follow_path(run, path, [0., .01, .02], method=method)
                          for run, method in zip(runs, methods)]
                for left, right in zip(*frames):
                    assert_allclose(left.observation.energies, right.observation.energies, atol=2e-8, rtol=0)
                    assert_allclose(left.overlap_raw, right.overlap_raw, atol=2e-8, rtol=0)
                for left, right in zip(runs[0].states, runs[1].states):
                    assert_allclose(left.density(), right.density(), atol=2e-8, rtol=0)
                assert_allclose(methods[0].spin, methods[1].spin, atol=2e-8, rtol=0)
                self.assertEqual(methods[0].active_state, methods[1].active_state)
                assert_allclose(runs[0].observe().forces, runs[1].observe().forces, atol=2e-8, rtol=0)
                assert_allclose(ZVectorNAC().compute(runs[0]).raw, ZVectorNAC().compute(runs[1]).raw,
                                atol=2e-8, rtol=0)


if __name__ == "__main__":
    unittest.main()

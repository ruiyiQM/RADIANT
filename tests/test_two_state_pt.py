"""Public two-state PT, DTO phases and all-state rollback regressions."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose

from radiant.backends.pyscf import PySCFProvider
from radiant.backends.pyscf_nac import ZVectorNAC
from radiant.core.coupling import ZeroNACError
from radiant.core.execution import LocalExecution
from radiant.core.overlap import dto_pair, pair_overlap
from radiant.core.propagation import CNOptions
from radiant.core.state import NuclearState, OrbitalState
from radiant.driver import Trajectory, follow_path
from radiant.methods.mash import MASH
from test_gpu_provider import mean_field
from test_pt_mixed import lih_mean_field, lih_two_one_state
from toy_provider import ToyProvider


class TwoStatePTTests(unittest.TestCase):
    def test_each_state_is_validated_before_initialization_or_override(self):
        invalid = OrbitalState((np.eye(2),), ([1., 0.],))
        for variant in ("pt_cn", "pt_im"):
            options = CNOptions(variant)
            for branch in (0, 1):
                providers = [ToyProvider(), ToyProvider(1)]
                states = [p.initial_state() for p in providers]
                states[branch] = invalid
                with self.assertRaisesRegex(ValueError, "occupied"):
                    LocalExecution(providers, states, options=options)
                run = LocalExecution(providers)
                run.states = tuple(states)
                with self.assertRaisesRegex(ValueError, "occupied"):
                    run.advance([[.1, 0., 0.]], .1, options=options)
                self.assertEqual(run.time, 0.)
                assert_allclose(run.positions, [[0., 0., 0.]], atol=0, rtol=0)
                self.assertTrue(all(p.cache is None for p in providers))

    def test_pair_matches_separate_propagation_and_explicit_dto(self):
        for variant in ("pt_cn", "pt_im"):
            with self.subTest(variant=variant):
                options = CNOptions(variant, 1e-12, 120)
                providers = [ToyProvider(), ToyProvider(1)]
                run = LocalExecution(providers, alphas=[2., 3.], options=options, time=2.)
                before = run.snapshot()
                separate = [LocalExecution([deepcopy(p)], alphas=[alpha], options=options, time=2.)
                            for p, alpha in zip(providers, (2., 3.))]
                end = [[.01, 0., 0.]]
                expected = [one.advance(end, .1, n_substeps=2) for one in separate]
                actual = run.advance(end, .1, n_substeps=2)
                for state, one in zip(run.states, separate):
                    assert_allclose(state.density(), one.states[0].density(), atol=1e-13, rtol=0)
                cross = providers[0].cross_overlap(before.providers[0].mf.mol, run.providers[0].mf.mol)
                gauge = dto_pair(tuple(one.states[0] for one in separate), run.providers[0].overlap,
                                 reference=before.gauge, cross=cross)
                raw = pair_overlap(before.gauge, gauge, cross)
                assert_allclose(actual.overlap_raw, raw, atol=1e-13, rtol=0)
                energies = np.array([item.observation.energies[0] for item in expected])
                assert_allclose(actual.observation.energies, energies, atol=0, rtol=0)
                left, right = MASH([.6, 0., .8]), MASH([.6, 0., .8])
                initial = np.array([p.energy(s) for p, s in zip(before.providers, before.states)])
                left.propagate(initial, energies, raw, .1)
                right.propagate(initial, actual.observation.energies, actual.overlap_raw, .1)
                assert_allclose(left.spin, right.spin, atol=1e-13, rtol=0)
                self.assertEqual(run.time, 2.1)

    def test_second_state_and_late_substep_failure_restore_pair_and_retry(self):
        for variant in ("pt_cn", "pt_im"):
            for branch, threshold in ((1, .1), (0, .3)):
                with self.subTest(variant=variant, branch=branch):
                    options = CNOptions(variant, 1e-12, 120)
                    run = LocalExecution([ToyProvider(), ToyProvider(1)], options=options)
                    run.providers[branch].failure_time = threshold
                    before = run.snapshot()
                    with self.assertRaisesRegex(RuntimeError, "Injected"):
                        run.advance([[.2, 0., 0.]], .4, n_substeps=4)
                    self.assertEqual(run.time, before.time)
                    assert_allclose(run.positions, before.providers[0].positions, atol=0, rtol=0)
                    for state, saved in zip(run.states, before.states):
                        assert_allclose(state.density(), saved.density(), atol=0, rtol=0)
                    for state, saved in zip(run.gauge, before.gauge):
                        assert_allclose(state.coefficients[0], saved.coefficients[0], atol=0, rtol=0)
                    self.assertTrue(all(p.cache is None for p in run.providers))
                    run.providers[branch].failure_time = None
                    clean = LocalExecution([ToyProvider(), ToyProvider(1)], options=options)
                    expected = clean.advance([[.2, 0., 0.]], .4, n_substeps=4)
                    retried = run.advance([[.2, 0., 0.]], .4, n_substeps=4)
                    assert_allclose(retried.overlap_raw, expected.overlap_raw, atol=1e-13, rtol=0)
                    assert_allclose(retried.observation.forces, expected.observation.forces, atol=0, rtol=0)

    def test_reversible_hop_and_failed_nac_restore_the_whole_step(self):
        spin = np.array([np.sqrt(1 - .001**2), 0., .001])
        for variant in ("pt_cn", "pt_im"):
            for fail in (False, True):
                def direction(execution):
                    if fail:
                        raise ZeroNACError("Injected zero NAC at the located crossing")
                    return SimpleNamespace(raw=np.array([[1., 0., 0.]]))
                run = Trajectory(LocalExecution([ToyProvider(0, force=.1), ToyProvider(1, force=.2)],
                                                 options=CNOptions(variant, 1e-12, 120)),
                                 NuclearState([[0., 0., 0.]], [[1., 0., 0.]], [2.]),
                                 method=MASH(spin), nac=SimpleNamespace(compute=direction), hop_tolerance=1e-4)
                before = run.execution.snapshot()
                initial_energy = run.observation.energies[1] + run.nuclei.kinetic_energy
                with self.subTest(variant=variant, failed_nac=fail):
                    if fail:
                        with self.assertRaises(ZeroNACError):
                            run.step(.4)
                        self.assertEqual(run.time, 0.)
                        self.assertEqual(run.execution.time, 0.)
                        self.assertEqual(run.frames, [])
                        assert_allclose(run.nuclei.positions, [[0., 0., 0.]], atol=0, rtol=0)
                        assert_allclose(run.nuclei.momenta, [[1., 0., 0.]], atol=0, rtol=0)
                        assert_allclose(run.method.spin, spin, atol=1e-15, rtol=0)
                        for state, saved in zip(run.execution.states, before.states):
                            assert_allclose(state.density(), saved.density(), atol=0, rtol=0)
                        self.assertTrue(all(p.cache is None for p in run.execution.providers))
                    else:
                        frame = run.step(.4)
                        self.assertEqual(run.active_state, 0)
                        self.assertEqual(run.time, .4)
                        self.assertAlmostEqual(frame["total_energy"], initial_energy, places=12)
                    event = run.hop_events[0]
                    self.assertGreater(event[1], 0.)
                    self.assertLess(event[1], .4)
                    self.assertEqual(event[4], "zero_nac" if fail else "accepted")


class MolecularPairPTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rks = PySCFProvider(lih_mean_field())
        uks = PySCFProvider(mean_field(True))
        excited = OrbitalState((uks.mf.mo_coeff[0][:, :2], uks.mf.mo_coeff[1][:, 1:2]),
                               uks.initial_state().occupations)
        cls.cases = ((rks, (rks.initial_state(), lih_two_one_state(rks))),
                     (uks, (uks.initial_state(), excited)))

    def path(self, provider, states, variant, steps):
        run = LocalExecution([deepcopy(provider), deepcopy(provider)], states,
                             alphas=[2., 2.], options=CNOptions(variant, 1e-12, 120))
        positions = np.repeat(run.positions[None], steps + 1, axis=0)
        positions[:, -1, 2] += np.linspace(0., .002, steps + 1)
        method = MASH([.6, 0., .8])
        frames = follow_path(run, positions, np.linspace(0., .04, steps + 1), method=method)
        return run, method, frames

    def test_moving_pairs_converge_to_cn_in_density_dto_and_spin(self):
        for provider, states in self.cases:
            reference, ref_method, _ = self.path(provider, states, "cn_accelerated", 32)
            for variant in ("pt_cn", "pt_im"):
                errors = []
                for steps in (4, 8):
                    run, method, frames = self.path(provider, states, variant, steps)
                    error = max(np.linalg.norm(a.density() - b.density())
                                for a, b in zip(run.states, reference.states))
                    errors.append(error)
                    with self.subTest(unrestricted=provider.unrestricted, variant=variant, steps=steps):
                        self.assertLess(error, 2e-5)
                        self.assertAlmostEqual(method.spin[2], ref_method.spin[2], delta=2e-5)
                        for state in run.states:
                            self.assertLess(state.orthogonality_error(run.providers[0].overlap), 1e-10)
                        self.assertTrue(all(frame.overlap_raw.shape == (2, 2) for frame in frames))
                        assert_allclose(run.observe().forces, reference.observe().forces, atol=2e-5, rtol=0)
                self.assertLess(errors[1], .4 * errors[0])

    def test_state_phases_preserve_populations_and_transform_dto_amplitudes(self):
        for provider, states in self.cases:
            rng = np.random.default_rng(17)
            phased = tuple(OrbitalState(tuple(c * np.exp(1j * rng.uniform(-2., 2., c.shape[1]))
                                               for c in state.coefficients), state.occupations)
                           for state in states)
            for variant in ("pt_cn", "pt_im"):
                run, method, _ = self.path(provider, states, variant, 4)
                other = LocalExecution([deepcopy(provider), deepcopy(provider)], phased,
                                       alphas=[2., 2.], options=CNOptions(variant, 1e-12, 120))
                initial = dto_pair(states, provider.overlap)
                phases = np.diag(pair_overlap(initial, other.gauge, provider.overlap))
                phases = phases / np.abs(phases)
                transformed = MASH([.6, 0., .8])
                transformed.coefficients *= phases.conj()
                positions = np.repeat(other.positions[None], 5, axis=0)
                positions[:, -1, 2] += np.linspace(0., .002, 5)
                follow_path(other, positions, np.linspace(0., .04, 5), method=transformed)
                with self.subTest(unrestricted=provider.unrestricted, variant=variant):
                    for left, right in zip(run.states, other.states):
                        assert_allclose(left.density(), right.density(), atol=1e-11, rtol=0)
                    final = np.diag(pair_overlap(run.gauge, other.gauge, run.providers[0].overlap))
                    final = final / np.abs(final)
                    assert_allclose(transformed.coefficients, final.conj() * method.coefficients,
                                    atol=1e-10, rtol=0)
                    self.assertAlmostEqual(transformed.spin[2], method.spin[2], delta=1e-10)

    def test_static_nac_remains_consistent_after_pt_propagation(self):
        for provider, states in self.cases:
            reference, _, _ = self.path(provider, states, "cn_accelerated", 16)
            expected = ZVectorNAC().compute(reference)
            for variant in ("pt_cn", "pt_im"):
                run, _, _ = self.path(provider, states, variant, 8)
                result = ZVectorNAC().compute(run)
                with self.subTest(unrestricted=provider.unrestricted, variant=variant):
                    self.assertTrue(np.isfinite(result.raw).all())
                    axes = [np.linalg.svd(np.array([r.raw.real.ravel(), r.raw.imag.ravel()]),
                                          full_matrices=False)[2][0] for r in (expected, result)]
                    self.assertGreater(abs(np.dot(*axes)), .9999)


if __name__ == "__main__":
    unittest.main()

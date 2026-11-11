# MIT License
#
# Copyright (c) 2026 Ruiyi Zhou and Jeremy O. Richardson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) 2026 Ruiyi Zhou
"""Frozen PT equation, independent density dynamics and atomic local execution."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from radiant.backends.pyscf import PySCFProvider
from radiant.core.coupling import PaperFD
from radiant.core.execution import LocalExecution
from radiant.core.propagation import CNOptions, StepFailure, crank_nicolson
from radiant.core.state import NuclearState, OrbitalState
from radiant.core.transport import lowdin
from radiant.driver import Trajectory
from test_pyscf import mean_field
from toy_provider import ToyProvider


OPTIONS = CNOptions('pt_cn', 1e-12, 120)
FIXTURE = json.loads((Path(__file__).parent/'fixtures'/'pt_cn.json').read_text())


def decode(value):
    return np.array(value['real']) + 1j*np.array(value['imag'])


def model(case):
    overlap = decode(case['overlap'])
    state = OrbitalState(tuple(decode(c) for c in case['initial']), tuple(case['occupations']))
    h = decode(case['hamiltonian'])

    def fock(density, time):
        if case['name'] == 'restricted_nonlinear':
            return h + .1*overlap@np.diag(density.diagonal().real)@overlap
        if case['name'] == 'unrestricted_nonlinear':
            return h + np.array([.2*overlap@(density[s].real + .7*density[1-s].real)@overlap
                                 for s in (0, 1)])
        return h
    return state, overlap, fock


class PTCNTests(unittest.TestCase):
    def test_frozen_source_coefficients_and_raw_residual(self):
        for case in FIXTURE['cases']:
            with self.subTest(case=case['name']):
                initial, overlap, fock = model(case)
                final, diagnostic = crank_nicolson(initial, overlap, fock, 0., case['dt'],
                                                   alpha=case['alpha'], options=OPTIONS)
                for actual, expected in zip(final.coefficients, case['final']):
                    assert_allclose(actual, decode(expected), atol=1e-12, rtol=0)
                self.assertLessEqual(diagnostic.residual, OPTIONS.tolerance)
                self.assertLess(diagnostic.orthogonality_error, 1e-12)
                # The source checks the raw trial, then orthonormalizes it.
                def action(state):
                    h = np.asarray(fock(state.density(), 0.))*case['alpha']
                    hs = (h,) if len(state.coefficients) == 1 else h
                    return [np.linalg.solve(overlap, v@c)-c@(c.conj().T@v@c)
                            for c, v in zip(state.coefficients, hs)]
                residuals = [new-old+.5j*case['dt']*(a+b) for new, old, a, b in
                             zip(final.coefficients, initial.coefficients, action(final), action(initial))]
                self.assertAlmostEqual(diagnostic.residual, case['residual'], delta=1e-12)
                self.assertAlmostEqual(np.linalg.norm([np.linalg.norm(r) for r in residuals]),
                                       case['accepted_residual'], delta=1e-12)

    def test_density_second_order_against_exact_schrodinger_dynamics(self):
        for unrestricted in (False, True):
            state, overlap, _ = model(FIXTURE['cases'][int(unrestricted)])
            h = decode(FIXTURE['cases'][int(unrestricted)]['hamiltonian'])
            hs = h if unrestricted else (h,)
            for alpha in (1., 8.):
                exact = OrbitalState(tuple(expm(-1j*np.linalg.solve(overlap, v)*.4)@c
                                           for v, c in zip(hs, state.coefficients)), state.occupations)
                errors = []
                for scaled_dt in (.1, .05, .025):
                    candidate = state.copy()
                    for k in range(round(.4/scaled_dt)):
                        candidate, _ = crank_nicolson(candidate, overlap, lambda d,t: h,
                            k*scaled_dt/alpha, scaled_dt/alpha, alpha=alpha, options=OPTIONS)
                    errors.append(np.linalg.norm(candidate.density()-exact.density()))
                    for c, occ in zip(candidate.coefficients, candidate.occupations):
                        self.assertAlmostEqual(np.trace((c*occ)@c.conj().T@overlap).real,
                                               sum(occ), places=11)
                with self.subTest(unrestricted=unrestricted, alpha=alpha, errors=errors):
                    self.assertGreater(errors[0]/errors[1], 3.7)
                    self.assertGreater(errors[1]/errors[2], 3.7)

    def test_nonlinear_joint_density_against_independent_ode(self):
        for case in FIXTURE['cases'][:2]:
            initial, overlap, fock = model(case)
            sizes = np.cumsum([0]+[c.size for c in initial.coefficients])
            def unpack(vector):
                return OrbitalState(tuple(vector[sizes[i]:sizes[i+1]].reshape(c.shape)
                                          for i, c in enumerate(initial.coefficients)), initial.occupations)
            def derivative(time, vector):
                state = unpack(vector)
                h = np.asarray(fock(state.density(), time))
                hs = (h,) if len(state.coefficients) == 1 else h
                return np.concatenate([(-1j*np.linalg.solve(overlap, v@c)).ravel()
                                       for v, c in zip(hs, state.coefficients)])
            reference = solve_ivp(derivative, (0., .2), np.concatenate([c.ravel() for c in initial.coefficients]),
                                  method='DOP853', rtol=1e-12, atol=1e-14)
            self.assertTrue(reference.success)
            errors = []
            for dt in (.02, .01):
                candidate = initial.copy()
                for k in range(round(.2/dt)):
                    candidate, _ = crank_nicolson(candidate, overlap, fock, k*dt, dt, options=OPTIONS)
                errors.append(np.linalg.norm(candidate.density()-unpack(reference.y[:, -1]).density()))
            with self.subTest(case=case['name'], errors=errors):
                self.assertLess(errors[-1], 3e-6)
                self.assertGreater(errors[0]/errors[1], 3.7)

    def test_stationary_subspaces_remove_fast_phase_and_rotations_are_covariant(self):
        for case in FIXTURE['cases'][:2]:
            initial, overlap, fock = model(case)
            inverse, _ = lowdin(overlap)
            h = decode(case['hamiltonian'])
            hs = (h,) if len(initial.coefficients) == 1 else h
            stationary = OrbitalState(tuple(inverse@np.linalg.eigh(inverse@v@inverse)[1][:, :c.shape[1]]
                                           for v, c in zip(hs, initial.coefficients)), initial.occupations)
            final, diagnostic = crank_nicolson(stationary, overlap, lambda d,t: h, 0., 2.,
                                               alpha=80., options=OPTIONS)
            self.assertEqual(diagnostic.iterations, 1)
            for actual, expected in zip(final.coefficients, stationary.coefficients):
                assert_allclose(actual, expected, atol=2e-13, rtol=0)
            rotated = initial.copy()
            for i, c in enumerate(rotated.coefficients):
                n = c.shape[1]
                generator = np.arange(n*n).reshape(n, n)
                rotated.coefficients[i][:] = c@expm(.2j*(generator+generator.T))
            a, _ = crank_nicolson(initial, overlap, fock, 0., .03, options=OPTIONS)
            b, _ = crank_nicolson(rotated, overlap, fock, 0., .03, options=OPTIONS)
            assert_allclose(a.density(), b.density(), atol=1e-12, rtol=0)

    def test_empty_and_unequal_spin_populations_in_both_orders(self):
        initial, overlap, fock = model(FIXTURE['cases'][2])
        h = fock(initial.density(), 0.)
        inverse, _ = lowdin(overlap)
        for populations in ((0,1), (1,0), (1,2), (2,1)):
            state = OrbitalState(tuple(inverse[:, :n] for n in populations), tuple(np.ones(n) for n in populations))
            final, _ = crank_nicolson(state, overlap, lambda d,t: h, 0., .02, options=OPTIONS)
            expected = OrbitalState(tuple(expm(-.02j*np.linalg.solve(overlap, v))@c
                                         for c,v in zip(state.coefficients, h)), state.occupations)
            with self.subTest(populations=populations):
                assert_allclose(final.density(), expected.density(), atol=5e-7, rtol=0)
                self.assertEqual(tuple(c.shape[1] for c in final.coefficients), populations)

    def test_signed_steps_physical_times_and_autonomous_alpha_dt(self):
        initial, overlap, fock = model(FIXTURE['cases'][1])
        a, _ = crank_nicolson(initial, overlap, fock, 7., .02, alpha=8., options=OPTIONS)
        b, _ = crank_nicolson(initial, overlap, fock, 7., .16, options=OPTIONS)
        assert_allclose(a.density(), b.density(), atol=1e-12, rtol=0)
        forward, _ = crank_nicolson(initial, overlap, fock, 7., .002, alpha=8., options=OPTIONS)
        reverse, _ = crank_nicolson(forward, overlap, fock, 7.002, -.002, alpha=8., options=OPTIONS)
        assert_allclose(reverse.density(), initial.density(), atol=2e-9, rtol=0)
        h = decode(FIXTURE['cases'][1]['hamiltonian'])
        seen = []
        def field(density, time):
            seen.append(time)
            return h*(1+time)
        candidate = initial.copy()
        for k in range(20):
            candidate, _ = crank_nicolson(candidate, overlap, field, k*.001, .001, alpha=3., options=OPTIONS)
        exact = OrbitalState(tuple(expm(-3j*np.linalg.solve(overlap, v)*(.02+.5*.02**2))@c
                                   for v,c in zip(h, initial.coefficients)), initial.occupations)
        assert_allclose(candidate.density(), exact.density(), atol=1e-8, rtol=0)
        self.assertEqual(min(seen), 0.)
        self.assertAlmostEqual(max(seen), .02)

    def test_raw_residual_cannot_be_hidden_by_zero_preconditioned_update(self):
        state, overlap, fock = model(FIXTURE['cases'][0])
        with patch('radiant.core.propagation._pt_preconditioner', return_value=lambda r: np.zeros_like(r)):
            with self.assertRaises(StepFailure) as failure:
                crank_nicolson(state, overlap, fock, 0., .1, options=CNOptions('pt_cn', 1e-12, 3))
        self.assertGreater(failure.exception.residual, 1e-3)
        self.assertEqual(failure.exception.iterations, 3)

    def test_initial_invalid_fock_reports_zero_iterations(self):
        state = OrbitalState((np.eye(2)[:, :1],), ([2],))
        before = state.coefficients[0].copy()
        with self.assertRaises(StepFailure) as failure:
            crank_nicolson(state, np.eye(2), lambda d, t: np.full((2, 2), np.nan),
                           0., .1, options=OPTIONS)
        self.assertEqual(str(failure.exception), 'Fock provider returned invalid/non-Hermitian matrices')
        self.assertEqual(failure.exception.iterations, 0)
        self.assertEqual(failure.exception.residual, np.inf)
        self.assertIsNone(failure.exception.__cause__)
        assert_array_equal(state.coefficients[0], before)

    def test_invalid_fock_reports_first_and_later_attempted_iteration(self):
        for call, expected in ((2, 1), (4, 3)):
            with self.subTest(call=call):
                state = OrbitalState((np.eye(2)[:, :1],), ([2],))
                before = state.coefficients[0].copy()
                calls = []
                def fock(density, time):
                    calls.append(time)
                    return np.full((2, 2), np.nan) if len(calls) == call else np.array([[0., .2], [.2, .1]])
                with self.assertRaises(StepFailure) as failure:
                    crank_nicolson(state, np.eye(2), fock, 0., .1,
                                   options=CNOptions('pt_cn', 1e-30, 10))
                self.assertEqual(len(calls), call)
                self.assertEqual(str(failure.exception), 'Fock provider returned invalid/non-Hermitian matrices')
                self.assertEqual(failure.exception.residual, np.inf)
                self.assertIsNone(failure.exception.__cause__)
                assert_array_equal(state.coefficients[0], before)
                self.assertEqual(failure.exception.iterations, expected)

    def test_provider_stepfailure_preserves_identity_residual_and_cause(self):
        for fail_call in (1, 2):
            for iterations in (0, 37):
                with self.subTest(fail_call=fail_call, iterations=iterations):
                    state = OrbitalState((np.eye(2)[:, :1],), ([2],))
                    before = state.coefficients[0].copy()
                    cause = ValueError('underlying failure')
                    original = StepFailure('provider failure', residual=.25, iterations=iterations)
                    calls = []
                    def fock(density, time):
                        calls.append(time)
                        if len(calls) == fail_call:
                            raise original from cause
                        return np.array([[0., .2], [.2, .1]])
                    with self.assertRaises(StepFailure) as failure:
                        crank_nicolson(state, np.eye(2), fock, 0., .1, options=OPTIONS)
                    self.assertIs(failure.exception, original)
                    self.assertIs(type(failure.exception), StepFailure)
                    self.assertEqual(str(failure.exception), 'provider failure')
                    self.assertEqual(failure.exception.residual, .25)
                    self.assertIs(failure.exception.__cause__, cause)
                    self.assertEqual(failure.exception.iterations, iterations)
                    self.assertEqual(len(calls), fail_call)
                    assert_array_equal(state.coefficients[0], before)

    def test_invalid_fock_at_accepted_validation_reports_last_iteration(self):
        state = OrbitalState((np.eye(2)[:, :1],), ([2],))
        before = state.coefficients[0].copy()
        h = np.array([[0., .2], [.2, .1]])
        _, diagnostic = crank_nicolson(state, np.eye(2), lambda d, t: h, 0., .1, options=OPTIONS)
        self.assertGreater(diagnostic.iterations, 1)
        calls = []
        def fock(density, time):
            calls.append(time)
            return np.full((2, 2), np.nan) if len(calls) == diagnostic.iterations + 2 else h
        with self.assertRaises(StepFailure) as failure:
            crank_nicolson(state, np.eye(2), fock, 0., .1, options=OPTIONS)
        self.assertEqual(len(calls), diagnostic.iterations + 2)
        self.assertEqual(str(failure.exception), 'Fock provider returned invalid/non-Hermitian matrices')
        self.assertEqual(failure.exception.iterations, diagnostic.iterations)
        self.assertEqual(failure.exception.residual, np.inf)
        self.assertIsNone(failure.exception.__cause__)
        assert_array_equal(state.coefficients[0], before)

    def test_in_loop_failure_reports_iteration_and_rolls_back_trajectory(self):
        class InvalidSecondFock(ToyProvider):
            def __init__(self):
                super().__init__(force=.1)
                self.calls = 0

            def fock(self, density, time):
                self.calls += 1
                value = super().fock(density, time)
                return np.full_like(value, np.nan) if self.calls == 2 else value

        state = OrbitalState((np.eye(2)[:, :1],), ([2],))
        run = Trajectory(LocalExecution([InvalidSecondFock()], [state], options=OPTIONS),
                         NuclearState([[0, 0, 0]], [[.1, 0, 0]], [2.]))
        before = deepcopy(run.nuclei)
        with self.assertRaises(StepFailure) as failure:
            run.step(.1, n_substeps=1)
        self.assertEqual(failure.exception.iterations, 1)
        self.assertEqual(failure.exception.residual, np.inf)
        self.assertEqual(run.time, 0.)
        self.assertEqual(run.execution.time, 0.)
        assert_array_equal(run.nuclei.positions, before.positions)
        assert_array_equal(run.nuclei.momenta, before.momenta)
        assert_array_equal(run.execution.positions, before.positions)
        assert_array_equal(run.execution.states[0].coefficients[0], state.coefficients[0])
        self.assertIsNone(run.execution.providers[0].cache)
        self.assertEqual(run.execution.providers[0].calls, 0)

    def test_unsupported_occupations_pairs_and_fd_fail_before_advancing(self):
        inverse = np.eye(2, dtype=complex)
        # Restricted 2/1 groups are accepted (test_pt_mixed); virtual columns are not.
        for occ in ([2,0], [1,0], []):
            state = OrbitalState((inverse[:, :len(occ)],), (occ,))
            with self.assertRaises(ValueError):
                crank_nicolson(state, np.eye(2), lambda d,t: self.fail('Fock must not be called'),
                               0., .1, options=OPTIONS)
        with self.assertRaises(ValueError):
            LocalExecution([ToyProvider(), ToyProvider(1)], options=OPTIONS)
        pair = LocalExecution([ToyProvider(), ToyProvider(1)])
        with self.assertRaises(ValueError):
            pair.advance([[.1,0,0]], .1, options=OPTIONS)
        self.assertEqual(pair.time, 0.)
        self.assertTrue(all(p.cache is None for p in pair.providers))
        with self.assertRaises(ValueError):
            PaperFD(options=OPTIONS)

    def test_failures_restore_nuclei_geometry_both_spins_and_caches(self):
        for failure in ('limit', 'backend', 'beta_nonfinite', 'final_fock', 'late_substep'):
            provider = ToyProvider(force=.1)
            base_fock = provider.fock
            calls = [0]
            def fock(density, time):
                calls[0] += 1
                value = base_fock(density.sum(axis=0), time)
                result = np.stack((value, value.conj()))
                if failure == 'backend':
                    raise RuntimeError('Injected provider failure')
                if failure == 'beta_nonfinite' or (failure == 'final_fock' and calls[0] == 3):
                    result[1,0,0] = np.nan
                return result
            provider.fock = fock
            if failure == 'late_substep':
                provider.failure_time = .2
            state = OrbitalState((np.eye(2)[:, :1], np.eye(2)[:, :0]), ([1], []))
            options = CNOptions('pt_cn', 1e-12, 1) if failure == 'limit' else (
                CNOptions('pt_cn', 1., 120) if failure == 'final_fock' else OPTIONS)
            run = Trajectory(LocalExecution([provider], [state], options=options),
                             NuclearState([[0,0,0]], [[.1,0,0]], [2.]))
            old_nuclei = deepcopy(run.nuclei)
            with self.subTest(failure=failure), self.assertRaises((RuntimeError, StepFailure)):
                run.step(.4, n_substeps=4)
            self.assertEqual(run.time, 0.)
            self.assertEqual(run.execution.time, 0.)
            assert_array_equal(run.nuclei.positions, old_nuclei.positions)
            assert_array_equal(run.nuclei.momenta, old_nuclei.momenta)
            assert_array_equal(run.execution.positions, old_nuclei.positions)
            self.assertIsNone(run.execution.providers[0].cache)
            for actual, expected in zip(run.execution.states[0].coefficients, state.coefficients):
                assert_array_equal(actual, expected)

    def test_native_pyscf_moving_basis_matches_small_step_orbital_cn(self):
        for kind in ('rhf', 'rks', 'uhf', 'uks'):
            provider = PySCFProvider(mean_field(kind))
            inverse, _ = lowdin(provider.overlap)
            n = inverse.shape[0]
            generator = np.zeros((n,n), complex)
            generator[0,-1] = .06+.04j
            generator[-1,0] = -generator[0,-1].conjugate()
            occupied = provider.initial_state()
            state = OrbitalState(tuple(inverse@expm(generator)[:, :c.shape[1]] for c in occupied.coefficients),
                                 occupied.occupations)
            end = provider.positions.copy()
            end[1,2] += .002
            observations = []
            for variant in ('pt_cn', 'cn_accelerated'):
                run = LocalExecution([deepcopy(provider)], [state], alphas=[2.],
                                     options=CNOptions(variant, 1e-12, 120), time=3.)
                before = run.snapshot()
                frame = run.advance(end, .02, n_substeps=20)
                first = run.states[0].density()
                run.restore(before)
                repeated = run.advance(end, .02, n_substeps=20)
                assert_allclose(run.states[0].density(), first, atol=1e-12, rtol=0)
                assert_allclose(frame.observation.forces, repeated.observation.forces, atol=1e-12, rtol=0)
                self.assertAlmostEqual(run.time, 3.02)
                observations.append((first, frame.observation))
            with self.subTest(kind=kind):
                assert_allclose(observations[0][0], observations[1][0], atol=2e-7, rtol=0)
                assert_allclose(observations[0][1].energies, observations[1][1].energies, atol=2e-8, rtol=0)
                assert_allclose(observations[0][1].forces, observations[1][1].forces, atol=2e-7, rtol=0)

    def test_short_ehrenfest_trajectory_has_second_order_nuclear_convergence(self):
        provider = PySCFProvider(mean_field('rhf'))
        nuclei = NuclearState(provider.positions, [[.1,0,.2], [-.1,0,-.2]], [1836.,1836.])
        positions = []
        for dt in (.2, .1, .05, .0125):
            run = Trajectory(LocalExecution([deepcopy(provider)], options=OPTIONS), nuclei)
            run.run(round(.8/dt), dt)
            self.assertEqual(run.time, .8)
            positions.append(run.nuclei.positions.copy())
        errors = [np.linalg.norm(p-positions[-1]) for p in positions[:-1]]
        self.assertGreater(errors[0]/errors[1], 3.5, errors)
        self.assertGreater(errors[1]/errors[2], 3.5, errors)


def projection_cases():
    """Restricted 2/1, restricted closed-shell and unrestricted states, all far from stationary."""
    from test_pt_mixed import mean_field_fock, random_model
    state, overlap, core = random_model()
    restricted = mean_field_fock(overlap, core)
    closed = OrbitalState((state.coefficients[0][:, :2],), ([2, 2],))
    inverse, _ = lowdin(overlap)
    q = np.linalg.qr(np.random.default_rng(3).normal(size=(6, 6)) + 0j)[0]
    unrestricted = OrbitalState((inverse @ q[:, :3], inverse @ q[:, 3:5]), ([1, 1, 1], [1, 1]))

    def spin_fock(density, time):
        return np.array([core + .1 * overlap @ (density[s].real + .5 * density[1 - s].real) @ overlap
                         for s in (0, 1)])
    return [('restricted_2_1', state, overlap, restricted), ('restricted_closed', closed, overlap, restricted),
            ('unrestricted', unrestricted, overlap, spin_fock)]


def pt_cn_residual(state, trial, overlap, fock, dt):
    """Independent trapezoidal PT-CN residual; groups are (spin, occupation) column sets."""
    def focks(s):
        value = np.asarray(fock(s.density(), 0.))
        return value[None] if value.ndim == 2 else value
    start, final = focks(state), focks(trial)
    errors = []
    for spin, (c0, c1, f) in enumerate(zip(state.coefficients, trial.coefficients, state.occupations)):
        for value in np.unique(f):
            columns = np.flatnonzero(np.asarray(f) == value)

            def action(c, h):
                hc = h @ c
                return np.linalg.solve(overlap, hc) - c @ (c.conj().T @ hc)
            a, b = c0[:, columns], c1[:, columns]
            errors.append(b - a + .5j * dt * (action(a, start[spin]) + action(b, final[spin])))
    return float(np.linalg.norm([np.linalg.norm(e) for e in errors]))


class ProjectionDiagnosticsTests(unittest.TestCase):
    def test_raw_gram_defect_and_returned_residual_match_independent_root(self):
        from scipy.optimize import root
        dt = .5
        for name, state, overlap, fock in projection_cases():
            with self.subTest(case=name):
                returned, diagnostic = crank_nicolson(state, overlap, fock, 0., dt, options=OPTIONS)
                shapes = [c.shape for c in state.coefficients]
                sizes = np.cumsum([0] + [int(np.prod(s)) for s in shapes])

                def unpack(vector):
                    z = vector[:sizes[-1]] + 1j * vector[sizes[-1]:]
                    return OrbitalState(tuple(z[sizes[i]:sizes[i + 1]].reshape(s) for i, s in enumerate(shapes)),
                                        state.occupations)

                def equations(vector):
                    trial = unpack(vector)
                    start = np.asarray(fock(state.density(), 0.))
                    final = np.asarray(fock(trial.density(), 0.))
                    start, final = (start[None], final[None]) if start.ndim == 2 else (start, final)
                    parts = []
                    for spin, (c0, c1, f) in enumerate(zip(state.coefficients, trial.coefficients,
                                                           state.occupations)):
                        out = np.empty_like(c1)
                        for value in np.unique(f):
                            cols = np.flatnonzero(np.asarray(f) == value)
                            a, b = c0[:, cols], c1[:, cols]
                            act = [np.linalg.solve(overlap, h @ c) - c @ (c.conj().T @ h @ c)
                                   for c, h in ((a, start[spin]), (b, final[spin]))]
                            out[:, cols] = b - a + .5j * dt * (act[0] + act[1])
                        parts.append(out.ravel())
                    z = np.concatenate(parts)
                    return np.concatenate((z.real, z.imag))

                z0 = np.concatenate([c.ravel() for c in state.coefficients])
                solution = root(equations, np.concatenate((z0.real, z0.imag)), method='hybr', tol=1e-13)
                raw = unpack(solution.x)
                self.assertLess(np.max(np.abs(equations(solution.x))), 1e-11)
                independent_gram = raw.orthogonality_error(overlap)
                # Non-vacuous: the trapezoidal root really leaves the orbital manifold here.
                self.assertGreater(independent_gram, 1e-3)
                self.assertAlmostEqual(diagnostic.raw_orthogonality_error, independent_gram, delta=1e-9)
                self.assertLess(diagnostic.orthogonality_error, 1e-12)
                independent_residual = pt_cn_residual(state, returned, overlap, fock, dt)
                assert_allclose(diagnostic.returned_residual, independent_residual, rtol=1e-8, atol=1e-13)
                self.assertGreater(diagnostic.returned_residual, 1e3 * diagnostic.residual)

    def test_stationary_state_has_no_projection_defect(self):
        name, state, overlap, fock = projection_cases()[1]
        core = fock(state.density(), 0.)
        inverse, _ = lowdin(overlap)
        vectors = np.linalg.eigh(inverse @ core @ inverse)[1]
        eigen = OrbitalState((inverse @ vectors[:, :2],), ([2, 2],))
        for alpha in (1., 40.):
            with self.subTest(alpha=alpha):
                _, diagnostic = crank_nicolson(eigen, overlap, lambda d, t: core, 0., .5,
                                               alpha=alpha, options=OPTIONS)
                self.assertLess(diagnostic.raw_orthogonality_error, 1e-12)
                self.assertLessEqual(diagnostic.returned_residual, OPTIONS.tolerance)

    def test_cn_variants_return_the_root_so_raw_and_returned_coincide(self):
        name, state, overlap, fock = projection_cases()[0]
        for variant in ('cn', 'cn_accelerated'):
            with self.subTest(variant=variant):
                _, diagnostic = crank_nicolson(state, overlap, fock, 0., .1,
                                               options=CNOptions(variant, 1e-12, 200))
                self.assertEqual(diagnostic.raw_orthogonality_error, diagnostic.orthogonality_error)
                self.assertEqual(diagnostic.returned_residual, diagnostic.residual)


if __name__ == '__main__':
    unittest.main()

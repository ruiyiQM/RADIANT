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
"""CN and Lowdin checks independent of an electronic-structure engine."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from radiant.core.propagation import CNOptions, StepFailure, crank_nicolson
from radiant.core.state import NuclearState, OrbitalState
from radiant.core.transport import lowdin, transport


class PropagationTests(unittest.TestCase):
    def test_both_variants_match_explicit_cayley_and_signed_reverse(self):
        s = np.array([[1., .2], [.2, 1.3]])
        inv, _ = lowdin(s)
        h = np.array([[.2, .1j], [-.1j, -.3]])
        state = OrbitalState((inv[:, :1],), ([2],))
        for variant in ("cn", "cn_accelerated"):
            for dt in (.2, -.2):
                final, diagnostic = crank_nicolson(state, s, lambda dm, t: h, 2., dt,
                    alpha=3., options=CNOptions(variant, 1e-12))
                expected = np.linalg.solve(s + .5j * dt * 3 * h,
                                          (s - .5j * dt * 3 * h) @ state.coefficients[0])
                assert_allclose(final.coefficients[0], expected, atol=1e-12, rtol=0)
                self.assertLess(diagnostic.residual, 1e-12)
                self.assertLess(diagnostic.orthogonality_error, 1e-12)
                reverse, _ = crank_nicolson(final, s, lambda dm, t: h, 2.+dt, -dt,
                    alpha=3., options=CNOptions(variant, 1e-12))
                assert_allclose(reverse.coefficients[0], state.coefficients[0], atol=1e-12, rtol=0)

    def test_nonlinear_ao_residual_and_physical_clock(self):
        initial = np.array([[np.cos(1e-3)], [np.sin(1e-3)]], complex)
        state = OrbitalState((initial,), ([2],))
        for variant in ("cn", "cn_accelerated"):
            for dt in (4., -4.):
                times = []
                def fock(dm, t):
                    times.append(t)
                    return np.diag([-1., -.1]) + .6 * dm.real
                final, diag = crank_nicolson(state, np.eye(2), fock, 7., dt,
                    options=CNOptions(variant, 1e-10, 200))
                h = (fock(state.density(), 7.) + fock(final.density(), 7.+dt))/2
                assert_allclose((np.eye(2)+.5j*dt*h)@final.coefficients[0],
                                (np.eye(2)-.5j*dt*h)@initial, atol=1e-10, rtol=0)
                self.assertEqual(set(times), {7., 7.+dt})
                self.assertLess(diag.residual, 1e-10)

    def test_second_order_density_against_exact_linear_evolution(self):
        h = np.array([[.3, .2j], [-.2j, -.1]])
        state = OrbitalState((np.array([[1.], [0.]], complex),), ([2],))
        exact = expm(-1j*h) @ state.coefficients[0]
        errors = []
        for n in (10, 20, 40):
            candidate = state
            for k in range(n):
                candidate, _ = crank_nicolson(candidate, np.eye(2), lambda d,t: h, k/n, 1/n)
            errors.append(np.linalg.norm(candidate.density()-2*exact@exact.conj().T))
        assert_allclose(np.array(errors[:-1])/errors[1:], 4., rtol=.003)

    def test_unrestricted_joint_trials_empty_and_unequal_populations(self):
        for counts in ((1, 0), (2, 1), (0, 0)):
            state = OrbitalState(tuple(np.eye(3, dtype=complex)[:, :n] for n in counts),
                                 tuple(np.ones(n) for n in counts))
            calls = []
            def fock(dm, t):
                calls.append(dm.copy())
                total = dm.sum(axis=0)
                return np.array([np.diag([.1,.2,.3])+.2*total-.1*d for d in dm])
            result, _ = crank_nicolson(state, np.eye(3), fock, 0., .1)
            self.assertTrue(all(d.shape == (2,3,3) for d in calls))
            for c, n in zip(result.coefficients, counts):
                self.assertEqual(c.shape, (3,n))
            assert_allclose(np.trace(result.density(), axis1=1, axis2=2), counts, atol=1e-12)

    def test_joint_nonlinear_density_converges_to_independent_ode(self):
        overlap = np.array([[1.2,.1,0.],[.1,1.,.05],[0.,.05,.9]])
        inverse, _ = lowdin(overlap)
        rng = np.random.default_rng(17)
        q = np.linalg.qr(rng.normal(size=(3,3)) + 1j*rng.normal(size=(3,3)))[0]
        state = OrbitalState((inverse[:, :2], inverse @ q[:, :1]), ([1,1], [1]))
        h = np.array([[.2,.1j,.07],[-.1j,-.3,.04],[.07,.04,.4]])
        field = np.diag([.03,-.02,.01])
        alpha, start, duration = 2.5, .3, .8

        def fock(dm, time):
            common = h + .2*dm.sum(axis=0) + np.sin(.7*time)*field
            return np.array([common-.1*dm[0], common-.1*dm[1]])

        def rhs(time, flattened):
            ca, cb = flattened[:6].reshape(3,2), flattened[6:].reshape(3,1)
            dm = np.array([ca @ ca.conj().T, cb @ cb.conj().T])
            ha, hb = fock(dm, time)
            return np.concatenate(((-1j*alpha*np.linalg.solve(overlap, ha@ca)).ravel(),
                                   (-1j*alpha*np.linalg.solve(overlap, hb@cb)).ravel()))

        initial = np.concatenate([c.ravel() for c in state.coefficients])
        reference = solve_ivp(rhs, (start,start+duration), initial, method="DOP853",
                              rtol=3e-14, atol=1e-14)
        self.assertTrue(reference.success, reference.message)
        ca = reference.y[:6,-1].reshape(3,2)
        cb = reference.y[6:,-1].reshape(3,1)
        expected = np.array([ca@ca.conj().T, cb@cb.conj().T])
        self.assertGreater(np.linalg.norm(expected-state.density()), .1)
        for variant in ("cn", "cn_accelerated"):
            errors = []
            for count in (4,8,16):
                candidate = state
                for k in range(count):
                    candidate, diagnostic = crank_nicolson(candidate, overlap, fock,
                        start+k*duration/count, duration/count, alpha=alpha,
                        options=CNOptions(variant, 2e-13, 120))
                    self.assertLess(diagnostic.residual, 2e-13)
                assert_allclose(np.trace(candidate.density()@overlap,axis1=1,axis2=2),
                                [2.,1.],atol=1e-12,rtol=0)
                errors.append(np.linalg.norm(candidate.density()-expected))
            self.assertLess(errors[-1], 1e-3)
            assert_allclose(np.array(errors[:-1])/errors[1:], 4., rtol=.035)

    def test_cn_alpha_dt_equivalence_requires_an_autonomous_fock(self):
        state = OrbitalState((np.array([[np.sqrt(.8)],[1j*np.sqrt(.2)]]),), ([2],))
        h = np.array([[.2,.1j],[-.1j,-.3]])
        for variant in ("cn", "cn_accelerated"):
            for time_dependent in (False, True):
                endpoints = []
                for alpha, dt in ((3.,.1),(1.,.3)):
                    times = []
                    def fock(dm, time):
                        times.append(time)
                        return h+.3*dm + (np.sin(time)*np.diag([.2,-.1]) if time_dependent else 0.)
                    final, _ = crank_nicolson(state,np.eye(2),fock,.7,dt,alpha=alpha,
                        options=CNOptions(variant,1e-13,120))
                    endpoints.append(final.density())
                    self.assertEqual(set(times),{.7,.7+dt})
                if time_dependent:
                    self.assertGreater(np.linalg.norm(endpoints[0]-endpoints[1]),1e-4)
                else:
                    assert_allclose(endpoints[0],endpoints[1],atol=1e-12,rtol=0)

    def test_failed_and_invalid_trials_do_not_mutate_input(self):
        state = OrbitalState((np.eye(2, dtype=complex)[:, :1],), ([2],))
        before = state.copy()
        with self.assertRaises(StepFailure):
            crank_nicolson(state, np.eye(2), lambda d,t: np.array([[0,1],[1,0]]),
                           0., .5, options=CNOptions(max_iterations=1))
        with self.assertRaises(StepFailure):
            crank_nicolson(state, np.eye(2), lambda d,t: np.full((2,2), np.nan), 0., .1)
        for dt, alpha in ((0,1), (.1,0), (np.nan,1)):
            with self.assertRaises(ValueError):
                crank_nicolson(state, np.eye(2), lambda d,t: np.eye(2), 0., dt, alpha=alpha)
        assert_allclose(state.coefficients[0], before.coefficients[0], atol=0)

    def test_transport_round_trip_and_invalid_states(self):
        left = np.array([[1.,.2],[.2,1.2]])
        right = np.array([[1.1,-.1],[-.1,1.4]])
        inverse, half = lowdin(left)
        state = OrbitalState((inverse.astype(complex),), ([2,0],))
        new = transport(state, left, right)
        assert_allclose(lowdin(right)[1]@new.coefficients[0], half@state.coefficients[0], atol=1e-12, rtol=0)
        assert_allclose(transport(new,right,left).density(),state.density(),atol=1e-12,rtol=0)
        for overlap in (np.diag([1.,0.]), np.array([[1.,.2],[.1,1.]])):
            with self.assertRaises(ValueError):
                lowdin(overlap)
        with self.assertRaises(ValueError):
            OrbitalState((np.eye(2),), ([1.5, .5],))
        with self.assertRaises(ValueError):
            NuclearState([[0,0,0]], [[0,0,0]], [-1])


if __name__ == "__main__":
    unittest.main()

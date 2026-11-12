# Copyright (c) 2026 Ruiyi Zhou
"""Grouped Gram-inverse parallel-transport implicit midpoint (PT-IM).

With M = (C0 + C1)/2, G = M^H S M and B = blockdiag_g(M_g^H F M_g), the step
solves C1 - C0 + i dt (S^-1 F[D_M] M - M G^-1 B) = 0 with the unnormalized
midpoint density D_M = sum_g f_g M_g M_g^H. Because B and G are Hermitian the
root keeps the full Gram, including the orthogonality between the doubly and
singly occupied groups, so no projection follows the solve.
"""

from copy import deepcopy
import unittest

import numpy as np
from numpy.testing import assert_allclose
from pyscf import dft, gto
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import root

from radiant.backends.pyscf import PySCFProvider
from radiant.core.coupling import PaperFD
from radiant.core.execution import LocalExecution
from radiant.core.propagation import CNOptions, crank_nicolson
from radiant.core.state import OrbitalState
from radiant.core.transport import lowdin
from test_pt_mixed import (group_rotation, lih_mean_field, lih_two_one_state,
                           lowdin_density, mean_field_fock, random_model)
from toy_provider import ToyProvider


OPTIONS = CNOptions('pt_im', 1e-12, 200)


def unrestricted_case():
    state, overlap, core = random_model()
    inverse, _ = lowdin(overlap)
    q = np.linalg.qr(np.random.default_rng(3).normal(size=(6, 6)) + 0j)[0]
    unrestricted = OrbitalState((inverse @ q[:, :3], inverse @ q[:, 3:5]), ([1, 1, 1], [1, 1]))

    def spin_fock(density, time):
        return np.array([core + .1 * overlap @ (density[s].real + .5 * density[1 - s].real) @ overlap
                         for s in (0, 1)])
    return unrestricted, overlap, spin_fock


def cases():
    """Restricted 2/1, restricted closed-shell and unrestricted states, all far from stationary.

    The restricted Fock carries an explicit time dependence, so the midpoint
    time of the Fock is part of what the independent residual checks.
    """
    state, overlap, core = random_model()
    field = overlap @ np.diag(np.linspace(-.5, .5, overlap.shape[0])) @ overlap
    base = mean_field_fock(overlap, core)

    def fock(density, time):
        return base(density, time) + 2. * time * field
    closed = OrbitalState((state.coefficients[0][:, :2],), ([2, 2],))
    return [('restricted_2_1', state, overlap, fock), ('restricted_closed', closed, overlap, fock),
            ('unrestricted', *unrestricted_case())]


def midpoint_residual(state, trial, overlap, fock, dt):
    """Independent PT-IM residual written from the defining equation."""
    mids = [.5 * (a + b) for a, b in zip(state.coefficients, trial.coefficients)]
    value = np.asarray(fock(OrbitalState(tuple(mids), state.occupations).density(), dt / 2))
    focks = value[None] if value.ndim == 2 else value
    errors = []
    for m, c0, c1, f, h in zip(mids, state.coefficients, trial.coefficients, state.occupations, focks):
        gauge = np.zeros((m.shape[1], m.shape[1]), complex)
        for value in np.unique(f):
            columns = np.flatnonzero(np.asarray(f) == value)
            gauge[np.ix_(columns, columns)] = m[:, columns].conj().T @ h @ m[:, columns]
        phi = np.linalg.solve(overlap, h @ m) - m @ np.linalg.solve(m.conj().T @ overlap @ m, gauge)
        errors.append(c1 - c0 + 1j * dt * phi)
    return errors


class GroupedMidpointTests(unittest.TestCase):
    def test_public_step_is_the_independent_midpoint_root_and_keeps_the_full_gram(self):
        for name, state, overlap, fock in cases():
            for dt in (.05, .5):
                with self.subTest(case=name, dt=dt):
                    shapes = [c.shape for c in state.coefficients]

                    def packed(x):
                        parts, start = [], 0
                        for shape in shapes:
                            n = int(np.prod(shape))
                            parts.append((x[start:start + n] + 1j * x[start + n:start + 2 * n]).reshape(shape))
                            start += 2 * n
                        trial = OrbitalState(tuple(parts), state.occupations)
                        return np.concatenate([np.concatenate([e.real.ravel(), e.imag.ravel()])
                                               for e in midpoint_residual(state, trial, overlap, fock, dt)])
                    x0 = np.concatenate([np.concatenate([c.real.ravel(), c.imag.ravel()])
                                         for c in state.coefficients])
                    independent = root(packed, x0, method='hybr', options={'xtol': 1e-14})
                    # hybr may stop on its xtol test; what matters is the residual it reached.
                    self.assertLess(np.max(np.abs(packed(independent.x))), 1e-13)
                    final, diagnostic = crank_nicolson(state, overlap, fock, 0., dt, options=OPTIONS)
                    self.assertLessEqual(diagnostic.residual, OPTIONS.tolerance)
                    assert_allclose(np.concatenate([np.concatenate([c.real.ravel(), c.imag.ravel()])
                                                    for c in final.coefficients]),
                                    independent.x, atol=1e-11, rtol=0)
                    # The root itself keeps the full Gram, including across groups;
                    # the returned orbitals differ from it only at that level.
                    self.assertLess(diagnostic.raw_orthogonality_error, 1e-11)
                    self.assertLess(final.orthogonality_error(overlap), 1e-13)
                    self.assertTrue(np.isnan(diagnostic.returned_residual))

    def test_linear_dynamics_second_order_against_exact_propagator(self):
        state, overlap, core = random_model()
        exact = OrbitalState((expm(-1j * np.linalg.solve(overlap, core) * .4) @ state.coefficients[0],),
                             state.occupations)
        for alpha in (1., 8.):
            errors = []
            for scaled_dt in (.1, .05, .025):
                candidate = state.copy()
                for k in range(round(.4 / scaled_dt)):
                    candidate, _ = crank_nicolson(candidate, overlap, lambda d, t: core,
                                                  k * scaled_dt / alpha, scaled_dt / alpha,
                                                  alpha=alpha, options=OPTIONS)
                errors.append(np.linalg.norm(candidate.density() - exact.density()))
                self.assertLess(candidate.orthogonality_error(overlap), 1e-13)
            with self.subTest(alpha=alpha, errors=errors):
                self.assertGreater(errors[0] / errors[1], 3.7)
                self.assertGreater(errors[1] / errors[2], 3.7)

    def test_loose_tolerance_gram_defect_is_tolerance_sized_and_removed(self):
        """The root keeps the Gram only to the solver tolerance; it must not build up."""
        state, overlap, core = random_model()
        fock = mean_field_fock(overlap, core)
        loose = CNOptions('pt_im', 1e-7, 200)
        candidate, raw, returned = state.copy(), 0., 0.
        for k in range(400):
            candidate, diagnostic = crank_nicolson(candidate, overlap, fock, k * .05, .05, options=loose)
            self.assertLessEqual(diagnostic.residual, loose.tolerance)
            raw = max(raw, diagnostic.raw_orthogonality_error)
            returned = max(returned, candidate.orthogonality_error(overlap))
        self.assertLess(raw, 10 * loose.tolerance)
        self.assertLess(returned, 1e-13)

    def test_nonlinear_density_against_independent_density_ode(self):
        state, overlap, core = random_model()
        fock = mean_field_fock(overlap, core)
        inverse, _ = lowdin(overlap)
        n = overlap.shape[0]

        def derivative(time, vector):
            d = vector.reshape(n, n)
            h = inverse @ fock(inverse @ d @ inverse, time) @ inverse
            return (-1j * (h @ d - d @ h)).ravel()

        reference = solve_ivp(derivative, (0., .2), lowdin_density(state, overlap).ravel(),
                              method='DOP853', rtol=1e-12, atol=1e-14)
        self.assertTrue(reference.success)
        expected = reference.y[:, -1].reshape(n, n)
        errors = []
        for dt in (.02, .01, .005):
            candidate = state.copy()
            for k in range(round(.2 / dt)):
                candidate, diagnostic = crank_nicolson(candidate, overlap, fock, k * dt, dt, options=OPTIONS)
                self.assertLessEqual(diagnostic.residual, OPTIONS.tolerance)
            errors.append(np.linalg.norm(lowdin_density(candidate, overlap) - expected))
            spectrum = np.linalg.eigvalsh(lowdin_density(candidate, overlap))
            assert_allclose(np.sort(spectrum), [0, 0, 1, 1, 2, 2], atol=1e-11, rtol=0)
        with self.subTest(errors=errors):
            self.assertLess(errors[-1], 5e-5)
            self.assertGreater(errors[0] / errors[1], 3.7)
            self.assertGreater(errors[1] / errors[2], 3.7)

    def test_doubly_singly_rotation_is_physical_not_gauge(self):
        """No virtual space: only d<->s mixing can change the density."""
        h = np.array([[0., .3 - .2j], [.3 + .2j, .7]])
        state = OrbitalState((np.eye(2, dtype=complex),), ([2, 1],))
        exact_u = expm(-1j * h * .5)
        exact = exact_u @ np.diag([2., 1.]) @ exact_u.conj().T
        errors = []
        for steps in (10, 20):
            candidate = state.copy()
            for k in range(steps):
                candidate, _ = crank_nicolson(candidate, np.eye(2), lambda d, t: h,
                                              k * .5 / steps, .5 / steps, options=OPTIONS)
            errors.append(np.linalg.norm(candidate.density() - exact))
        self.assertGreater(np.linalg.norm(exact - np.diag([2., 1.])), .2)
        self.assertLess(errors[-1], 2e-4)
        self.assertGreater(errors[0] / errors[1], 3.7)

    def test_time_reversal_alpha_dt_product_and_group_covariance(self):
        state, overlap, core = random_model()
        fock = mean_field_fock(overlap, core)
        forward, _ = crank_nicolson(state, overlap, fock, 7., .05, alpha=8., options=OPTIONS)
        reverse, _ = crank_nicolson(forward, overlap, fock, 7.05, -.05, alpha=8., options=OPTIONS)
        # The midpoint rule is symmetric: the reverse step retraces the forward one.
        assert_allclose(reverse.coefficients[0], state.coefficients[0], atol=1e-11, rtol=0)
        a, da = crank_nicolson(state, overlap, fock, 7., .02, alpha=8., options=OPTIONS)
        b, db = crank_nicolson(state, overlap, fock, 7., .16, options=OPTIONS)
        assert_allclose(a.density(), b.density(), atol=1e-12, rtol=0)
        self.assertEqual(da.iterations, db.iterations)
        reference, _ = crank_nicolson(state, overlap, fock, 0., .1, options=OPTIONS)
        rotation = group_rotation(np.random.default_rng(3), (2, 2))
        rotated = OrbitalState((state.coefficients[0] @ rotation,), state.occupations)
        c, _ = crank_nicolson(rotated, overlap, fock, 0., .1, options=OPTIONS)
        assert_allclose(c.coefficients[0], reference.coefficients[0] @ rotation, atol=1e-11, rtol=0)
        order = [2, 0, 3, 1]
        permuted = OrbitalState((state.coefficients[0][:, order],), (np.asarray([2, 2, 1, 1])[order],))
        d, _ = crank_nicolson(permuted, overlap, fock, 0., .1, options=OPTIONS)
        assert_allclose(d.coefficients[0], reference.coefficients[0][:, order], atol=1e-11, rtol=0)

    def test_stationary_states_are_fixed_points_and_support_boundary(self):
        eye = np.eye(3, dtype=complex)
        h = np.diag([0., .4, 1.])
        for occupations in ([1], [2, 2], [2, 1], [1, 2, 1]):
            state = OrbitalState((eye[:, :len(occupations)],), (occupations,))
            final, diagnostic = crank_nicolson(state, np.eye(3), lambda d, t: h, 0., 50., options=OPTIONS)
            with self.subTest(occupations=occupations):
                assert_allclose(final.coefficients[0], state.coefficients[0], atol=1e-14, rtol=0)
                self.assertEqual(diagnostic.iterations, 1)
        never = lambda d, t: self.fail('Fock must not be called')
        for occupations in ([2, 1, 0], [1, 0], []):
            state = OrbitalState((eye[:, :len(occupations)],), (occupations,))
            with self.subTest(occupations=occupations), self.assertRaises(ValueError):
                crank_nicolson(state, np.eye(3), never, 0., .1, options=OPTIONS)
        with self.assertRaises(ValueError):
            LocalExecution([ToyProvider(), ToyProvider(1)], options=OPTIONS)
        with self.assertRaises(ValueError):
            PaperFD(options=OPTIONS)
        for history in (0, -1, 2.5, True):
            with self.subTest(history=history), self.assertRaises(ValueError):
                CNOptions('pt_im', 1e-9, 80, history)


class MolecularMidpointTests(unittest.TestCase):
    def test_moving_basis_two_one_matches_small_step_orbital_cn(self):
        provider = PySCFProvider(lih_mean_field())
        state = lih_two_one_state(provider)
        end = provider.positions.copy()
        end[1, 2] += .002
        results = []
        for variant in ('pt_im', 'cn_accelerated'):
            run = LocalExecution([deepcopy(provider)], [state], alphas=[2.],
                                 options=CNOptions(variant, 1e-12, 120), time=3.)
            frame = run.advance(end, .02, n_substeps=20)
            self.assertLess(run.states[0].orthogonality_error(run.providers[0].overlap), 1e-11)
            results.append((run.states[0].density(), frame.observation))
        assert_allclose(results[0][0], results[1][0], atol=2e-7, rtol=0)
        assert_allclose(results[0][1].energies, results[1][1].energies, atol=2e-8, rtol=0)
        assert_allclose(results[0][1].forces, results[1][1].forces, atol=2e-7, rtol=0)

    def test_closed_shell_moving_basis_converges_at_large_alpha_dt(self):
        # The N2 path of the PT-CN Anderson regression, here with alpha*dt = 10.
        def positions(bond):
            return np.array([[0., 0., -bond / 2], [0., 0., bond / 2]])
        bond, velocity, dt = 2.2, -.0058, .5
        mol = gto.M(atom=[('N', p) for p in positions(bond)], unit='Bohr', basis='sto-3g', verbose=0)
        mf = dft.RKS(mol, xc='pbe')
        mf.chkfile, mf.grids.level, mf.grids.prune = None, 1, None
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
        mf.kernel()
        self.assertTrue(mf.converged)
        run = LocalExecution([PySCFProvider(mf)], alphas=[20.], options=CNOptions('pt_im', 1e-8, 120))
        for step in range(4):
            with self.subTest(step=step):
                result = run.advance(positions(bond + velocity * dt), dt,
                                     midpoint=positions(bond + velocity * dt / 2), n_substeps=1)
                bond += velocity * dt
                diagnostic = result.diagnostics[0]
                self.assertLessEqual(diagnostic.residual, 1e-8)
                self.assertLess(diagnostic.orthogonality_error, 1e-8)
                self.assertLess(diagnostic.iterations, 30)


if __name__ == '__main__':
    unittest.main()

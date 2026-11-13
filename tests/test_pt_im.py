# Copyright (c) 2026 Ruiyi Zhou
"""Parallel-transport implicit midpoint (PT-IM).

With M = (C0 + C1)/2, the step solves C1 - C0 + i dt phi(M) = 0. For one
occupation group per spin, with G = M^H S M and B = M^H F M,
phi = S^-1 F[D_M] M - M G^-1 B with the unnormalized midpoint density
D_M = f M M^H. A restricted 2/1 spin uses the hamc field, the gradient of
H(Cb) = E[Cb p(Gb) Cb^H] with Cb = M diag(f)^(1/2), Gb = Cb^H S Cb and
p = f_c / x on each occupation cluster of Gb. Both roots keep the full Gram,
including the orthogonality between the doubly and singly occupied groups,
so no projection follows the solve.
"""

from copy import deepcopy
import unittest
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose
from pyscf import dft, gto
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor, eigh, expm, null_space
from scipy.optimize import root

from radiant.backends.pyscf import PySCFProvider
from radiant.core import propagation
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


def two_one_projector(m, f, overlap):
    """Cb, Gb^-1 and the spectral projector of Gb onto its n_1 smallest eigenvalues."""
    cb = m * np.sqrt(f)
    gb = cb.conj().T @ overlap @ cb
    values, vectors = eigh(gb)
    lower = vectors[:, :np.count_nonzero(f == 1)]
    return cb, np.linalg.inv(gb), lower @ lower.conj().T, values, vectors


def hamc_density(m, f, overlap):
    """Fock density Cb p(Gb) Cb^H with p(Gb) = (2 I - P_1) Gb^-1."""
    cb, inverse, p1, _, _ = two_one_projector(m, f, overlap)
    return cb @ ((2 * np.eye(len(f)) - p1) @ inverse) @ cb.conj().T


def hamc_field(m, f, overlap, h):
    """Independent hamc field: S phi R = F Cb p(Gb) + S Cb Dp(Gb)[Cb^H F Cb].

    p(X) = (2 I - P_1(X)) X^-1, so Dp[K] = -DP_1[K] X^-1 - p(X) K X^-1, with
    the first-order perturbation of the spectral projector P_1.
    """
    cb, inverse, p1, values, vectors = two_one_projector(m, f, overlap)
    n1 = np.count_nonzero(f == 1)
    k = cb.conj().T @ h @ cb
    low, high = vectors[:, :n1], vectors[:, n1:]
    weight = 1. / (values[:n1, None] - values[None, n1:])
    coupling = low @ (weight * (low.conj().T @ k @ high)) @ high.conj().T
    dp1 = coupling + coupling.conj().T
    p = (2 * np.eye(len(f)) - p1) @ inverse
    dp = -dp1 @ inverse - p @ k @ inverse
    return (np.linalg.solve(overlap, h @ cb @ p) + cb @ dp) / np.sqrt(f)


def grouped_midpoint_residual(state, trial, overlap, fock, dt):
    """Residual of the grouped field S^-1 F M - M G^-1 blockdiag_g(M_g^H F M_g).

    PT-IM uses it for one group per spin; for 2/1 it is the first solver phase.
    """
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


def midpoint_residual(state, trial, overlap, fock, dt):
    """Independent PT-IM residual written from the defining equation."""
    mids = [.5 * (a + b) for a, b in zip(state.coefficients, trial.coefficients)]
    occupations = [np.asarray(f, float) for f in state.occupations]
    two_one = len(mids) == 1 and len(np.unique(occupations[0])) > 1
    density = (hamc_density(mids[0], occupations[0], overlap) if two_one
               else OrbitalState(tuple(mids), state.occupations).density())
    value = np.asarray(fock(density, dt / 2))
    focks = value[None] if value.ndim == 2 else value
    errors = []
    for m, c0, c1, f, h in zip(mids, state.coefficients, trial.coefficients, occupations, focks):
        if two_one:
            phi = hamc_field(m, f, overlap, h)
        else:
            phi = (np.linalg.solve(overlap, h @ m)
                   - m @ np.linalg.solve(m.conj().T @ overlap @ m, m.conj().T @ h @ m))
        errors.append(c1 - c0 + 1j * dt * phi)
    return errors


def tangent_symplectic_defect(state, overlap, fock, dt, options, eps=1e-4):
    """|(D Phi T)^T Om (D Phi T) - T^T Om T| / |T^T Om T| of one restricted step.

    Om(u, v) = Im tr(u^H S v) in Cb = C diag(f)^(1/2). T spans the tangent
    space of Cb^H S Cb = diag(f) at the state, D Phi T is the central
    difference of the step along Loewdin-retracted directions.
    """
    c0 = state.coefficients[0]
    root = np.sqrt(np.asarray(state.occupations[0], float))
    cb0 = c0 * root
    size = c0.size

    def complex_of(x):
        return (x[:size] + 1j * x[size:]).reshape(c0.shape)

    def real_of(z):
        return np.concatenate([z.real.ravel(), z.imag.ravel()])

    def gram_change(x):
        q = cb0.conj().T @ overlap @ complex_of(x)
        return real_of(q + q.conj().T)
    tangent = null_space(np.array([gram_change(e) for e in np.eye(2 * size)]).T)
    metric = np.kron(overlap, np.eye(c0.shape[1]))
    omega = np.block([[np.zeros_like(metric), metric], [-metric, np.zeros_like(metric)]])
    columns = []
    for k in range(tangent.shape[1]):
        direction = complex_of(tangent[:, k]) / root
        ends = []
        for sign in (1., -1.):
            c = c0 + sign * eps * direction
            values, vectors = eigh(c.conj().T @ overlap @ c)
            c = c @ ((vectors * values**-.5) @ vectors.conj().T)
            final, _ = crank_nicolson(OrbitalState((c,), state.occupations), overlap, fock, 0., dt,
                                      options=options)
            ends.append(real_of(final.coefficients[0] * root))
        columns.append((ends[0] - ends[1]) / (2 * eps))
    image = np.array(columns).T
    reference = tangent.T @ omega @ tangent
    return np.abs(image.T @ omega @ image - reference).max() / np.abs(reference).max()


def gradient_fock(overlap, core):
    """Nonlinear Fock F = dE/dD of E = tr(h D) + .05 tr(SDSD) + .01 tr(SDSD)^2."""
    def energy(density):
        quadratic = np.trace(overlap @ density @ overlap @ density).real
        return np.trace(core @ density).real + .05 * quadratic + .01 * quadratic**2

    def fock(density, time):
        quadratic = np.trace(overlap @ density @ overlap @ density).real
        return core + (.1 + .04 * quadratic) * overlap @ density @ overlap
    return energy, fock


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


class GroupPreconditionerTests(unittest.TestCase):
    def test_closed_form_block_inverts_the_occupied_jacobian(self):
        # With E = C0^H F C0 diagonal the closed form neglects nothing, and
        # the occupied coordinates depend on F only through E. Orbital
        # energies near -12 make the inter-group coupling tau |e_i + e_a| about 36.
        rng = np.random.default_rng(11)
        b = rng.normal(size=(7, 7))
        overlap = np.eye(7) + .2 * b @ b.T / 7
        a = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
        fock = .5 * (a + a.conj().T) - 12. * overlap
        # A random occupied span, rotated so that E is diagonal; the span is
        # not invariant under S^-1 F, so the tangent blocks still couple to it.
        inverse, _ = lowdin(overlap)
        span = inverse @ np.linalg.qr(rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3)))[0]
        _, rotation = eigh(span.conj().T @ fock @ span)
        joint = span @ rotation
        initials = [joint[:, :2], joint[:, 2:3]]
        occupations = ([2, 2, 1],)
        dt = 3.

        def coordinates(delta):
            trial = OrbitalState((joint @ (np.eye(3) + delta),), occupations)
            errors = grouped_midpoint_residual(OrbitalState((joint,), occupations), trial, overlap,
                                               lambda d, t: fock, dt)[0]
            return -joint.conj().T @ overlap @ errors

        delta = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        jacobian = (coordinates(1e-6 * delta) - coordinates(-1e-6 * delta)) / 2e-6
        self.assertGreater(np.linalg.norm(jacobian + delta), 10 * np.linalg.norm(delta))
        natives = [propagation._pt_preconditioner(overlap, fock, c, dt / 2) for c in initials]
        precondition = propagation._midpoint_group_preconditioner(overlap, fock, initials, natives, dt / 2)
        residual = joint @ jacobian
        step = np.concatenate(precondition([residual[:, :2], residual[:, 2:]]), axis=1)
        assert_allclose(step, -joint @ delta, atol=1e-7, rtol=0)
        # A tangent residual must not leak into the occupied coordinates.
        tangent = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
        tangent -= joint @ (joint.conj().T @ overlap @ tangent)
        residual = residual + tangent
        step = np.concatenate(precondition([residual[:, :2], residual[:, 2:]]), axis=1)
        assert_allclose(joint.conj().T @ overlap @ step, -delta, atol=1e-7, rtol=0)

    def test_only_spins_with_several_groups_use_the_joint_block(self):
        # Closed-shell and unrestricted spins keep the split preconditioner
        # unchanged, so their steps are those of the single-group scheme.
        calls = []
        original = propagation._midpoint_group_preconditioner

        def spy(overlap, hamiltonian, initials, natives, half_dt):
            calls.append(len(initials))
            return original(overlap, hamiltonian, initials, natives, half_dt)
        for name, state, overlap, fock in cases():
            calls.clear()
            with self.subTest(case=name), mock.patch.object(propagation, '_midpoint_group_preconditioner', spy):
                crank_nicolson(state, overlap, fock, 0., .3, options=OPTIONS)
                crank_nicolson(state, overlap, fock, 0., .3, options=CNOptions('pt_cn', 1e-12, 200))
                self.assertEqual(set(calls), {2} if name == 'restricted_2_1' else set())

    def test_two_one_core_pairs_converge_in_few_updates_at_large_alpha_dt(self):
        # N2 HOMO -> LUMO vertical 2/1 state at alpha*dt = 10, where tau |e_i + e_a|
        # reaches about 70 for the N 1s pairs. Without the joint occupied
        # block both steps take 28 updates.
        mol = gto.M(atom=[('N', (0., 0., -1.1)), ('N', (0., 0., 1.1))], unit='Bohr', basis='sto-3g', verbose=0)
        mf = dft.RKS(mol, xc='pbe')
        mf.chkfile, mf.grids.level, mf.grids.prune = None, 1, None
        mf.conv_tol = 1e-12
        mf.kernel()
        self.assertTrue(mf.converged)
        provider = PySCFProvider(mf)
        columns = [0, 1, 2, 3, 4, 5, 6, 7]
        state = OrbitalState((mf.mo_coeff[:, columns] + 0j,), ([2, 2, 2, 2, 2, 2, 1, 1],))
        for alpha, dt in ((1., 10.), (20., .5)):
            with self.subTest(alpha=alpha):
                final, diagnostic = crank_nicolson(state, provider.overlap, provider.fock, 0., dt, alpha=alpha,
                                                   options=CNOptions('pt_im', 1e-12, 60))
                self.assertLessEqual(diagnostic.residual, 1e-12)
                self.assertLess(diagnostic.iterations, 20)
                self.assertLess(final.orthogonality_error(provider.overlap), 1e-11)


class HamiltonianTwoOneTests(unittest.TestCase):
    def test_hamc_field_is_the_gradient_of_the_modified_energy(self):
        state, overlap, core = random_model()
        energy, fock = gradient_fock(overlap, core)
        rng = np.random.default_rng(5)
        f = np.asarray(state.occupations[0], float)
        c = state.coefficients[0]
        groups = [np.flatnonzero(f == 2), np.flatnonzero(f == 1)]

        def field(m):
            orbitals, data = propagation._hamc_fock_orbitals(overlap, [m[:, g] for g in groups],
                                                             [f[g] for g in groups])
            density = sum(f[g][0] * x @ x.conj().T for x, g in zip(orbitals, groups))
            assert_allclose(density, hamc_density(m, f, overlap), atol=1e-13, rtol=0)
            h = fock(density, 0.)
            phi = np.empty_like(m)
            for g, part in zip(groups, propagation._hamc_phi(overlap, cho_factor(overlap), h, data)):
                phi[:, g] = part
            assert_allclose(phi, hamc_field(m, f, overlap, h), atol=1e-12, rtol=0)
            return phi, h
        # On the Gram constraint the hamc field is the grouped PT field.
        phi, h = field(c)
        grouped = propagation._midpoint_phi(overlap, cho_factor(overlap), h, [c[:, g] for g in groups])
        assert_allclose(phi, np.concatenate(grouped, axis=1), atol=1e-12, rtol=0)
        # Off the constraint, as at a midpoint, it is the gradient of the
        # modified energy: dH = 2 Re tr(dM^H S phi diag(f)).
        m = c + .1 * (rng.normal(size=c.shape) + 1j * rng.normal(size=c.shape))
        phi, _ = field(m)
        for k in range(3):
            direction = rng.normal(size=c.shape) + 1j * rng.normal(size=c.shape)
            numerical = (energy(hamc_density(m + 1e-5 * direction, f, overlap))
                         - energy(hamc_density(m - 1e-5 * direction, f, overlap))) / 2e-5
            analytic = 2 * np.trace(direction.conj().T @ overlap @ (phi * f)).real
            self.assertAlmostEqual(analytic / numerical, 1., delta=1e-7)

    def test_two_one_step_is_symplectic_and_single_group_step_is_not(self):
        state, overlap, core = random_model()
        _, fock = gradient_fock(overlap, core)
        options = CNOptions('pt_im', 1e-12, 300)
        for dt in (.5, 1.):
            with self.subTest(dt=dt):
                self.assertLess(tangent_symplectic_defect(state, overlap, fock, dt, options), 1e-7)
        # The single-group PT-IM integrates the non-Hamiltonian PT field (An
        # and Lin), so the same measure detects a defect there.
        closed = OrbitalState((state.coefficients[0][:, :2],), ([2, 2],))
        self.assertGreater(tangent_symplectic_defect(closed, overlap, fock, .5, options), 1e-2)

    def test_two_one_root_does_not_depend_on_the_energy_zero(self):
        state, overlap, core = random_model()
        fock = mean_field_fock(overlap, core)
        final, _ = crank_nicolson(state, overlap, fock, 0., .5, options=OPTIONS)
        shifted, _ = crank_nicolson(state, overlap, lambda d, t: fock(d, t) + 10. * overlap, 0., .5,
                                    options=OPTIONS)
        assert_allclose(shifted.coefficients[0], final.coefficients[0], atol=1e-11, rtol=0)
        # At this midpoint the grouped field would gain 10 M G^-1 G_inter.
        m = .5 * (state.coefficients[0] + final.coefficients[0])
        gained = propagation._midpoint_phi(overlap, cho_factor(overlap), 10. * overlap, [m[:, :2], m[:, 2:]])
        self.assertGreater(np.linalg.norm(np.concatenate(gained, axis=1)), 1e-3)

    def test_exact_occupied_block_inverts_the_hamc_jacobian(self):
        # As the closed-form test of _midpoint_group_preconditioner, but E has
        # inter-group blocks, which the closed-form pairs neglect.
        rng = np.random.default_rng(11)
        b = rng.normal(size=(7, 7))
        overlap = np.eye(7) + .2 * b @ b.T / 7
        a = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
        fock = .5 * (a + a.conj().T) - 12. * overlap
        inverse, _ = lowdin(overlap)
        joint = inverse @ np.linalg.qr(rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3)))[0]
        initials = [joint[:, :2], joint[:, 2:3]]
        occupations = ([2, 2, 1],)
        dt = 3.

        def coordinates(delta):
            trial = OrbitalState((joint @ (np.eye(3) + delta),), occupations)
            errors = midpoint_residual(OrbitalState((joint,), occupations), trial, overlap,
                                       lambda d, t: fock, dt)[0]
            return -joint.conj().T @ overlap @ errors

        delta = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        jacobian = (coordinates(1e-6 * delta) - coordinates(-1e-6 * delta)) / 2e-6
        natives = [propagation._pt_preconditioner(overlap, fock, c, dt / 2) for c in initials]
        residual = joint @ jacobian
        exact = propagation._hamc_group_preconditioner(overlap, fock, initials, natives, dt, [2., 1.])
        step = np.concatenate(exact([residual[:, :2], residual[:, 2:]]), axis=1)
        assert_allclose(step, -joint @ delta, atol=1e-7, rtol=0)
        closed_form = propagation._midpoint_group_preconditioner(overlap, fock, initials, natives, dt / 2)
        step = np.concatenate(closed_form([residual[:, :2], residual[:, 2:]]), axis=1)
        self.assertGreater(np.abs(step + joint @ delta).max(), .1)

    def test_assembled_occupied_block_inverts_its_linear_map(self):
        # The solver assembles the real-linear map Delta -> Q term by term; here
        # the map is applied directly, for larger blocks and shuffled occupations.
        def occupied_map(energies, f, dt, delta):
            root = np.sqrt(f)
            p1 = np.where(f[:, None] == f[None, :], -1. / f[:, None], 0.)
            twos = (f == 2.).astype(int)
            p2 = np.choose(twos[:, None, None] + twos[None, :, None] + twos[None, None, :], [1., 1., -.5, .25])
            b0 = root[:, None] * energies * root[None, :]
            dg = root[:, None] * (delta + delta.conj().T) * root[None, :] / 2
            db = root[:, None] * (delta.conj().T @ energies + energies @ delta) * root[None, :] / 2
            d2 = np.einsum('ikj,ik,kj->ij', p2, dg, b0) + np.einsum('ikj,ik,kj->ij', p2, b0, dg)
            dtrans = (energies @ delta * root[None, :] / 2 + (energies * root[None, :]) @ (p1 * dg)
                      + delta * root[None, :] @ (p1 * b0) / 2 + root[:, None] * (p1 * db + d2))
            return -delta - 1j * dt * dtrans / root[None, :]

        rng = np.random.default_rng(5)
        for occupations in ([2, 1, 2], [1, 2, 2, 1, 2], [2] * 6 + [1] * 3, [1, 1, 1, 1], [2] * 5):
            f = np.asarray(occupations, float)
            n = len(f)
            a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            energies = .5 * (a + a.conj().T)
            delta = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            for dt in (.3, 10.):
                with self.subTest(occupations=occupations, dt=dt):
                    solve = propagation._hamc_occupied_solver(energies, f, dt)
                    assert_allclose(solve(occupied_map(energies, f, dt, delta)), delta, atol=1e-12, rtol=0)

    def test_exact_block_only_near_the_gram_constraint_and_for_small_blocks(self):
        state, overlap, core = random_model()
        fock = mean_field_fock(overlap, core)
        calls = []
        original = propagation._hamc_group_preconditioner

        def spy(*args):
            calls.append(len(args[2]))
            return original(*args)
        with mock.patch.object(propagation, '_hamc_group_preconditioner', spy):
            exact, _ = crank_nicolson(state, overlap, fock, 0., .1, options=OPTIONS)
            self.assertTrue(calls)
            calls.clear()
            with mock.patch.object(propagation, '_HAMC_EXACT_MAX_ORBITALS', 3):
                closed_form, _ = crank_nicolson(state, overlap, fock, 0., .1, options=OPTIONS)
            self.assertFalse(calls)
            # At dt = 2 the smallest midpoint Gram eigenvalue is about 0.35.
            final, _ = crank_nicolson(state, overlap, fock, 0., 2., options=CNOptions('pt_im', 1e-10, 300))
            self.assertLess(midpoint_gram_minimum(state, final, overlap), propagation._HAMC_EXACT_GRAM)
            self.assertFalse(calls)
        assert_allclose(closed_form.coefficients[0], exact.coefficients[0], atol=1e-11, rtol=0)

    def test_strongly_rotated_two_one_states_converge_at_large_dt(self):
        # The random 2/1 states are far from stationary. The exact occupied
        # block, linearized at Gb = diag(f), diverged here.
        state, overlap, core = random_model()
        for name, fock in (('mean_field', mean_field_fock(overlap, core)),
                           ('gradient', gradient_fock(overlap, core)[1])):
            with self.subTest(fock=name):
                final, diagnostic = crank_nicolson(state, overlap, fock, 0., 2.,
                                                   options=CNOptions('pt_im', 1e-10, 300))
                self.assertLessEqual(diagnostic.residual, 1e-10)
                self.assertLess(np.abs(midpoint_residual(state, final, overlap, fock, 2.)[0]).max(), 1e-9)

    def test_single_group_spins_never_use_the_hamc_field(self):
        calls = []
        original = propagation._hamc_fock_orbitals

        def spy(overlap, mids, occupations):
            calls.append(len(mids))
            return original(overlap, mids, occupations)
        for name, state, overlap, fock in cases():
            calls.clear()
            with self.subTest(case=name), mock.patch.object(propagation, '_hamc_fock_orbitals', spy):
                crank_nicolson(state, overlap, fock, 0., .3, options=OPTIONS)
                crank_nicolson(state, overlap, fock, 0., .3, options=CNOptions('pt_cn', 1e-12, 200))
                self.assertEqual(set(calls), {2} if name == 'restricted_2_1' else set())


def rotated_closed_shell(atoms, theta):
    """Closed-shell sto-3g PBE state whose HOMO is rotated by theta into the LUMO."""
    mol = gto.M(atom=atoms, unit='Bohr', basis='sto-3g', verbose=0)
    mf = dft.RKS(mol, xc='pbe')
    mf.chkfile, mf.grids.level, mf.grids.prune = None, 1, None
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    occupied = int(np.sum(mf.mo_occ > 0))
    c = mf.mo_coeff[:, :occupied].astype(complex)
    c[:, -1] = np.cos(theta) * mf.mo_coeff[:, occupied - 1] + np.sin(theta) * mf.mo_coeff[:, occupied]
    return PySCFProvider(mf), OrbitalState((c,), (np.full(occupied, 2.),))


def midpoint_gram_minimum(state, final, overlap):
    m = .5 * (state.coefficients[0] + final.coefficients[0])
    return np.linalg.eigvalsh(m.conj().T @ overlap @ m)[0]


class MidpointFrameTests(unittest.TestCase):
    def test_closed_shell_core_converges_in_few_updates_at_large_dt(self):
        # At large dt the midpoint span leaves the initial span by about
        # 1 - g_min (g_min the smallest midpoint Gram eigenvalue). A split
        # fixed at C0 lets tau |e_core| amplify that error: Cl 1s is near
        # -100 Ha. With that split these steps take 90 and 142 updates.
        provider, state = rotated_closed_shell([('H', (0., 0., -1.3)), ('Cl', (0., 0., 1.1))], .3)
        for dt in (20., 40.):
            with self.subTest(dt=dt):
                final, diagnostic = crank_nicolson(state, provider.overlap, provider.fock, 0., dt,
                                                   options=CNOptions('pt_im', 1e-10, 200))
                self.assertLessEqual(diagnostic.residual, 1e-10)
                self.assertLess(diagnostic.iterations, 70)
                self.assertGreater(midpoint_gram_minimum(state, final, provider.overlap), .9)

    def test_large_step_keeps_the_root_continued_from_small_dt(self):
        # The midpoint rule has further roots at large dt. Continuing the root
        # from dt = 2 in steps of 2 gives g_min = 0.685 at dt = 40; following
        # the midpoint frame from the first update instead reaches a root with
        # g_min = 0.28. The initial frame is kept until the residual has
        # fallen by _MIDPOINT_FRAME_SWITCH.
        provider, state = rotated_closed_shell([('N', (0., 0., -1.1)), ('N', (0., 0., 1.1))], .6)
        final, diagnostic = crank_nicolson(state, provider.overlap, provider.fock, 0., 40.,
                                           options=CNOptions('pt_im', 1e-10, 300))
        self.assertLessEqual(diagnostic.residual, 1e-10)
        self.assertAlmostEqual(midpoint_gram_minimum(state, final, provider.overlap), .685, delta=.005)


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

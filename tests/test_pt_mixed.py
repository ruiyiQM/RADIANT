# Copyright (c) 2026 Ruiyi Zhou
"""Restricted 2/1 (doubly/singly occupied) PT-CN with occupation-group gauge.

The restricted density P = 2 C_d C_d^H + C_s C_s^H is invariant only under
unitary mixing inside each occupation group. PT therefore removes the
intra-group rotation of each group separately and keeps the d<->s coupling.
"""

from copy import deepcopy
import unittest

import numpy as np
from numpy.testing import assert_allclose
from pyscf import dft, gto
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from radiant.backends.pyscf import PySCFProvider
from radiant.core.execution import LocalExecution
from radiant.core.propagation import CNOptions, crank_nicolson
from radiant.core.state import OrbitalState
from radiant.core.transport import lowdin, transport
from toy_provider import ToyProvider


OPTIONS = CNOptions('pt_cn', 1e-12, 120)
OCCUPATIONS = [2, 2, 1, 1]


def random_model(seed=7, nao=6):
    """Nonorthogonal metric, Hermitian core and an S-orthonormal 2/1 state."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(nao, nao))
    overlap = np.eye(nao) + .08 * (a + a.T)
    b = rng.normal(size=(nao, nao)) + 1j * rng.normal(size=(nao, nao))
    core = .5 * (b + b.conj().T) + np.diag(np.linspace(-1., 1.5, nao))
    core = overlap @ core @ overlap
    inverse, _ = lowdin(overlap)
    q = np.linalg.qr(rng.normal(size=(nao, nao)) + 1j * rng.normal(size=(nao, nao)))[0]
    state = OrbitalState((inverse @ q[:, :len(OCCUPATIONS)],), (OCCUPATIONS,))
    return state, overlap, core


def mean_field_fock(overlap, core):
    """Hartree-like local term plus an exchange-like term that sees Im(P)."""
    def fock(density, time):
        return (core + .1 * overlap @ np.diag(density.diagonal().real) @ overlap
                - .05 * overlap @ density @ overlap)
    return fock


def lowdin_density(state, overlap):
    _, half = lowdin(overlap)
    return half @ state.density() @ half


def group_rotation(rng, sizes):
    blocks = [np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0] for n in sizes]
    result = np.zeros((sum(sizes), sum(sizes)), complex)
    start = 0
    for block, n in zip(blocks, sizes):
        result[start:start + n, start:start + n] = block
        start += n
    return result


class MixedOccupationPTTests(unittest.TestCase):
    def test_group_action_is_the_density_equation_and_full_block_is_not(self):
        """Derivation check: grouped PT generates -i[S^-1 F P - P F S^-1]."""
        state, overlap, core = random_model()
        c, f = state.coefficients[0], np.asarray(OCCUPATIONS, float)
        fock = mean_field_fock(overlap, core)(state.density(), 0.)
        target = -1j * (np.linalg.solve(overlap, fock) @ state.density()
                        - state.density() @ fock @ np.linalg.inv(overlap))

        def density_rate(groups):
            rate = np.zeros_like(target)
            for columns in groups:
                cg = c[:, columns]
                action = np.linalg.solve(overlap, fock @ cg) - cg @ (cg.conj().T @ fock @ cg)
                velocity = -1j * action
                rate += f[columns[0]] * (velocity @ cg.conj().T + cg @ velocity.conj().T)
            return rate

        grouped = density_rate([np.array([0, 1]), np.array([2, 3])])
        assert_allclose(grouped, target, atol=1e-12, rtol=0)
        # Treating the whole 2/1 block as one gauge group drops d<->s dynamics.
        weighted = np.zeros_like(target)
        action = np.linalg.solve(overlap, fock @ c) - c @ (c.conj().T @ fock @ c)
        weighted += (-1j * action * f) @ c.conj().T
        weighted += weighted.conj().T
        self.assertGreater(np.linalg.norm(weighted - target), 1e-2)

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
                self.assertLess(candidate.orthogonality_error(overlap), 1e-12)
            with self.subTest(alpha=alpha, errors=errors):
                self.assertGreater(errors[0] / errors[1], 3.7)
                self.assertGreater(errors[1] / errors[2], 3.7)

    def test_nonlinear_density_against_independent_density_ode(self):
        state, overlap, core = random_model()
        fock = mean_field_fock(overlap, core)
        inverse, half = lowdin(overlap)
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
            # Occupation spectrum {2,2,1,1,0,0} and electron count are exact invariants.
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
        # A closed shell with the same two orbitals has no dynamics at all.
        closed, diagnostic = crank_nicolson(OrbitalState((np.eye(2, dtype=complex),), ([2, 2],)),
                                            np.eye(2), lambda d, t: h, 0., .5, options=OPTIONS)
        assert_allclose(closed.density(), 2 * np.eye(2), atol=1e-13, rtol=0)
        self.assertEqual(diagnostic.iterations, 1)

    def test_group_gauge_and_column_order_covariance(self):
        state, overlap, core = random_model()
        fock = mean_field_fock(overlap, core)
        reference, _ = crank_nicolson(state, overlap, fock, 0., .05, options=OPTIONS)
        rng = np.random.default_rng(3)
        rotated = OrbitalState((state.coefficients[0] @ group_rotation(rng, (2, 2)),), state.occupations)
        a, _ = crank_nicolson(rotated, overlap, fock, 0., .05, options=OPTIONS)
        assert_allclose(a.density(), reference.density(), atol=1e-11, rtol=0)
        order = [2, 0, 3, 1]
        permuted = OrbitalState((state.coefficients[0][:, order],), (np.asarray(OCCUPATIONS)[order],))
        b, _ = crank_nicolson(permuted, overlap, fock, 0., .05, options=OPTIONS)
        assert_allclose(b.density(), reference.density(), atol=1e-11, rtol=0)
        assert_allclose(b.occupations[0], np.asarray(OCCUPATIONS)[order])
        self.assertLess(b.orthogonality_error(overlap), 1e-12)

    def test_fixed_state_alpha_dt_product_and_time_reversal(self):
        state, overlap, core = random_model()
        fock = mean_field_fock(overlap, core)
        a, da = crank_nicolson(state, overlap, fock, 7., .02, alpha=8., options=OPTIONS)
        b, db = crank_nicolson(state, overlap, fock, 7., .16, options=OPTIONS)
        assert_allclose(a.density(), b.density(), atol=1e-12, rtol=0)
        self.assertEqual(da.iterations, db.iterations)
        forward, _ = crank_nicolson(state, overlap, fock, 7., .002, alpha=8., options=OPTIONS)
        reverse, _ = crank_nicolson(forward, overlap, fock, 7.002, -.002, alpha=8., options=OPTIONS)
        assert_allclose(reverse.density(), state.density(), atol=2e-9, rtol=0)

    def test_support_boundary(self):
        eye = np.eye(3, dtype=complex)
        never = lambda d, t: self.fail('Fock must not be called')
        for occupations in ([2, 1, 0], [1, 0], []):
            state = OrbitalState((eye[:, :len(occupations)],), (occupations,))
            with self.subTest(occupations=occupations), self.assertRaises(ValueError):
                crank_nicolson(state, np.eye(3), never, 0., .1, options=OPTIONS)
        unrestricted = OrbitalState((eye[:, :2], eye[:, :1]), ([1, 0], [1]))
        with self.assertRaises(ValueError):
            crank_nicolson(unrestricted, np.eye(3), never, 0., .1, options=OPTIONS)
        h = np.diag([0., .4, 1.])
        for occupations in ([1], [1, 1], [2, 1], [1, 2, 1]):
            state = OrbitalState((eye[:, :len(occupations)],), (occupations,))
            final, diagnostic = crank_nicolson(state, np.eye(3), lambda d, t: h, 0., .1, options=OPTIONS)
            with self.subTest(occupations=occupations):
                assert_allclose(final.density(), state.density(), atol=1e-13, rtol=0)
                self.assertEqual(diagnostic.iterations, 1)
        # Two-state PT remains unsupported even when one branch is restricted 2/1.
        with self.assertRaises(ValueError):
            LocalExecution([ToyProvider(), ToyProvider(1)], options=OPTIONS)


def lih_mean_field():
    mol = gto.M(atom='Li 0 0 0; H 0 0 3.0', basis='sto-3g', unit='Bohr', verbose=0)
    mf = dft.RKS(mol, xc='lda,vwn')
    mf.grids.level = 1
    mf.grids.prune = None
    mf.chkfile = None
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def lih_two_one_state(provider):
    """Core doubly occupied, HOMO and LUMO singly occupied, with a complex kick."""
    c = provider.mf.mo_coeff
    inverse, half = lowdin(provider.overlap)
    x = half @ c
    n = x.shape[1]
    generator = np.zeros((n, n), complex)
    generator[1, 3] = .05 + .03j
    generator[0, 2] = .02 - .01j
    generator -= generator.conj().T
    x = x @ expm(generator)
    return OrbitalState((inverse @ x[:, :3],), ([2, 1, 1],))


class MolecularMixedOccupationTests(unittest.TestCase):
    def test_two_one_fixed_x_force_matches_energy_difference(self):
        provider = PySCFProvider(lih_mean_field())
        state = lih_two_one_state(provider)
        force = provider.forces(state)
        origin, overlap = provider.positions, provider.overlap
        for ia in range(2):
            energies = []
            for sign in (1, -1):
                trial = deepcopy(provider)
                positions = origin.copy()
                positions[ia, 2] += sign * 1e-4
                trial.prepare_geometry(positions)
                energies.append(trial.energy(transport(state, overlap, trial.overlap)))
            with self.subTest(atom=ia):
                self.assertAlmostEqual(force[ia, 2], -(energies[0] - energies[1]) / 2e-4, delta=2e-7)

    def test_moving_basis_two_one_pt_matches_small_step_orbital_cn(self):
        provider = PySCFProvider(lih_mean_field())
        state = lih_two_one_state(provider)
        end = provider.positions.copy()
        end[1, 2] += .002
        results = []
        for variant in ('pt_cn', 'cn_accelerated'):
            run = LocalExecution([deepcopy(provider)], [state], alphas=[2.],
                                 options=CNOptions(variant, 1e-12, 120), time=3.)
            frame = run.advance(end, .02, n_substeps=20)
            self.assertLess(run.states[0].orthogonality_error(run.providers[0].overlap), 1e-12)
            results.append((run.states[0].density(), frame.observation))
        assert_allclose(results[0][0], results[1][0], atol=2e-7, rtol=0)
        assert_allclose(results[0][1].energies, results[1][1].energies, atol=2e-8, rtol=0)
        assert_allclose(results[0][1].forces, results[1][1].forces, atol=2e-7, rtol=0)


if __name__ == '__main__':
    unittest.main()

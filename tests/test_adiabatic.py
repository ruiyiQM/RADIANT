# Copyright (c) 2026 Ruiyi Zhou
# SPDX-License-Identifier: MIT
"""Adiabatic following on an exactly solvable, rotating two-level path."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from radiant.core.propagation import CNOptions, crank_nicolson
from radiant.core.state import OrbitalState


class AdiabaticFollowingTests(unittest.TestCase):
    rate = .2
    duration = 4.
    diagonal = np.diag([-.5, .5])
    generator = np.array([[0., -1.], [1., 0.]])
    initial = np.array([[1.], [0.]], complex)

    def rotation(self, time):
        return np.cos(self.rate*time)*np.eye(2) + np.sin(self.rate*time)*self.generator

    def hamiltonian(self, time):
        rotation = self.rotation(time)
        return rotation @ self.diagonal @ rotation.T

    def exact(self, time, alpha):
        # In the rotating frame, i phi_dot = (alpha D - i U^T U_dot) phi.
        effective = alpha*self.diagonal - 1j*self.rate*self.generator
        return self.rotation(time) @ expm(-1j*time*effective) @ self.initial

    def trajectory(self, variant, alpha, steps):
        state = OrbitalState((self.initial,), ([2],))
        overlap = np.eye(2)
        options = CNOptions(variant, 2e-12, 120)
        dt = self.duration/steps
        densities = []
        for step in range(steps):
            state, diagnostic = crank_nicolson(state, overlap,
                lambda dm, time: self.hamiltonian(time), step*dt, dt,
                alpha=alpha, options=options)
            self.assertLessEqual(diagnostic.residual, options.tolerance)
            self.assertLess(diagnostic.orthogonality_error, 1e-10)
            densities.append(state.density())
        return np.arange(1, steps+1)*dt, np.array(densities)

    def density_errors(self, times, densities, alpha):
        reference = [self.exact(time, alpha) for time in times]
        expected = np.array([2*c @ c.conj().T for c in reference])
        return np.linalg.norm(densities-expected, axis=(1, 2))

    def test_rotating_eigenstate_density_has_second_order_convergence(self):
        alpha = 4.
        reference = solve_ivp(lambda time, c: -1j*alpha*self.hamiltonian(time) @ c,
            (0., self.duration), self.initial[:, 0], method="DOP853",
            rtol=3e-13, atol=1e-14, max_step=.02)
        self.assertTrue(reference.success, reference.message)
        assert_allclose(reference.y[:, -1], self.exact(self.duration, alpha)[:, 0],
                        atol=2e-11, rtol=0)
        for variant in ("cn_accelerated", "pt_cn"):
            errors = []
            for steps in (32, 64, 128):
                times, densities = self.trajectory(variant, alpha, steps)
                errors.append(self.density_errors(times, densities, alpha).max())
            with self.subTest(variant=variant, errors=errors):
                self.assertLess(errors[-1], .004)
                assert_allclose(np.array(errors[:-1])/errors[1:], 4., rtol=.05)

    def test_resolved_large_alpha_recovers_adiabatic_following(self):
        for variant in ("cn_accelerated", "pt_cn"):
            maxima = []
            for alpha in (1., 4., 16.):
                # Resolve the faster electronics while keeping the physical path fixed.
                times, densities = self.trajectory(variant, alpha, int(32*alpha))
                excited = np.array([self.rotation(time)[:, 1] for time in times])
                leakage = np.einsum("ni,nij,nj->n", excited.conj(), densities, excited).real/2
                bound = self.rate**2/((alpha/2)**2+self.rate**2)
                with self.subTest(variant=variant, alpha=alpha):
                    # Small excitation alone can hide an incorrect propagated density.
                    self.assertLess(self.density_errors(times, densities, alpha).max(), .004)
                    self.assertGreaterEqual(leakage.min(), -1e-12)
                    assert_allclose(leakage.max(), bound, rtol=.02, atol=1e-7)
                maxima.append(leakage.max())
            self.assertTrue(all(left > right for left, right in zip(maxima, maxima[1:])))


if __name__ == "__main__":
    unittest.main()

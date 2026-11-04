import unittest

import numpy as np

from radiant import cayley_frequency, inverse_cayley_frequency


def crank_nicolson_step(hamiltonian, dt):
    """Return the Crank-Nicolson propagator for a constant Hamiltonian."""
    identity = np.eye(hamiltonian.shape[0])
    lhs = identity + 0.5j * dt * hamiltonian
    rhs = identity - 0.5j * dt * hamiltonian
    return np.linalg.solve(lhs, rhs)


class CayleyFrequencyTest(unittest.TestCase):

    def test_matches_explicit_crank_nicolson_phase(self):
        omega = 0.25
        hamiltonian = np.diag([0.0, omega])
        for dt in (0.05, 1.0, 5.0, 15.0):
            step = crank_nicolson_step(hamiltonian, dt)
            # Relative phase between the two levels after one step.
            phase = -np.angle(step[1, 1] / step[0, 0])
            self.assertAlmostEqual(phase / dt, cayley_frequency(omega, dt),
                                   places=12)
            # The propagator is unitary, so the amplitude is not damped.
            self.assertAlmostEqual(abs(step[1, 1]), 1.0, places=12)

    def test_small_step_limit_and_error(self):
        omega = np.array([0.1, 0.25, 0.5])
        dt = 1e-3
        relative = 1.0 - cayley_frequency(omega, dt) / omega
        np.testing.assert_allclose(relative, (omega * dt) ** 2 / 12.0,
                                   rtol=1e-4)

    def test_bounded_by_nyquist(self):
        dt = 5.0
        values = cayley_frequency(np.array([1e2, 1e4, 1e6]), dt)
        self.assertTrue(np.all(values < np.pi / dt))

    def test_inverse_round_trip(self):
        omega = np.linspace(-0.6, 0.6, 13)
        for dt in (0.5, 5.0, 10.0):
            np.testing.assert_allclose(
                inverse_cayley_frequency(cayley_frequency(omega, dt), dt),
                omega, rtol=1e-12, atol=1e-14)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            cayley_frequency(0.1, 0.0)
        with self.assertRaises(ValueError):
            inverse_cayley_frequency(np.pi, 1.0)


if __name__ == "__main__":
    unittest.main()

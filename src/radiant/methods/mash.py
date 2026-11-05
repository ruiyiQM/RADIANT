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

"""Paper CN+MASH with the public order 0=ground, 1=excited, Sz>0 -> 1.

Adapted from TDMASH v0.1.0 Spin.py. Coefficients retain the branch order;
the public Bloch transformation is (Sx_old, -Sy_old, -Sz_old), including
conversion back to coefficients after the paper pi-MASH frustrated flip.
"""

import numpy as np

from radiant.core.overlap import overlap_coupling


def coefficients_to_spin(coefficients):
    c = np.asarray(coefficients, dtype=complex)
    if c.shape != (2,) or not np.isfinite(c).all() or abs(np.vdot(c, c) - 1) > 1e-10:
        raise ValueError("MASH coefficients must be finite and normalized")
    cross = np.conj(c[0]) * c[1]
    return np.array([2 * cross.real, -2 * cross.imag, abs(c[1])**2 - abs(c[0])**2])


def spin_to_coefficients(spin):
    spin = np.asarray(spin, dtype=float)
    if spin.shape != (3,) or not np.isfinite(spin).all() or abs(np.linalg.norm(spin) - 1) > 1e-10:
        raise ValueError("MASH requires a finite unit Bloch vector")
    x, y, z = spin
    if z > 0:
        c1 = np.sqrt((1 + z) / 2)
        c0 = (x + 1j * y) / (2 * c1)
        return np.array([c0, c1]) * np.exp(-1j * np.angle(c0))
    c0 = np.sqrt(max((1 - z) / 2, 0.))
    return np.array([c0, (x - 1j * y) / (2 * c0)])


def legacy_to_public_spin(spin, *, legacy_state_order):
    """Require an explicit data manifest; never infer an old file's ordering."""
    if tuple(legacy_state_order) == ("ground", "excited"):
        result = np.asarray(spin) * [1, -1, -1]
    elif tuple(legacy_state_order) == ("excited", "ground"):
        result = np.asarray(spin).copy()
    else:
        raise ValueError("Unknown legacy state ordering")
    spin_to_coefficients(result)
    return result


class MASH:
    def __init__(self, initial_spin):
        self.coefficients = spin_to_coefficients(initial_spin)

    @property
    def spin(self):
        return coefficients_to_spin(self.coefficients)

    @property
    def active_state(self):
        return 1 if self.spin[2] > 0 else 0

    def propagate(self, energy_left, energy_right, overlap, dt):
        """Paper half-energy / full-overlap / half-energy physical-time split."""
        left, right = np.asarray(energy_left), np.asarray(energy_right)
        if left.shape != (2,) or right.shape != (2,) or not np.isfinite([left, right]).all():
            raise ValueError("MASH requires two finite physical energies")
        q, _, _ = overlap_coupling(overlap, dt)
        candidate = np.exp(-.5j * dt * (left - left[0])) * self.coefficients
        candidate = q.conj().T @ candidate
        candidate *= np.exp(-.5j * dt * (right - right[0]))
        coefficients_to_spin(candidate)
        self.coefficients = candidate

    def resolve_hop(self, momentum, direction, energies, masses, *, initial_state):
        """Paper real-axis rescaling; returns momentum, outcome and parallel KE.

        shortcut: only the paper real-axis direction is established, derive and
        review the general complex momentum direction before extending it.
        """
        from radiant.core.coupling import ZeroNACError
        momentum, direction = np.asarray(momentum), np.asarray(direction)
        masses, energies = np.asarray(masses), np.asarray(energies)
        if (momentum.shape != (len(masses), 3) or direction.shape != momentum.shape
                or masses.ndim != 1 or np.any(masses <= 0) or energies.shape != (2,)
                or not all(np.isfinite(a).all() for a in (momentum, direction, masses, energies))
                or initial_state not in (0, 1) or self.active_state == initial_state):
            raise ValueError("Invalid MASH hop proposal")
        weighted = momentum / np.sqrt(masses[:, None])
        axis = direction.real / np.sqrt(masses[:, None])
        norm = np.linalg.norm(axis)
        if norm <= 64 * np.finfo(float).eps:
            raise ZeroNACError("The paper real NAC axis is undefined")
        axis /= norm
        parallel = np.sum(weighted * axis)
        kinetic = .5 * parallel**2
        gap = energies[1 - initial_state] - energies[initial_state]
        if kinetic < gap:
            self.coefficients = spin_to_coefficients(-self.spin)
            return momentum.copy(), "frustrated", kinetic
        sign = parallel if parallel != 0 else axis.flat[np.argmax(np.abs(axis))]
        parallel_new = np.copysign(np.sqrt(max(parallel**2 - 2 * gap, 0.)), sign)
        return ((weighted + (parallel_new - parallel) * axis) * np.sqrt(masses[:, None]),
                "accepted", kinetic)

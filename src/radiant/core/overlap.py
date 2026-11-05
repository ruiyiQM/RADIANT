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

"""Restricted two-state DTO/singlet overlaps adapted from TDMASH v0.1.0.

Rows are old/minus states; columns are new/plus states. Same-time Gram
matrices use the same formula but are diagnostics, never time couplings.
"""

import numpy as np
from scipy.linalg import logm

from .state import OrbitalState


class GaugeFailure(ValueError):
    """An orbital gauge or overlap logarithm is numerically undefined."""


def align_phases(coefficients, reference, cross):
    diagonal = np.einsum("ij,ij->j", reference.conj(), cross @ coefficients)
    if np.any(np.abs(diagonal) <= 64 * np.finfo(float).eps):
        raise GaugeFailure("Corresponding orbital overlap vanishes")
    return coefficients * np.exp(-1j * np.angle(diagonal))


def dto_pair(states, overlap, *, reference=None, cross=None, align_all=False):
    """Order the two singly occupied orbitals as hole, particle.

    The normal paper path aligns active DTO orbitals to the last frame. FD
    aligns every orbital in both branches to the same central reference.
    """
    ground, excited = states
    if len(ground.coefficients) != 1 or len(excited.coefficients) != 1:
        raise ValueError("Unrestricted two-state DTO has not been validated")
    cg, ce = ground.coefficients[0], excited.coefficients[0].copy()
    fg, fe = ground.occupations[0], excited.occupations[0]
    if np.any(fg == 1) or np.sum(fe == 1) != 2 or np.sum(fg) != np.sum(fe):
        raise ValueError("DTO requires a closed-shell ground and a two-active-orbital singlet")
    active = fe == 1
    projection = cg[:, fg == 2].conj().T @ overlap @ ce[:, active]
    values, vectors = np.linalg.eigh(projection.conj().T @ projection)
    # This rejects numerical degeneracy, not a calibrated physical DTO gap.
    if values[1] - values[0] <= 64 * np.finfo(float).eps * max(1., np.max(np.abs(values))):
        raise GaugeFailure("DTO hole/particle eigenvalues are degenerate")
    ce[:, active] = ce[:, active] @ vectors[:, ::-1]
    if reference is None:
        ce[:, active] = align_phases(ce[:, active], excited.coefficients[0][:, active], overlap)
    elif align_all:
        cg = align_phases(cg, reference[0].coefficients[0], cross)
        ce = align_phases(ce, reference[1].coefficients[0], cross)
    else:
        ce[:, active] = align_phases(ce[:, active], reference[1].coefficients[0][:, active], cross)
    return (OrbitalState((cg,), (fg,)), OrbitalState((ce,), (fe,)))


def _determinants(state):
    if len(state.coefficients) != 1:
        raise ValueError("Only restricted singlet overlaps are supported")
    c, f = state.coefficients[0], state.occupations[0]
    closed, active = c[:, f == 2], c[:, f == 1]
    if active.shape[1] == 0:
        return ((closed, closed, 1.),)
    if active.shape[1] != 2:
        raise ValueError("A restricted open singlet needs exactly two singly occupied orbitals")
    h = np.column_stack((closed, active[:, 0]))
    p = np.column_stack((closed, active[:, 1]))
    return ((h, p, 1 / np.sqrt(2)), (p, h, 1 / np.sqrt(2)))


def singlet_overlap(left, right, cross):
    """Closed/open sqrt(2) and both open/open determinant products included."""
    if np.sum(left.occupations[0]) != np.sum(right.occupations[0]):
        raise ValueError("Singlet electron counts differ")
    return sum(wl * wr * np.linalg.det(al.conj().T @ cross @ ar)
               * np.linalg.det(bl.conj().T @ cross @ br)
               for al, bl, wl in _determinants(left)
               for ar, br, wr in _determinants(right))


def pair_overlap(left, right, cross):
    return np.array([[singlet_overlap(a, b, cross) for b in right] for a in left])


def overlap_coupling(overlap, interval):
    """Return polar Q, singular values and principal log(Q)/physical interval.

    Complex diagonal terms are retained. Numerical rank loss and a principal
    logarithm branch cut are explicit failures; no real-part clipping is used.
    """
    overlap = np.asarray(overlap, dtype=complex)
    if overlap.shape != (2, 2) or not np.isfinite(overlap).all() or not np.isfinite(interval) or interval == 0:
        raise ValueError("Require a finite two-state overlap and nonzero interval")
    u, singular, vh = np.linalg.svd(overlap)
    if singular[-1] <= 64 * np.finfo(float).eps * max(1., singular[0]):
        raise GaugeFailure("Two-state overlap lost numerical rank")
    q = u @ vh
    phases = np.angle(np.linalg.eigvals(q))
    if np.any(np.pi - np.abs(phases) <= 64 * np.finfo(float).eps):
        raise GaugeFailure("Overlap logarithm is on its principal branch cut")
    coupling = logm(q) / interval
    if np.max(np.abs(coupling + coupling.conj().T)) > 1e-10 / abs(interval):
        raise GaugeFailure("Overlap logarithm is not anti-Hermitian")
    return q, singular, coupling

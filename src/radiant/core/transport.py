# Copyright (c) 2026 Ruiyi Zhou
"""Symmetric Lowdin coordinates without silently discarding AO directions."""

import numpy as np
from scipy.linalg import solve_sylvester

from .state import OrbitalState


def lowdin(overlap):
    """Return S**(-1/2) and S**(1/2), rejecting rank loss."""
    s = np.asarray(overlap)
    if (s.ndim != 2 or s.shape[0] != s.shape[1] or not np.isfinite(s).all()
            or not np.allclose(s, s.conj().T, rtol=0, atol=1e-12)):
        raise ValueError("AO overlap must be finite and Hermitian")
    values, vectors = np.linalg.eigh(s)
    if np.min(values) <= 1e-15:
        raise ValueError("AO overlap is not positive definite; truncation is unsupported")
    return ((vectors / np.sqrt(values)) @ vectors.conj().T,
            (vectors * np.sqrt(values)) @ vectors.conj().T)


def transport(state, overlap_left, overlap_right):
    """Keep X=S**(1/2) C fixed; this is not the cross-geometry overlap."""
    if np.shape(overlap_left) != np.shape(overlap_right):
        raise ValueError("AO dimensions changed during transport")
    _, half_left = lowdin(overlap_left)
    inverse_right, _ = lowdin(overlap_right)
    return OrbitalState(tuple(inverse_right @ half_left @ c for c in state.coefficients),
                        state.occupations)


def lowdin_pulay(overlap, density, fock):
    """Return W such that the fixed-X basis response is -Tr(W dS).

    Differentiating S**(-1/2) uses the Sylvester derivative of S**(1/2).
    This is distinct from the stationary SCF energy-weighted density.
    """
    inverse, half = lowdin(overlap)
    rhs = density @ fock @ inverse + inverse @ fock @ density
    return solve_sylvester(half, half, rhs)

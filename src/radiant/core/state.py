# Copyright (c) 2026 Ruiyi Zhou
"""Validated electronic and nuclear coordinates (atomic units throughout)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class OrbitalState:
    """One restricted spatial block or two independent alpha/beta blocks.

    Restricted occupations may be 0, 1 or 2; per-spin occupations are 0 or 1.
    No zero padding is needed for empty or unequal spin populations.
    """

    coefficients: tuple
    occupations: tuple

    def __post_init__(self):
        self.coefficients = tuple(np.array(c, dtype=complex, copy=True)
                                  for c in self.coefficients)
        self.occupations = tuple(np.array(f, dtype=float, copy=True)
                                for f in self.occupations)
        if len(self.coefficients) not in (1, 2) or len(self.occupations) != len(self.coefficients):
            raise ValueError("Use one restricted or two unrestricted blocks")
        if self.coefficients[0].ndim != 2:
            raise ValueError("Orbital blocks must be matrices")
        nao = self.coefficients[0].shape[0]
        for c, f in zip(self.coefficients, self.occupations):
            if c.ndim != 2 or f.ndim != 1 or c.shape != (nao, f.size):
                raise ValueError("Orbital/occupation dimensions do not match")
            if not np.isfinite(c).all() or not np.isfinite(f).all():
                raise ValueError("Orbitals and occupations must be finite")
            if not np.isin(f, (0, 1, 2) if len(self.coefficients) == 1 else (0, 1)).all():
                raise ValueError("Fractional occupations are not supported")

    def density(self):
        blocks = tuple((c * f) @ c.conj().T
                       for c, f in zip(self.coefficients, self.occupations))
        return blocks[0] if len(blocks) == 1 else np.array(blocks)

    def copy(self):
        return OrbitalState(self.coefficients, self.occupations)

    def orthogonality_error(self, overlap):
        return max((float(np.max(np.abs(c.conj().T @ overlap @ c - np.eye(c.shape[1])), initial=0))
                    for c in self.coefficients), default=0.)


@dataclass
class NuclearState:
    positions: np.ndarray
    momenta: np.ndarray
    masses: np.ndarray

    def __post_init__(self):
        self.positions = np.array(self.positions, dtype=float, copy=True)
        self.momenta = np.array(self.momenta, dtype=float, copy=True)
        self.masses = np.array(self.masses, dtype=float, copy=True)
        if (self.masses.ndim != 1 or self.positions.shape != (self.masses.size, 3)
                or self.momenta.shape != self.positions.shape):
            raise ValueError("Nuclear coordinates, momenta and masses do not match")
        if (not all(np.isfinite(a).all() for a in (self.positions, self.momenta, self.masses))
                or np.any(self.masses <= 0)):
            raise ValueError("Nuclear data must be finite with positive masses")

    @property
    def kinetic_energy(self):
        return float(np.sum(self.momenta**2 / self.masses[:, None]) / 2)

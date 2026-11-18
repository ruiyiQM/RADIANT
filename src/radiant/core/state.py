# Copyright (c) 2026 Ruiyi Zhou
"""Validated electronic and nuclear coordinates (atomic units throughout)."""

from dataclasses import dataclass

import numpy as np

from .arrays import array_module


@dataclass
class OrbitalState:
    """One restricted spatial block or two independent alpha/beta blocks.

    Restricted occupations may be 0, 1 or 2; per-spin occupations are 0 or 1.
    No zero padding is needed for empty or unequal spin populations.
    """

    coefficients: tuple
    occupations: tuple

    def __post_init__(self):
        xp = array_module(self.coefficients[0]) if self.coefficients else np
        self.coefficients = tuple(xp.array(c, dtype=complex, copy=True)
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
            if not np.isfinite(f).all():
                raise ValueError("Orbitals and occupations must be finite")
            if not np.isin(f, (0, 1, 2) if len(self.coefficients) == 1 else (0, 1)).all():
                raise ValueError("Fractional occupations are not supported")
        if not xp.all(xp.stack([xp.isfinite(c).all() for c in self.coefficients])):
            raise ValueError("Orbitals and occupations must be finite")
        self._weights = tuple(xp.asarray(f) for f in self.occupations)
        if xp is not np:
            for f in self.occupations:
                f.flags.writeable = False

    def density(self):
        xp = array_module(self.coefficients[0])
        blocks = tuple((c * f) @ c.conj().T
                       for c, f in zip(self.coefficients, self._weights))
        return blocks[0] if len(blocks) == 1 else xp.stack(blocks)

    def copy(self):
        return self.with_coefficients(self.coefficients)

    def __deepcopy__(self, memo):
        result = self.copy()
        memo[id(self)] = result
        return result

    def with_coefficients(self, coefficients):
        xp = array_module(self.coefficients[0])
        if xp is np:
            return OrbitalState(coefficients, self.occupations)
        result = object.__new__(OrbitalState)
        result.coefficients = tuple(xp.array(c, dtype=complex, copy=True) for c in coefficients)
        if len(result.coefficients) != len(self.coefficients) or any(
                c.shape != old.shape
                for c, old in zip(result.coefficients, self.coefficients)):
            raise ValueError("Replacement orbitals must retain finite block dimensions")
        if not xp.all(xp.stack([xp.isfinite(c).all() for c in result.coefficients])):
            raise ValueError("Replacement orbitals must retain finite block dimensions")
        result.occupations, result._weights = self.occupations, self._weights
        return result

    def orthogonality_error(self, overlap):
        xp = array_module(self.coefficients[0])
        errors = [xp.max(abs(c.conj().T @ overlap @ c - xp.eye(c.shape[1])))
                  for c in self.coefficients if c.shape[1]]
        return float(xp.max(xp.stack(errors))) if errors else 0.


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

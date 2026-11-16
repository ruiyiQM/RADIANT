# Copyright (c) 2026 Ruiyi Zhou
"""GPU4PySCF Fock/energy for pure molecular LDA/GGA, with CPU fixed-X forces.

Accept a converged CPU RKS/UKS object. Orbitals, propagation, overlap and NAC
remain on the CPU. CuPy and GPU4PySCF are needed only at the first evaluation;
device failures are propagated, never replaced by a CPU Fock.
"""

from pyscf import dft
import numpy as np

from .pyscf import PySCFProvider


class GPU4PySCFProvider(PySCFProvider):
    """Pure LDA/GGA provider; construct a new provider to change model settings."""

    def __init__(self, mean_field, *, density_convention="complex"):
        super().__init__(mean_field, density_convention=density_convention)
        if (not isinstance(self.mf, dft.rks.KohnShamDFT)
                or self.mf._numint._xc_type(self.mf.xc) not in ("LDA", "GGA")
                or any(self.mf._numint.rsh_and_hybrid_coeff(self.mf.xc, spin=self.mf.mol.spin))):
            raise ValueError("GPU Focks require pure LDA/GGA RKS or UKS")
        self._gpu_mf = None

    def __deepcopy__(self, memo):
        # Device caches are disposable; snapshots must not clone device buffers or contexts.
        if self._gpu_mf is not None:
            memo[id(self._gpu_mf)] = None
        return super().__deepcopy__(memo)

    def prepare_geometry(self, positions):
        molecule = self.mf.mol
        overlap = super().prepare_geometry(positions)
        if self.mf.mol is not molecule:
            self._gpu_mf = None
        return overlap

    def _gpu_mean_field(self, density):
        import cupy
        if self._gpu_mf is None:
            self.mf.initialize_grids(self.mf.mol, density)
            candidate = self.mf.to_gpu()
            # to_gpu resets grids: reuse the CPU quadrature, including its density pruning.
            candidate.grids.coords = cupy.asarray(self.mf.grids.coords)
            candidate.grids.weights = cupy.asarray(self.mf.grids.weights)
            candidate.chkfile = None
            self._gpu_mf = candidate
        return self._gpu_mf

    def fock(self, density, time=0.):
        if not np.isfinite(time):
            raise ValueError("Physical time must be finite")
        density = self._density(density)
        import cupy
        # In a real AO basis pure LDA/GGA and Coulomb depend only on Re(P).
        mean_field = self._gpu_mean_field(density.real)
        potential = mean_field.get_veff(dm=cupy.asarray(density.real, order="C"))
        return self.mf.get_hcore() + cupy.asnumpy(potential)

    def energy(self, state):
        density = self._density(state.density())
        import cupy
        mean_field = self._gpu_mean_field(density.real)
        return float(mean_field.energy_tot(dm=cupy.asarray(density.real, order="C"),
                                          h1e=cupy.asarray(self.mf.get_hcore())))

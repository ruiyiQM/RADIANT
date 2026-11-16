"""Resident CuPy orbitals and native GPU4PySCF matrices and fixed-X forces.

Electronic matrices stay on device during dynamics. Model/grid initialization
metadata, small endpoint results and persistent checkpoints remain host boundaries.
"""

from copy import copy, deepcopy

import numpy as np
from pyscf import gto

from .gpu_pyscf import GPU4PySCFProvider
from ..core.arrays import array_module, to_host
from ..core.state import OrbitalState
from ..core.transport import lowdin, lowdin_pulay


class GPUResidentProvider(GPU4PySCFProvider):
    device_resident = True

    def __init__(self, mean_field, *, density_convention="complex"):
        super().__init__(mean_field, density_convention=density_convention, gpu_forces=True)
        self._device_overlap = self._device_hcore = None
        self._device_nuclear = None

    @property
    def array_module(self):
        import cupy
        return cupy

    def _gpu_mean_field(self, density=None):
        if self._gpu_mf is None:
            # No SCF orbitals are needed by explicit-density GPU kernels.
            template = copy(self.mf)
            template.mo_coeff = template.mo_occ = template.mo_energy = None
            candidate = template.to_gpu()
            for key in ("coords", "weights", "atm_idx", "quadrature_weights"):
                value = getattr(self.mf.grids, key, None)
                if value is not None:
                    setattr(candidate.grids, key, self.array_module.asarray(value))
            candidate.chkfile = None
            self._gpu_mf = candidate
        return self._gpu_mf

    @property
    def overlap(self):
        if self._device_overlap is None:
            self._device_overlap = self.array_module.asarray(self._gpu_mean_field().get_ovlp())
        return self._device_overlap

    @property
    def hcore(self):
        if self._device_hcore is None:
            self._device_hcore = self.array_module.asarray(self._gpu_mean_field().get_hcore())
        return self._device_hcore

    def initial_state(self):
        state = super().initial_state()
        return OrbitalState(tuple(self.array_module.asarray(c) for c in state.coefficients), state.occupations)

    def prepare_geometry(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != self.positions.shape or not np.isfinite(positions).all():
            raise ValueError("Invalid molecular geometry")
        if np.array_equal(positions, self.positions):
            return self.overlap
        mol = self.mf.mol.set_geom_(positions, unit="Bohr", inplace=False)
        if mol.ao_labels() != self.mf.mol.ao_labels():
            raise ValueError("AO identity changed")
        candidate = deepcopy(self.mf)
        candidate.reset(mol)
        before = (self.mf, self._gpu_mf, self._device_overlap, self._device_hcore, self._device_nuclear)
        native = None
        if self._gpu_mf is not None:
            native = copy(self._gpu_mf)
            native.grids = self._gpu_mf.grids.copy()
            native.nlcgrids = self._gpu_mf.nlcgrids.copy()
            native._numint = copy(self._gpu_mf._numint)
            native.reset(mol)
        self.mf, self._gpu_mf = candidate, native
        self._device_overlap = self._device_hcore = self._device_nuclear = None
        try:
            lowdin(self.overlap)
        except BaseException:
            self.mf, self._gpu_mf, self._device_overlap, self._device_hcore, self._device_nuclear = before
            raise
        return self.overlap

    @staticmethod
    def cross_overlap(left_mol, right_mol):
        from gpu4pyscf.pbc.gto.int1e import int1e_ovlp
        combined = gto.conc_mol(left_mol, right_mol)
        return int1e_ovlp(combined)[:left_mol.nao, left_mol.nao:]

    def _density(self, density):
        xp = self.array_module
        if array_module(density) is not xp:
            raise ValueError("Resident Focks require a device density")
        n = self.mf.mol.nao
        shape = (2, n, n) if self.unrestricted else (n, n)
        if (density.shape != shape or not xp.isfinite(density).all() or
                not xp.allclose(density, density.conj().swapaxes(-1, -2), atol=1e-12, rtol=0)):
            raise ValueError("Density must be finite and Hermitian with the model's spin shape")
        return density.real if self.density_convention == "paper_real" else density

    def fock(self, density, time=0.):
        if not np.isfinite(time):
            raise ValueError("Physical time must be finite")
        density = self._density(density)
        return self.hcore + self._gpu_mean_field().get_veff(dm=density.real)

    def _nuclear(self):
        if self._device_nuclear is None:
            xp = self.array_module
            coords = xp.asarray(self.positions)
            charges = xp.asarray(self.mf.mol.atom_charges())
            delta = coords[:, None, :] - coords[None, :, :]
            distance = xp.linalg.norm(delta, axis=2)
            xp.fill_diagonal(distance, xp.inf)
            pairs = charges[:, None] * charges[None, :]
            self._device_nuclear = (.5 * xp.sum(pairs / distance),
                                    -xp.sum((pairs / distance**3)[:, :, None] * delta, axis=1))
        return self._device_nuclear

    def energy(self, state, *, hcore=None, potential=None):
        density = self._density(state.density()).real
        if potential is None:
            potential = self._gpu_mean_field().get_veff(dm=density)
        hcore = self.hcore if hcore is None else hcore
        total = density.sum(axis=0) if self.unrestricted else density
        one = self.array_module.einsum("ij,ji->", hcore, total).real
        return float(one + potential.ecoul + potential.exc + self._nuclear()[0])

    def evaluate(self, state, *, forces=True):
        density = self._density(state.density())
        potential = self._gpu_mean_field().get_veff(dm=density.real)
        energy = self.energy(state, hcore=self.hcore, potential=potential)
        force = self.forces(state, fock=self.hcore + potential) if forces else None
        return energy, force

    def forces(self, state, *, fock=None):
        xp = self.array_module
        density = self._density(state.density())
        fock = self.fock(density) if fock is None else fock
        dms = density if self.unrestricted else density[None]
        focks = fock if self.unrestricted else fock[None]
        metric = sum(lowdin_pulay(self.overlap, d, h) for d, h in zip(dms, focks))
        gradient = self._gpu_mean_field().nuc_grad_method()
        gradient.grid_response = True
        core = gradient._hcore_energy(dms.real.sum(axis=0), metric.real)
        two = gradient.energy_ee(dm=density.real)
        result = -(xp.asarray(core) + xp.asarray(two) + self._nuclear()[1])
        if result.shape != self.positions.shape or not xp.isfinite(result).all():
            raise ValueError("Invalid resident fixed-X force")
        return to_host(result)

    def capture_grids(self):
        """Materialize quadrature only at the persistent-checkpoint boundary."""
        if self._gpu_mf is not None:
            exported = {}
            for key in ("coords", "weights", "atm_idx", "quadrature_weights"):
                value = getattr(self._gpu_mf.grids, key, None)
                if value is not None:
                    exported[key] = to_host(value)
            for key, value in exported.items():
                setattr(self.mf.grids, key, value)

    def derivative_integrals(self):
        from gpu4pyscf.pbc.gto.int1e import int1e_ipovlp
        xp = self.array_module
        mol = self.mf.mol
        ip = int1e_ipovlp(mol)
        basis = xp.zeros((mol.natm, 3, mol.nao, mol.nao))
        for atom, (_, _, lo, hi) in enumerate(mol.aoslice_by_atom()):
            basis[atom, :, :, lo:hi] = -ip.transpose(0, 2, 1)[:, :, lo:hi]
        return basis, basis + basis.transpose(0, 1, 3, 2)

    def response_model(self, mo, occ):
        from .gpu_response import response_model
        return response_model(self, mo, occ)

    def response_h1(self, model, mo, occ):
        from .gpu_response import ao_derivative
        return ao_derivative(self, model, mo, occ)

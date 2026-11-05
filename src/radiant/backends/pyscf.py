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

# Copyright (c) 2026 Ruiyi Zhou
"""Molecular RHF/RKS/UHF/UKS Fock, energy and fixed-Lowdin-coordinate force.

PySCF is a local provider, not a propagator. No SCF is run after construction.
The optional paper_real convention explicitly reproduces the restricted
TDMASH v0.1.0 density approximation; complex is the native PySCF functional.
"""

from copy import deepcopy

import numpy as np
from pyscf import dft, gto, scf
from pyscf.lib import logger

from radiant.core.state import OrbitalState
from radiant.core.transport import lowdin, lowdin_pulay


class PySCFProvider:
    def __init__(self, mean_field, *, density_convention="complex"):
        if not isinstance(mean_field, (scf.hf.RHF, scf.uhf.UHF)) or isinstance(mean_field, scf.rohf.ROHF):
            raise ValueError("Only molecular RHF/RKS/UHF/UKS are supported")
        if not mean_field.converged:
            raise ValueError("Initial SCF must have converged")
        if density_convention not in ("complex", "paper_real"):
            raise ValueError("Unknown density convention")
        self.unrestricted = isinstance(mean_field, scf.uhf.UHF)
        if self.unrestricted and density_convention == "paper_real":
            raise ValueError("paper_real is a restricted paper convention")
        self.mf = deepcopy(mean_field)
        self.density_convention = density_convention
        self._validate_model()

    def _validate_model(self):
        if self.mf.mol.has_ecp() or self.mf.mol._pseudo:
            raise ValueError("ECP/pseudopotential forces are not validated")
        if hasattr(self.mf, "with_df") or getattr(self.mf, "with_x2c", None) is not None:
            raise ValueError("Density fitting and relativistic forces are not validated")
        if isinstance(self.mf, dft.rks.KohnShamDFT):
            if self.mf.do_nlc() or self.mf._numint._xc_type(self.mf.xc) not in ("LDA", "GGA", "HF"):
                raise ValueError("Only LDA/GGA and their hybrids are validated")

    @property
    def positions(self):
        return self.mf.mol.atom_coords(unit="Bohr")

    @property
    def overlap(self):
        return self.mf.get_ovlp()

    def initial_state(self):
        coeff = self.mf.mo_coeff if self.unrestricted else (self.mf.mo_coeff,)
        occ = self.mf.mo_occ if self.unrestricted else (self.mf.mo_occ,)
        return OrbitalState(tuple(c[:, f > 0] for c, f in zip(coeff, occ)),
                            tuple(f[f > 0] for f in occ))

    def prepare_geometry(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != self.positions.shape or not np.isfinite(positions).all():
            raise ValueError("Invalid molecular geometry")
        if np.array_equal(positions, self.positions):
            return self.overlap
        mol = self.mf.mol.set_geom_(positions, unit="Bohr", inplace=False)
        candidate = deepcopy(self.mf)
        candidate.reset(mol)
        overlap = candidate.get_ovlp()
        lowdin(overlap)
        if mol.ao_labels() != self.mf.mol.ao_labels():
            raise ValueError("AO identity changed")
        self.mf = candidate
        return overlap

    def _density(self, density):
        dm = np.asarray(density)
        n = self.mf.mol.nao
        expected = (2, n, n) if self.unrestricted else (n, n)
        if (dm.shape != expected or not np.isfinite(dm).all()
                or not np.allclose(dm, dm.conj().swapaxes(-1, -2), atol=1e-12, rtol=0)):
            raise ValueError("Density must be finite and Hermitian with the model's spin shape")
        return dm.real if self.density_convention == "paper_real" else dm

    def fock(self, density, time=0.):
        """Physical H[P], evaluated with the joint alpha/beta trial density."""
        if not np.isfinite(time):
            raise ValueError("Physical time must be finite")
        dm = self._density(density)
        return self.mf.get_hcore() + self.mf.get_veff(dm=dm)

    def energy(self, state):
        dm = self._density(state.density())
        return float(self.mf.energy_tot(dm=dm).real)

    def forces(self, state):
        """Negative dE/dR at fixed X=S**(1/2) C, including DFT grid response.

        PySCF's real AO derivative JK routine needs separate real/imaginary
        contractions; passing a complex density directly silently drops it.
        """
        dm0 = self._density(state.density())
        fock = self.fock(dm0)
        dm = dm0 if self.unrestricted else dm0[None]
        focks = fock if self.unrestricted else fock[None]
        grad = self.mf.nuc_grad_method()
        log = logger.new_logger(grad)
        if isinstance(self.mf, dft.rks.KohnShamDFT):
            grad.grid_response = True
        vhf = grad.get_veff(self.mf.mol, dm0.real)
        if np.any(dm0.imag):
            # The scalar density/XC potential is real; only exchange needs Im(P).
            if isinstance(self.mf, dft.rks.KohnShamDFT):
                omega, alpha, hybrid = self.mf._numint.rsh_and_hybrid_coeff(
                    self.mf.xc, spin=self.mf.mol.spin)
                imaginary = np.zeros((2 if self.unrestricted else 1, 3, *dm0.shape[-2:]))
                if hybrid:
                    imaginary += hybrid * np.asarray(grad.get_k(self.mf.mol, dm0.imag)).reshape(imaginary.shape)
                if omega and alpha != hybrid:
                    imaginary += (alpha - hybrid) * np.asarray(
                        grad.get_k(self.mf.mol, dm0.imag, omega=omega)).reshape(imaginary.shape)
                imaginary *= -1 if self.unrestricted else -.5
            else:
                imaginary = np.asarray(grad.get_veff(self.mf.mol, dm0.imag)).reshape(
                    (len(dm), 3, *dm0.shape[-2:]))
            derivative = np.asarray(vhf, dtype=complex).reshape(imaginary.shape) + 1j * imaginary
        else:
            derivative = np.asarray(vhf).reshape((len(dm), 3, *dm0.shape[-2:]))
        metric_density = sum(lowdin_pulay(self.overlap, d, h) for d, h in zip(dm, focks))
        hcore_deriv = grad.hcore_generator(self.mf.mol)
        s1 = grad.get_ovlp(self.mf.mol)
        de = np.zeros((self.mf.mol.natm, 3))
        for ia, (_, _, p0, p1) in enumerate(self.mf.mol.aoslice_by_atom()):
            de[ia] += np.einsum("xij,ji->x", hcore_deriv(ia), dm.sum(axis=0)).real
            de[ia] += 2 * np.einsum("sxij,sji->x", derivative[:, :, p0:p1], dm[:, :, p0:p1]).real
            de[ia] -= 2 * np.einsum("xij,ji->x", s1[:, p0:p1], metric_density[:, p0:p1]).real
            de[ia] += grad.extra_force(ia, locals())
        result = -(de + grad.grad_nuc())
        if not np.isfinite(result).all():
            raise ValueError("Non-finite fixed-X force")
        return result

    @staticmethod
    def cross_overlap(left_mol, right_mol):
        return gto.intor_cross("int1e_ovlp", left_mol, right_mol)

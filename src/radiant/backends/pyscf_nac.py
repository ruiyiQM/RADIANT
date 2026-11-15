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


"""Static DTO nonadiabatic coupling from two Z-vector solves (no 3N coupled-perturbed solves).

d = <0|d1> for the restricted closed-shell ground / open-shell singlet pair in the execution's DTO
gauge, with the ground orbitals relaxed by coupled-perturbed KS/HF and the excited orbitals
Loewdin-transported (their relaxation is neglected; it is below 5e-4 of |d| in the LDA tests):
  d = (A_t - B_t)/2 + (A_DTO - (B - B_t))/2,
  A_t, B_t  <0|d1>, <d0|1> with both determinants transported at fixed S**(1/2) C,
  B - B_t   ground relaxation; B = conj(<1|d0>), <1|d0> = Tr(<mu|d nu> t10) + sum_pi U_pi W_pi,
  A_DTO     the DTO pair rotation the ground relaxation drives: M = P^H P, P = C_o^H S [h p],
            changes by M'_ph, so h += eps p, p -= eps* h with eps = M'_ph / (m_h - m_p), and the
            open singlet by sqrt2 (eps D_pp - eps* D_hh).  When h lies in the ground occupied
            space and p outside it, A_DTO = -(B - B_t) and the relaxation enters with weight 1.
The virtual rows of U solve the real-linear system H U = r - c, H = diag(e_a - e_i) + kernel,
symmetric for <a, b> = Re sum conj(a) b; the relaxation is a real-linear functional of U, so its
real and imaginary parts need one Z-vector each, for all 3N coordinates at once.

Requires an energy that depends on Re P only (pure functionals, or density_convention
'paper_real'); this includes complex fractional-occupation-like ground minima. The ground is
treated as stationary: for propagated states the error is first order in their orbital gradient.
"""

from copy import deepcopy

import numpy as np
from pyscf import dft
from scipy.linalg import null_space

from ..core.coupling import NACResult, ZeroNACError
from ..core.overlap import _determinants
from ..core.state import OrbitalState
from ..core.transport import lowdin


def _adjugate(m):
    u, s, vh = np.linalg.svd(m)
    cofactors = np.array([np.prod(np.delete(s, j)) for j in range(len(s))])
    return np.linalg.det(u) * np.linalg.det(vh) * (vh.conj().T * cofactors) @ u.conj().T


def _transition_density(left, right, overlap):
    """AO matrix t with <left| sum_i o(i) |right> = Tr(O t), spin summed."""
    t = 0
    for al, bl, wl in _determinants(left):
        for ar, br, wr in _determinants(right):
            ma, mb = al.conj().T @ overlap @ ar, bl.conj().T @ overlap @ br
            t = t + wl * wr * (np.linalg.det(mb) * ar @ _adjugate(ma) @ al.conj().T
                               + np.linalg.det(ma) * br @ _adjugate(mb) @ bl.conj().T)
    return t


def _derivative_integrals(mol):
    """<mu| d nu/dR_A> and dS/dR_A, both (natm, 3, nao, nao)."""
    ip = mol.intor("int1e_ipovlp", comp=3)                       # <grad mu|nu>
    basis = np.zeros((mol.natm, 3, mol.nao, mol.nao))
    for atom, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        basis[atom, :, :, p0:p1] = -ip.transpose(0, 2, 1)[:, :, p0:p1]
    return basis, basis + basis.transpose(0, 1, 3, 2)


def _transport_generators(overlap, ds):
    """K with dC = K C at fixed S**(1/2) C: K = -S**(-1/2) Y, S**(1/2) Y + Y S**(1/2) = dS."""
    w, v = np.linalg.eigh(overlap)
    root = np.sqrt(w)
    y = np.einsum("pi,axpq,qj->axij", v, ds, v) / (root[:, None] + root[None, :])
    return -np.einsum("pi,axij,qj->axpq", v / root, y, v)


def _uses_real_density(provider):
    if provider.density_convention == "paper_real":
        return True
    if not isinstance(provider.mf, dft.rks.KohnShamDFT):
        return False
    return not any(provider.mf._numint.rsh_and_hybrid_coeff(provider.mf.xc, spin=provider.mf.mol.spin))


class ZVectorNAC:
    """Drop-in for PaperFD: compute(execution) -> NACResult, direction Re(raw)/|Re raw|."""

    def __init__(self, *, tolerance=1e-8, max_iterations=200):
        if not np.isfinite(tolerance) or tolerance <= 0 or int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("Z-vector NAC needs a positive tolerance and iteration limit")
        self.tolerance, self.max_iterations = tolerance, int(max_iterations)

    def compute(self, execution):
        provider = execution.providers[0]
        if len(execution.states) != 2 or getattr(provider, "unrestricted", True):
            raise ValueError("Z-vector NAC needs a restricted PySCF two-state execution")
        if not _uses_real_density(provider):
            raise ValueError("Z-vector NAC needs an energy of Re P only: a pure functional or paper_real")
        ground, gauge_excited = execution.states[0], execution.gauge[1]
        mol, overlap = provider.mf.mol, provider.overlap
        nocc = ground.coefficients[0].shape[1]
        density = ground.density().real
        inverse_half, half = lowdin(overlap)

        # canonical [occupied | complement]; the occupied rotation has determinant 1, so the
        # ground determinant and its phase (the execution gauge) are unchanged
        x = half @ ground.coefficients[0]
        c = inverse_half @ np.column_stack([x, null_space(x.conj().T)])
        fock = provider.fock(density)
        e = np.zeros(c.shape[1])
        for block in (slice(0, nocc), slice(nocc, None)):
            e[block], v = np.linalg.eigh(c[:, block].conj().T @ fock @ c[:, block])
            v[:, 0] *= np.conj(np.linalg.det(v))
            c[:, block] = c[:, block] @ v
        co, cv = c[:, :nocc], c[:, nocc:]
        delta = e[nocc:, None] - e[None, :nocc]

        # transported parts
        gauge = (OrbitalState((co,), ground.occupations), gauge_excited)
        t01 = _transition_density(gauge[0], gauge[1], overlap)
        t10 = _transition_density(gauge[1], gauge[0], overlap)
        basis, ds = _derivative_integrals(mol)
        k = _transport_generators(overlap, ds)
        a_t = np.einsum("axij,ji->ax", basis + overlap @ k, t01)
        w = (co.conj().T @ overlap @ t10 @ overlap @ c).T                      # (nmo, nocc)
        u_t = c.conj().T @ overlap @ k @ co
        b_t = np.einsum("axij,ji->ax", basis, t10) + np.einsum("axpi,pi->ax", u_t, w)   # <1|d0>, transported

        # response at Re P, through its natural orbitals scaled by sqrt(n/2) with occupation 2
        n, vectors = np.linalg.eigh(half @ density @ half)
        keep = n > 1e-10
        mo, occ = (inverse_half @ vectors[:, keep]) * np.sqrt(n[keep] / 2), np.full(int(keep.sum()), 2.)
        mf = deepcopy(provider.mf)
        mf.mo_coeff, mf.mo_occ = mo, occ
        response = mf.gen_response(mo, occ, hermi=1)
        s1 = c.conj().T @ ds @ co
        f1 = cv.conj().T @ np.asarray(mf.Hessian().make_h1(mo, occ)) @ co

        def kernel(dm):
            v = np.asarray(response(dm.reshape(-1, *dm.shape[-2:]))).reshape(dm.shape)
            return cv.conj().T @ v @ co

        def density_change(rotation, rows):
            d = rows @ rotation @ co.conj().T
            return 2 * (d + d.conj().swapaxes(-1, -2)).real

        def hessian(z):
            return delta * z + kernel(density_change(z, cv))

        u_occ = -.5 * s1[..., :nocc, :]
        rhs = -(f1 - s1[..., nocc:, :] * e[:nocc]) - kernel(density_change(u_occ, co))

        # relaxation(u) = sum u alpha + conj(u) beta = (A_DTO - (B - B_t))/2 from the virtual rows
        excited = gauge[1]
        core, pair = (excited.coefficients[0][:, excited.occupations[0] == f] for f in (2, 1))
        p, y = co.conj().T @ overlap @ pair, cv.conj().T @ overlap @ pair
        m = np.real(np.sum(p.conj() * p, axis=0))
        n_hh, n_pp = (np.linalg.det(co.conj().T @ overlap @ np.column_stack([core, pair[:, j]])) ** 2 for j in (0, 1))
        alpha_e = np.outer(y[:, 1].conj(), p[:, 0]) / (m[0] - m[1])
        beta_e = np.outer(y[:, 0], p[:, 1].conj()) / (m[0] - m[1])
        alpha = np.sqrt(.5) * (n_pp * alpha_e - n_hh * beta_e.conj())
        beta = np.sqrt(.5) * (n_pp * beta_e - n_hh * alpha_e.conj()) - .5 * w[nocc:].conj()
        # complex FON-like minima have a degenerate occupied/virtual pair (delta = 0) that only the
        # kernel stabilises, so the diagonal preconditioner is floored
        preconditioner = np.maximum(delta, .05)
        z_re = self._solve(hessian, preconditioner, alpha.conj() + beta)
        z_im = self._solve(hessian, preconditioner, 1j * (alpha.conj() - beta))
        relaxed = (np.real(np.einsum("ai,bxai->bx", z_re.conj(), rhs))
                   + 1j * np.real(np.einsum("ai,bxai->bx", z_im.conj(), rhs)))
        u_tv = u_t[..., nocc:, :]
        relaxed -= np.einsum("bxai,ai->bx", u_tv, alpha) + np.einsum("bxai,ai->bx", u_tv.conj(), beta)
        relaxed -= .5 * np.conj(np.einsum("axij,ij->ax", u_occ - u_t[..., :nocc, :], w[:nocc]))
        raw = .5 * (a_t - np.conj(b_t)) + relaxed
        norm = float(np.linalg.norm(raw.real))
        if not np.isfinite(norm) or norm <= 64 * np.finfo(float).eps:
            raise ZeroNACError("The Z-vector NAC has no real part; its direction is undefined")
        return NACResult(raw, raw.real / norm, norm)

    def _solve(self, hessian, preconditioner, target):
        """Diagonally preconditioned CG for the real-linear symmetric orbital hessian."""
        def inner(a, b):
            return float(np.real(np.sum(a.conj() * b)))
        z = target / preconditioner
        residual = target - hessian(z)
        direction = residual / preconditioner
        rz = inner(residual, direction)
        for _ in range(self.max_iterations):
            if np.max(np.abs(residual)) < self.tolerance:
                return z
            h = hessian(direction)
            curvature = inner(direction, h)
            if curvature <= 0:
                raise ValueError("The ground orbital Hessian is not positive: the ground is not a minimum")
            step = rz / curvature
            z, residual = z + step * direction, residual - step * h
            rz, previous = inner(residual, residual / preconditioner), rz
            direction = residual / preconditioner + rz / previous * direction
        if np.max(np.abs(residual)) < self.tolerance:
            return z
        raise RuntimeError("Z-vector conjugate gradient did not converge")

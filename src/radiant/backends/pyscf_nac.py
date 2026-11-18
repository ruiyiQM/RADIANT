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

An unrestricted pair is two single determinants, each a stationary point of the same energy, so
both are relaxed and there is no DTO rotation:
  d = (<0|d1> - conj<1|d0>)/2,  <Y|dX> = sum_s [Tr(<mu|d nu> t^s_YX) + sum_pi U^s_pi W^s_pi],
with u_oo = -S1_oo/2 and the virtual rows from the per-spin CPKS of state X. Each state needs two
Z-vectors (real and imaginary part); CPU MINRES or device Krylov handles an excited state's
indefinite Hessian. For a non-orthogonal pair the Loewdin-consistent coupling has the same direction and a
norm larger by 1/sqrt(1 - |<0|1>|^2).
"""

from copy import deepcopy

import numpy as np
from pyscf import dft
from scipy.sparse.linalg import LinearOperator, minres

from ..core.arrays import array_module, null_space, to_host, max_abs
from ..core.coupling import NACResult, ZeroNACError
from ..core.overlap import _determinants
from ..core.state import OrbitalState
from ..core.transport import lowdin


def _adjugate(m):
    xp = array_module(m)
    if not m.size:
        return m.copy()
    u, s, vh = xp.linalg.svd(m)
    cofactors = xp.stack([xp.prod(xp.delete(s, j)) for j in range(len(s))])
    return xp.linalg.det(u) * xp.linalg.det(vh) * (vh.conj().T * cofactors) @ u.conj().T


def _spin_transition_densities(left, right, overlap):
    """AO matrices (t_alpha, t_beta) with <left| sum_i o_s(i) |right> = Tr(O t_s)."""
    xp = array_module(overlap)
    ta = tb = 0
    for al, bl, wl in _determinants(left):
        for ar, br, wr in _determinants(right):
            ma, mb = al.conj().T @ overlap @ ar, bl.conj().T @ overlap @ br
            ta = ta + wl * wr * xp.linalg.det(mb) * ar @ _adjugate(ma) @ al.conj().T
            tb = tb + wl * wr * xp.linalg.det(ma) * br @ _adjugate(mb) @ bl.conj().T
    return ta, tb


def _transition_density(left, right, overlap):
    """AO matrix t with <left| sum_i o(i) |right> = Tr(O t), spin summed."""
    return sum(_spin_transition_densities(left, right, overlap))


def _canonical(occupied, overlap, fock):
    """[occupied | complement] diagonalising fock in each block; the occupied rotation has
    determinant 1, so the determinant and its phase are unchanged."""
    xp = array_module(overlap)
    inverse_half, half = lowdin(overlap)
    x = half @ occupied
    c = inverse_half @ xp.column_stack([x, null_space(x.conj().T)])
    nocc = occupied.shape[1]
    e = xp.zeros(c.shape[1])
    for block in (slice(0, nocc), slice(nocc, None)):
        if not c[:, block].shape[1]:
            continue
        e[block], v = xp.linalg.eigh(c[:, block].conj().T @ fock @ c[:, block])
        v[:, 0] *= xp.conj(xp.linalg.det(v))
        c[:, block] = c[:, block] @ v
    return c, e


def _derivative_integrals(mol, provider=None):
    """<mu| d nu/dR_A> and dS/dR_A, both (natm, 3, nao, nao)."""
    if getattr(provider, "device_resident", False):
        return provider.derivative_integrals()
    ip = mol.intor("int1e_ipovlp", comp=3)                       # <grad mu|nu>
    basis = np.zeros((mol.natm, 3, mol.nao, mol.nao))
    for atom, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        basis[atom, :, :, p0:p1] = -ip.transpose(0, 2, 1)[:, :, p0:p1]
    return basis, basis + basis.transpose(0, 1, 3, 2)


def _transport_generators(overlap, ds):
    """K with dC = K C at fixed S**(1/2) C: K = -S**(-1/2) Y, S**(1/2) Y + Y S**(1/2) = dS."""
    xp = array_module(overlap)
    w, v = xp.linalg.eigh(overlap)
    root = xp.sqrt(w)
    y = xp.einsum("pi,axpq,qj->axij", v, ds, v) / (root[:, None] + root[None, :])
    return -xp.einsum("pi,axij,qj->axpq", v / root, y, v)


def _uses_real_density(provider):
    if provider.density_convention == "paper_real":
        return True
    if not isinstance(provider.mf, dft.rks.KohnShamDFT):
        return False
    return not any(provider.mf._numint.rsh_and_hybrid_coeff(provider.mf.xc, spin=provider.mf.mol.spin))


class ZVectorNAC:
    """Drop-in for PaperFD: compute(execution) -> NACResult; MASH uses the complex raw."""

    def __init__(self, *, tolerance=1e-8, max_iterations=200):
        if not np.isfinite(tolerance) or tolerance <= 0 or int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("Z-vector NAC needs a positive tolerance and iteration limit")
        self.tolerance, self.max_iterations = tolerance, int(max_iterations)

    def compute(self, execution):
        provider = execution.providers[0]
        if len(execution.states) != 2 or not hasattr(provider, "unrestricted"):
            raise ValueError("Z-vector NAC needs a PySCF two-state execution")
        if not _uses_real_density(provider):
            raise ValueError("Z-vector NAC needs an energy of Re P only: a pure functional or paper_real")
        if provider.unrestricted:
            return self._result(self._unrestricted(execution))
        ground, gauge_excited = execution.states[0], execution.gauge[1]
        mol, overlap = provider.mf.mol, provider.overlap
        xp = array_module(overlap)
        nocc = ground.coefficients[0].shape[1]
        density = ground.density().real
        inverse_half, half = lowdin(overlap)

        # the ground determinant and its phase (the execution gauge) are unchanged
        c, e = _canonical(ground.coefficients[0], overlap, provider.fock(density))
        co, cv = c[:, :nocc], c[:, nocc:]
        delta = e[nocc:, None] - e[None, :nocc]

        # transported parts
        gauge = (OrbitalState((co,), ground.occupations), gauge_excited)
        t01 = _transition_density(gauge[0], gauge[1], overlap)
        t10 = _transition_density(gauge[1], gauge[0], overlap)
        basis, ds = _derivative_integrals(mol, provider)
        k = _transport_generators(overlap, ds)
        a_t = xp.einsum("axij,ji->ax", basis + overlap @ k, t01)
        w = (co.conj().T @ overlap @ t10 @ overlap @ c).T                      # (nmo, nocc)
        u_t = c.conj().T @ overlap @ k @ co
        b_t = xp.einsum("axij,ji->ax", basis, t10) + xp.einsum("axpi,pi->ax", u_t, w)   # <1|d0>, transported

        # response at Re P, through its natural orbitals scaled by sqrt(n/2) with occupation 2
        n, vectors = xp.linalg.eigh(half @ density @ half)
        keep = n > 1e-10
        mo, occ = (inverse_half @ vectors[:, keep]) * xp.sqrt(n[keep] / 2), xp.full(int(keep.sum()), 2.)
        mf = provider.response_model(mo, occ) if getattr(provider, "device_resident", False) else deepcopy(provider.mf)
        if getattr(provider, "device_resident", False):
            mo, occ = mf.mo_coeff, mf.mo_occ
        else:
            mf.mo_coeff, mf.mo_occ = mo, occ
        response = mf.gen_response(mo, occ, hermi=1)
        s1 = c.conj().T @ ds @ co
        f1 = (provider.response_h1(mf, mo, occ, left=cv, right=co)
              if getattr(provider, "device_resident", False)
              else cv.conj().T @ xp.asarray(mf.Hessian().make_h1(mo, occ)) @ co)

        def kernel(dm):
            v = xp.asarray(response(dm.reshape(-1, *dm.shape[-2:]))).reshape(dm.shape)
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
        m = xp.real(xp.sum(p.conj() * p, axis=0))
        n_hh, n_pp = (xp.linalg.det(co.conj().T @ overlap @ xp.column_stack([core, pair[:, j]])) ** 2 for j in (0, 1))
        alpha_e = xp.outer(y[:, 1].conj(), p[:, 0]) / (m[0] - m[1])
        beta_e = xp.outer(y[:, 0], p[:, 1].conj()) / (m[0] - m[1])
        alpha = xp.sqrt(.5) * (n_pp * alpha_e - n_hh * beta_e.conj())
        beta = xp.sqrt(.5) * (n_pp * beta_e - n_hh * alpha_e.conj()) - .5 * w[nocc:].conj()
        # complex FON-like minima have a degenerate occupied/virtual pair (delta = 0) that only the
        # kernel stabilises, so the diagonal preconditioner is floored
        preconditioner = xp.maximum(delta, .05)
        z_re, z_im = self._solve(hessian, preconditioner,
                                xp.stack((alpha.conj() + beta, 1j * (alpha.conj() - beta))))
        relaxed = (xp.real(xp.einsum("ai,bxai->bx", z_re.conj(), rhs))
                   + 1j * xp.real(xp.einsum("ai,bxai->bx", z_im.conj(), rhs)))
        u_tv = u_t[..., nocc:, :]
        relaxed -= xp.einsum("bxai,ai->bx", u_tv, alpha) + xp.einsum("bxai,ai->bx", u_tv.conj(), beta)
        relaxed -= .5 * xp.conj(xp.einsum("axij,ij->ax", u_occ - u_t[..., :nocc, :], w[:nocc]))
        return self._result(.5 * (a_t - xp.conj(b_t)) + relaxed)

    @staticmethod
    def _result(raw):
        """Keep raw phases; report the real component maximized over state phases."""
        xp = array_module(raw)
        if not xp.isfinite(raw).all():
            raise ZeroNACError("The Z-vector NAC is non-finite")
        phase = .5 * xp.arctan2(2 * xp.sum(raw.real * raw.imag),
                               xp.sum(raw.real**2 - raw.imag**2))
        direction = (xp.exp(-1j * phase) * raw).real
        norm = float(xp.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 64 * np.finfo(float).eps:
            raise ZeroNACError("The Z-vector NAC is zero; its direction is undefined")
        return NACResult(to_host(raw), to_host(direction / norm), norm)

    def _unrestricted(self, execution):
        occupied = [tuple(c[:, f == 1] for c, f in zip(s.coefficients, s.occupations)) for s in execution.gauge]
        g10 = self._relaxed_coupling(execution.providers[0], occupied[0], occupied[1])   # <1|d0>
        g01 = self._relaxed_coupling(execution.providers[1], occupied[1], occupied[0])   # <0|d1>
        return .5 * (g01 - g10.conj())

    def _relaxed_coupling(self, provider, right, left):
        """<left| d right>, natm x 3, with the right determinant relaxed by per-spin CPKS."""
        mol, overlap = provider.mf.mol, provider.overlap
        xp = array_module(overlap)
        density = xp.stack([c @ c.conj().T for c in right])
        fock = provider.fock(density)
        nocc = [c.shape[1] for c in right]
        cs, es = zip(*(_canonical(right[s], overlap, fock[s]) for s in range(2)))
        co = [c[:, :n] for c, n in zip(cs, nocc)]
        cv = [c[:, n:] for c, n in zip(cs, nocc)]
        t = _spin_transition_densities(OrbitalState(left, tuple(np.ones(c.shape[1]) for c in left)),
                                       OrbitalState(co, tuple(np.ones(n) for n in nocc)), overlap)
        basis, ds = _derivative_integrals(mol, provider)
        w = [(co[s].conj().T @ overlap @ t[s] @ overlap @ cs[s]).T for s in range(2)]   # (nmo, nocc)

        # response at Re P^s, through its natural orbitals scaled by sqrt(n) with occupation 1
        inverse_half, half = lowdin(overlap)
        mo, occ = [], []
        for s in range(2):
            n, vectors = xp.linalg.eigh(half @ density[s].real @ half)
            keep = n > 1e-10
            mo.append((inverse_half @ vectors[:, keep]) * xp.sqrt(n[keep]))
            occ.append(xp.ones(int(keep.sum())))
        mf = provider.response_model(mo, occ) if getattr(provider, "device_resident", False) else deepcopy(provider.mf)
        if getattr(provider, "device_resident", False):
            mo, occ = mf.mo_coeff, mf.mo_occ
        else:
            mf.mo_coeff, mf.mo_occ = mo, occ
        response = mf.gen_response(mo, occ, hermi=1)
        f1 = (provider.response_h1(mf, mo, occ, left=cv, right=co)
              if getattr(provider, "device_resident", False)
              else [cv[s].conj().T @ xp.asarray(h) @ co[s]
                    for s, h in enumerate(mf.Hessian().make_h1(mo, occ))])
        delta = [es[s][nocc[s]:, None] - es[s][None, :nocc[s]] for s in range(2)]
        s1 = [cs[s].conj().T @ ds @ co[s] for s in range(2)]

        def kernel(rotations, rows):
            dms = []
            for s in range(2):
                d = rows[s] @ rotations[s] @ co[s].conj().T
                dms.append((d + d.conj().swapaxes(-1, -2)).real.reshape(-1, *overlap.shape))
            v = xp.asarray(response(xp.stack(dms)))
            return [(cv[s].conj().T @ v[s] @ co[s]).reshape(*rotations[s].shape[:-2], *delta[s].shape)
                    for s in range(2)]

        u_occ = [-.5 * s1[s][..., :nocc[s], :] for s in range(2)]
        k_occ = kernel(u_occ, co)
        rhs = [-(f1[s] - s1[s][..., nocc[s]:, :] * es[s][:nocc[s]]) - k_occ[s]
               for s in range(2)]
        coupling = sum(xp.einsum("axij,ji->ax", basis, t[s]) + xp.einsum("axij,ij->ax", u_occ[s], w[s][:nocc[s]])
                       for s in range(2))

        # the relaxation sum_x U_x W is real-linear in U: one Z-vector each for its real and imaginary part
        sizes = [d.size for d in delta]

        def pack(parts):
            z = xp.concatenate([p.reshape(*p.shape[:-2], -1) for p in parts], axis=-1)
            return xp.concatenate([z.real, z.imag], axis=-1)

        def unpack(x):
            z = x[..., :sum(sizes)] + 1j * x[..., sum(sizes):]
            return [z[..., :sizes[0]].reshape(*x.shape[:-1], *delta[0].shape),
                    z[..., sizes[0]:].reshape(*x.shape[:-1], *delta[1].shape)]

        def hessian(x):
            u = unpack(x)
            k = kernel(u, cv)
            return pack([delta[s] * u[s] + k[s] for s in range(2)])

        n = 2 * sum(sizes)
        # complex FON-like minima have a degenerate occupied/virtual pair (delta = 0), so the
        # diagonal preconditioner is floored as in the restricted solver
        floor = 1 / xp.maximum(xp.abs(pack(delta)[:n // 2]), .05)
        if xp is np:
            preconditioner = LinearOperator((n, n), matvec=lambda x: np.concatenate([floor, floor]) * x, dtype=float)
            operator = LinearOperator((n, n), matvec=hessian, dtype=float)
        targets = [pack([part * w[s][nocc[s]:].conj() for s in range(2)]) for part in (1, 1j)]
        if xp is not np:
            from .gpu_response import solve_indefinite
            solutions, status = solve_indefinite(hessian, xp.stack(targets), xp.concatenate([floor, floor]),
                                                self.tolerance, self.max_iterations)
            if status != 0:
                raise RuntimeError("Z-vector indefinite solve did not converge")
        for index, (part, b) in enumerate(zip((1, 1j), targets)):
            if not xp.any(b):
                continue
            # MINRES measures the residual in the preconditioner norm, i.e. roughly the error of z
            if xp is np:
                z, status = minres(operator, b, M=preconditioner, rtol=self.tolerance, maxiter=self.max_iterations)
            else:
                z = solutions[index]
            if status != 0:
                raise RuntimeError("Z-vector indefinite solve did not converge")
            z = unpack(z)
            coupling = coupling + part * sum(xp.real(xp.einsum("ai,bxai->bx", z[s].conj(), rhs[s])) for s in range(2))
        return coupling

    def _solve(self, hessian, preconditioner, target):
        """Batch independent CG recurrences through the same real-linear Hessian."""
        xp = array_module(target)
        def inner(a, b):
            return xp.real(xp.sum(a.conj() * b, axis=(-2, -1), keepdims=True))
        z = target / preconditioner
        residual = target - hessian(z)
        direction = residual / preconditioner
        rz = inner(residual, direction)
        for _ in range(self.max_iterations):
            if max_abs(residual) < self.tolerance:
                return z
            h = hessian(direction)
            curvature = inner(direction, h)
            active = xp.max(abs(residual), axis=(-2, -1), keepdims=True) >= self.tolerance
            if xp.any(active & (curvature <= 0)):
                raise ValueError("The ground orbital Hessian is not positive: the ground is not a minimum")
            step = xp.where(active, rz, 0.) / xp.where(active, curvature, 1.)
            z, residual = z + step * direction, residual - step * h
            rz, previous = inner(residual, residual / preconditioner), rz
            direction = residual / preconditioner + rz / xp.where(previous != 0, previous, 1.) * direction
        if max_abs(residual) < self.tolerance:
            return z
        raise RuntimeError("Z-vector conjugate gradient did not converge")

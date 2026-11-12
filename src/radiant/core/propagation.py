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
"""Self-consistent orbital CN and occupied-subspace PT-CN and PT-IM.

Orbital CN follows TDMASH v0.1.0; PT-CN follows frozen d448adb PT.py.
PT-IM is the implicit midpoint rule of the PT dynamics (references in
_parallel_transport).
An unconverged candidate is never returned as an accepted state.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, eigh, lu_factor, lu_solve, solve_triangular

from .anderson import AndersonAcceleration
from .state import OrbitalState
from .transport import lowdin


class StepFailure(RuntimeError):
    """An electronic step failed; the caller must retain its accepted state."""

    def __init__(self, message, *, residual=np.inf, iterations=0):
        super().__init__(message)
        self.residual, self.iterations = residual, iterations


@dataclass(frozen=True)
class CNOptions:
    variant: str = "cn_accelerated"
    tolerance: float = 1e-9
    max_iterations: int = 80
    # Anderson history of the PT solvers (pt_cn and pt_im).
    history: int = 20

    def __post_init__(self):
        if self.variant not in ("cn", "cn_accelerated", "pt_cn", "pt_im"):
            raise ValueError("Supported propagators: cn, cn_accelerated, pt_cn, pt_im")
        if (not isinstance(self.history, int) or isinstance(self.history, bool)
                or self.history < 1):
            raise ValueError("history must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Tolerance must be finite and positive")
        if (not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool)
                or self.max_iterations < 1):
            raise ValueError("max_iterations must be a positive integer")


@dataclass(frozen=True)
class Convergence:
    """Diagnostics of one accepted electronic step.

    residual and coefficient_change describe the converged nonlinear root; for
    PT-CN the residual is the joint raw AO norm before final orthonormalization.
    orthogonality_error is the max Gram error of the returned orbitals.

    The trapezoidal PT-CN root is not S-orthonormal, so PT-CN returns a
    grouped Cholesky orthonormalization of it. raw_orthogonality_error is the
    max Gram error of the root itself (all groups of a spin jointly), and
    returned_residual is the PT-CN residual re-evaluated on the returned
    orbitals, in the same norm as residual. Plain CN returns its root, so the
    two fields equal orthogonality_error and residual. The PT-IM root keeps
    the Gram up to the solver tolerance; raw_orthogonality_error is that
    defect, removed by a symmetric orthonormalization before returning, and
    returned_residual is not evaluated (NaN).
    """

    iterations: int
    residual: float
    coefficient_change: float
    orthogonality_error: float
    raw_orthogonality_error: float = np.nan
    returned_residual: float = np.nan


PT_VARIANTS = ("pt_cn", "pt_im")


def validate_pt_state(state):
    """Accept restricted 2/1 or per-spin unit occupations; reject virtual columns."""
    allowed = (1, 2) if len(state.coefficients) == 1 else (1,)
    if (not any(c.shape[1] for c in state.coefficients)
            or any(not np.isin(f, allowed).all() for f in state.occupations)):
        raise ValueError("PT propagation requires occupied restricted (2/1) or per-spin unit "
                         "blocks without virtual columns")


def _occupation_groups(state):
    """Return (spin, columns) blocks whose internal unitary mixing is a gauge.

    Each unrestricted spin is one block. A restricted state is split into its
    doubly and singly occupied columns: rotations inside each group leave the
    density unchanged, while rotations between the groups change it and are
    therefore retained as physical dynamics.
    """
    if len(state.coefficients) == 2:
        return tuple((spin, np.arange(f.size)) for spin, f in enumerate(state.occupations))
    occupations = state.occupations[0]
    return tuple((0, np.flatnonzero(occupations == value)) for value in (2, 1)
                 if np.any(occupations == value))


def crank_nicolson(state, overlap, fock, time, dt, *, alpha=1., options=CNOptions()):
    """Return a new state and diagnostics; fock(density, physical_time) is local.

    Both unrestricted Focks see the same joint trial density. Residuals are
    measured in the original AO equation. PT-CN and PT-IM accept occupied
    columns only. Signed dt is allowed for field-free probes; alpha scales
    electronics only.
    """
    if (not np.isfinite([time, dt, alpha]).all() or dt == 0 or alpha <= 0):
        raise ValueError("Require finite time, nonzero dt and positive alpha")
    lowdin(overlap)
    if state.orthogonality_error(overlap) > 1e-8:
        raise ValueError("Initial orbitals are not S-orthonormal")
    if options.variant in PT_VARIANTS:
        validate_pt_state(state)
    nspin = len(state.coefficients)

    def evaluate(trial, t, iterations=0):
        value = np.asarray(fock(trial.density(), t), dtype=complex)
        matrices = value[None] if nspin == 1 else value
        if (matrices.shape != (nspin, *np.shape(overlap)) or not np.isfinite(matrices).all()
                or not np.allclose(matrices, matrices.conj().transpose(0, 2, 1), atol=1e-12, rtol=0)):
            raise StepFailure("Fock provider returned invalid/non-Hermitian matrices", iterations=iterations)
        return matrices * alpha

    if options.variant in PT_VARIANTS:
        return _parallel_transport(state, overlap, evaluate, time, dt, options)

    def solve(average):
        return OrbitalState(tuple(np.linalg.solve(overlap + .5j * dt * h,
                                (overlap - .5j * dt * h) @ c)
                                  for h, c in zip(average, state.coefficients)), state.occupations)

    start = evaluate(state, time)
    guess = start.copy()
    previous = state if options.variant == "cn_accelerated" else solve(start)
    if options.variant == "cn":
        guess = evaluate(previous, time + dt)
    accelerator = AndersonAcceleration(history_size=6, regularization=1e-10)
    residual = np.inf
    try:
        for iteration in range(1, options.max_iterations + 1):
            candidate = solve((start + guess) / 2)
            final = evaluate(candidate, time + dt)
            average = (start + final) / 2
            residual = max(float(np.max(np.abs(
                (overlap + .5j * dt * h) @ new - (overlap - .5j * dt * h) @ old), initial=0))
                for h, new, old in zip(average, candidate.coefficients, state.coefficients))
            change = max(float(np.max(np.abs(new - old), initial=0))
                         for new, old in zip(candidate.coefficients, previous.coefficients))
            orthogonality = candidate.orthogonality_error(overlap)
            if (residual < options.tolerance and change < options.tolerance
                    and (iteration > 1 or options.variant == "cn")):
                if orthogonality > 1e-8:
                    raise StepFailure("CN orthogonality failed", residual=residual, iterations=iteration)
                return candidate, Convergence(iteration, residual, change, orthogonality,
                                              orthogonality, residual)
            guess = (accelerator.compute(guess, final)
                     if options.variant == "cn_accelerated" else final)
            previous = candidate
    except np.linalg.LinAlgError as exc:
        raise StepFailure("CN linear solve failed", residual=residual) from exc
    raise StepFailure("CN nonlinear iteration did not converge", residual=residual,
                      iterations=options.max_iterations)


def _pt_preconditioner(overlap, hamiltonian, initial, half_dt):
    """Frozen grouped occupied-energy preconditioner from d448adb PT.py."""
    if initial.shape[1] == 0:
        return lambda residual: residual.copy()
    values, vectors = eigh(initial.conj().T @ overlap @ initial)
    reference = initial @ ((vectors * values**-.5) @ vectors.conj().T)
    occupied_ham = reference.conj().T @ hamiltonian @ reference
    occupied_ham = .5 * (occupied_ham + occupied_ham.conj().T)
    energies, rotation = eigh(occupied_ham)
    groups = [(0, len(energies))]
    if len(energies) > 1 and np.max(np.diff(energies)) > 1e-12 * max(1, np.max(np.abs(energies))):
        split = np.argmax(np.diff(energies)) + 1
        groups = [(0, split), (split, len(energies))]
    while len(groups) < 4:
        spreads = [abs(half_dt) * (energies[end - 1] - energies[start]) for start, end in groups]
        index = int(np.argmax(spreads))
        if spreads[index] <= 1:
            break
        start, end = groups[index]
        split = start + np.argmax(np.diff(energies[start:end])) + 1
        groups[index:index + 1] = [(start, split), (split, end)]
    factors = [lu_factor(overlap + 1j * half_dt *
                        (hamiltonian - np.mean(energies[start:end]) * overlap))
               for start, end in groups]

    def precondition(residual):
        normal = reference.conj().T @ overlap @ residual
        transformed = rotation.conj().T @ normal @ rotation
        # The occupied Jacobian couples a block to its adjoint.
        hermitian = (transformed + transformed.conj().T) / (
            1 + 1j * half_dt * (energies[:, None] - energies[None, :]))
        occupied = normal + 1j * half_dt * (rotation @ hermitian @ rotation.conj().T) @ occupied_ham
        tangent_rhs = overlap @ (residual - reference @ normal) @ rotation
        tangent = np.empty_like(tangent_rhs)
        for (start, end), factor in zip(groups, factors):
            tangent[:, start:end] = lu_solve(factor, tangent_rhs[:, start:end])
        tangent -= reference @ (reference.conj().T @ overlap @ tangent)
        return reference @ occupied + tangent @ rotation.conj().T
    return precondition


def _midpoint_phi(overlap, metric_factor, hamiltonian, mids):
    """Grouped Gram-inverse PT field phi = S^-1 F M - M G^-1 B of one spin.

    G = M^H S M is the full Gram of all groups of the spin, while
    B = blockdiag_g(M_g^H F M_g) keeps only the intra-group gauge. B and G are
    Hermitian, so M^H S phi is Hermitian and the root keeps the full Gram,
    including the orthogonality between groups, without any projection.
    """
    joint = np.concatenate(mids, axis=1)
    if not joint.shape[1]:
        return [m.copy() for m in mids]
    hm = hamiltonian @ joint
    energies = joint.conj().T @ hm
    bounds = np.cumsum([0] + [m.shape[1] for m in mids])
    gauge = np.zeros_like(energies)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        gauge[lo:hi, lo:hi] = energies[lo:hi, lo:hi]
    phi = (cho_solve(metric_factor, hm)
           - joint @ np.linalg.solve(joint.conj().T @ overlap @ joint, gauge))
    return [phi[:, lo:hi] for lo, hi in zip(bounds[:-1], bounds[1:])]


def _parallel_transport(state, overlap, evaluate, time, dt, options):
    """Joint real-block Anderson solve; evaluate supplies alpha-scaled Focks.

    Blocks are occupation groups (see _occupation_groups). Each group removes
    only its own internal rotation, A_g = S^-1 F C_g - C_g (C_g^H F C_g), so
    restricted 2/1 groups share one Fock and keep their mutual coupling.

    pt_cn solves the trapezoidal rule (Jia, An, Wang and Lin,
    arXiv:1805.10575, Eq. 7) and orthonormalizes the root. pt_im solves the
    implicit midpoint rule C1 - C0 + i dt phi(M) = 0 with M = (C0 + C1)/2,
    the Fock of D_M = sum_g f_g M_g M_g^H at time + dt/2 and the grouped field
    of _midpoint_phi. Its root keeps the Gram up to the solver tolerance and
    only that defect is removed before returning.

    The implicit midpoint rule for PT dynamics was proposed by An and Lin,
    Multiscale Model. Simul. 18, 612 (2020), arXiv:1804.02095. With one group
    per spin, pt_im is the PT-IM scheme of An, Fang and Lin, J. Comput. Phys.
    451, 110850 (2022), arXiv:2105.14755, Eqs. 4.1-4.4 with a constant
    occupation matrix, written in the S metric: the projector carries the
    full Gram inverse and the midpoint density is not Gram-normalized. For
    restricted 2/1 states that paper evolves an occupation matrix and removes
    every rotation within the occupied span; here the occupations stay
    diag(f_g) and only intra-group rotations are removed. Both forms give the
    same continuous density dynamics, but this grouped form keeps the
    occupation spectrum exact.
    """
    midpoint = options.variant == "pt_im"
    label = "PT-IM" if midpoint else "PT-CN"
    groups = _occupation_groups(state)
    spins = [spin for spin, _ in groups]
    members = [[i for i, s in enumerate(spins) if s == spin]
               for spin in range(len(state.coefficients))]
    initial = [state.coefficients[spin][:, columns] for spin, columns in groups]
    blocks = [c.copy() for c in initial]
    sizes = np.cumsum([0] + [c.size for c in initial])
    residual, change, iteration = np.inf, 0., 0

    def assemble(parts):
        coefficients = [np.empty_like(c) for c in state.coefficients]
        for (spin, columns), part in zip(groups, parts):
            coefficients[spin][:, columns] = part
        return OrbitalState(tuple(coefficients), state.occupations)

    try:
        metric_factor = cho_factor(overlap)

        def action(c, h):
            hc = h @ c
            return cho_solve(metric_factor, hc) - c @ (c.conj().T @ hc)

        start = evaluate(state, time)
        half_dt = .5 * dt
        preconditioners = [_pt_preconditioner(overlap, start[s], c, half_dt)
                           for c, s in zip(initial, spins)]
        if midpoint:
            # The midpoint field has no occupied-span Jacobian of the PT-CN
            # form: leave that component unpreconditioned, treat the rest natively.
            def split(native, c0):
                def precondition(r):
                    q = c0 @ (c0.conj().T @ overlap @ r)
                    p = native(r - q)
                    return q + p - c0 @ (c0.conj().T @ overlap @ p)
                return precondition
            preconditioners = [split(native, c0) for native, c0 in zip(preconditioners, initial)]

            def errors_of(parts, iteration):
                mids = [.5 * (c0 + c) for c0, c in zip(initial, parts)]
                middle = evaluate(assemble(mids), time + half_dt, iteration)
                errors = [None] * len(parts)
                for spin, group in enumerate(members):
                    if group:
                        phis = _midpoint_phi(overlap, metric_factor, middle[spin], [mids[i] for i in group])
                        for i, phi in zip(group, phis):
                            errors[i] = initial[i] - parts[i] - 1j * dt * phi
                return errors
        else:
            rhs = [c - 1j * half_dt * action(c, start[s]) for c, s in zip(initial, spins)]

            def errors_of(parts, iteration):
                final = evaluate(assemble(parts), time + dt, iteration)
                return [r - c - 1j * half_dt * action(c, final[s])
                        for r, c, s in zip(rhs, parts, spins)]

        accelerator = AndersonAcceleration(history_size=options.history, regularization=1e-12)
        for iteration in range(1, options.max_iterations + 1):
            errors = errors_of(blocks, iteration)
            residual = float(np.linalg.norm([np.linalg.norm(r) for r in errors]))
            if not np.isfinite(residual):
                raise StepFailure(f"{label} residual is not finite", residual=residual, iterations=iteration)
            if residual <= options.tolerance:
                break
            current = np.concatenate([c.ravel() for c in blocks])
            candidate = np.concatenate([(c + precondition(r)).ravel()
                                        for c, r, precondition in zip(blocks, errors, preconditioners)])
            # Scale the Tikhonov term with the update for every block count: a
            # fixed value swamps the shrinking history near convergence, which
            # turns Anderson off and stalls large alpha*dt steps.
            accelerator.regularization = 1e-12 * np.linalg.norm(candidate - current)**2
            # Real mixing retains the real-linear dependence on C and C*.
            mixed = accelerator.compute(np.stack((current.real, current.imag)),
                                        np.stack((candidate.real, candidate.imag)))
            vector = mixed[0] + 1j * mixed[1]
            if not np.isfinite(vector).all():
                raise StepFailure(f"{label} trial orbitals are not finite", residual=residual, iterations=iteration)
            change = float(np.max(np.abs(vector - current), initial=0))
            blocks = [vector[sizes[i]:sizes[i+1]].reshape(c.shape) for i, c in enumerate(initial)]
        else:
            raise StepFailure(f"{label} nonlinear iteration did not converge", residual=residual,
                              iterations=options.max_iterations)

        if midpoint:
            # The exact root keeps the full Gram, so the converged one deviates
            # only at the solver tolerance. A symmetric orthonormalization per
            # spin removes that tolerance-sized defect without favouring a group;
            # returned_residual is not evaluated to avoid an extra Fock build.
            raw_orthogonality = 0.
            for group in members:
                joint = np.concatenate([blocks[i] for i in group], axis=1)
                if joint.shape[1]:
                    gram = joint.conj().T @ overlap @ joint
                    raw_orthogonality = max(raw_orthogonality,
                                            float(np.max(np.abs(gram - np.eye(len(gram))))))
                    values, vectors = eigh(gram)
                    joint = joint @ ((vectors * values**-.5) @ vectors.conj().T)
                bounds = np.cumsum([0] + [blocks[i].shape[1] for i in group])
                for k, i in enumerate(group):
                    blocks[i] = joint[:, bounds[k]:bounds[k + 1]]
            accepted = assemble(blocks)
            orthogonality = accepted.orthogonality_error(overlap)
            if raw_orthogonality > 1e-6 or orthogonality > 1e-8:
                raise StepFailure("PT-IM orthogonality failed", residual=residual, iterations=iteration)
            return accepted, Convergence(iteration, residual, change, orthogonality,
                                         raw_orthogonality, np.nan)

        # Orthonormalize each spin jointly in group order (doubly occupied
        # first), so the leading group's span is kept and groups end orthogonal.
        raw_orthogonality = 0.
        for spin, group in enumerate(members):
            joint = np.concatenate([blocks[i] for i in group], axis=1)
            if joint.shape[1]:
                gram = joint.conj().T @ overlap @ joint
                raw_orthogonality = max(raw_orthogonality,
                                        float(np.max(np.abs(gram - np.eye(len(gram))))))
                root = cholesky(gram, lower=True)
                joint = solve_triangular(root, joint.conj().T, lower=True).conj().T
            bounds = np.cumsum([0] + [blocks[i].shape[1] for i in group])
            for k, i in enumerate(group):
                blocks[i] = joint[:, bounds[k]:bounds[k + 1]]
        accepted = assemble(blocks)
        returned = evaluate(accepted, time + dt, iteration)
        returned_residual = float(np.linalg.norm([
            np.linalg.norm(r - c - 1j * half_dt * action(c, returned[s]))
            for r, c, s in zip(rhs, blocks, spins)]))
        orthogonality = accepted.orthogonality_error(overlap)
        if orthogonality > 1e-8:
            raise StepFailure("PT-CN orthogonality failed", residual=residual, iterations=iteration)
        return accepted, Convergence(iteration, residual, change, orthogonality,
                                     raw_orthogonality, returned_residual)
    except np.linalg.LinAlgError as exc:
        raise StepFailure(f"{label} linear solve failed", residual=residual, iterations=iteration) from exc

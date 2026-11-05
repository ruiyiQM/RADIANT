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
"""Self-consistent trapezoidal CN with plain or Anderson Fock iteration.

The two iterations follow TDMASH v0.1.0 TDRHF.py. Unlike the legacy warning
path, an unconverged candidate is never returned as an accepted state.
"""

from dataclasses import dataclass

import numpy as np

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

    def __post_init__(self):
        if self.variant not in ("cn", "cn_accelerated"):
            raise ValueError("Supported propagators: cn, cn_accelerated")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Tolerance must be finite and positive")
        if (not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool)
                or self.max_iterations < 1):
            raise ValueError("max_iterations must be a positive integer")


@dataclass(frozen=True)
class Convergence:
    iterations: int
    residual: float
    coefficient_change: float
    orthogonality_error: float


def crank_nicolson(state, overlap, fock, time, dt, *, alpha=1., options=CNOptions()):
    """Return a new state and diagnostics; fock(density, physical_time) is local.

    Both unrestricted Focks see the same joint trial density. Residuals are
    measured in the original AO equation, over all occupied/virtual columns.
    Signed dt is allowed for field-free probes; alpha scales electronics only.
    """
    if (not np.isfinite([time, dt, alpha]).all() or dt == 0 or alpha <= 0):
        raise ValueError("Require finite time, nonzero dt and positive alpha")
    lowdin(overlap)
    if state.orthogonality_error(overlap) > 1e-8:
        raise ValueError("Initial orbitals are not S-orthonormal")
    nspin = len(state.coefficients)

    def evaluate(trial, t):
        value = np.asarray(fock(trial.density(), t), dtype=complex)
        matrices = value[None] if nspin == 1 else value
        if (matrices.shape != (nspin, *np.shape(overlap)) or not np.isfinite(matrices).all()
                or not np.allclose(matrices, matrices.conj().transpose(0, 2, 1), atol=1e-12, rtol=0)):
            raise StepFailure("Fock provider returned invalid/non-Hermitian matrices")
        return matrices * alpha

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
                return candidate, Convergence(iteration, residual, change, orthogonality)
            guess = (accelerator.compute(guess, final)
                     if options.variant == "cn_accelerated" else final)
            previous = candidate
    except np.linalg.LinAlgError as exc:
        raise StepFailure("CN linear solve failed", residual=residual) from exc
    raise StepFailure("CN nonlinear iteration did not converge", residual=residual,
                      iterations=options.max_iterations)

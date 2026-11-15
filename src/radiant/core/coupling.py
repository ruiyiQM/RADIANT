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

"""The paper signed, common-central-gauge FD probe (steps=1 only)."""

from dataclasses import dataclass

import numpy as np

from .overlap import overlap_coupling
from .propagation import CNOptions


class ZeroNACError(ValueError):
    """The reproduction NAC has no real momentum-rescaling direction."""


@dataclass(frozen=True)
class NACResult:
    raw: np.ndarray
    real_direction: np.ndarray
    real_norm: float


class PaperFD:
    """Independent probe alpha/dt/tolerance; branch times never reach the driver.

    The normalized real axis reproduces the paper approximation, not a
    calibrated general complex NAC magnitude. The full complex probe is kept.
    The signed branches (-dt at -displacement, +dt at +displacement) also pick
    up the intrinsic evolution of a non-stationary state pair, a term
    (dt/displacement) <0|d_t 1> that is the same for every coordinate.
    subtract_drift removes it with one undisplaced branch pair (zero for
    stationary states); subtract_drift=False reproduces the paper probe.
    """

    def __init__(self, *, displacement=.01, dt=4., alpha=1., steps=1,
                 options=CNOptions("cn_accelerated", 1e-6, 80), subtract_drift=True):
        if (not np.isfinite([displacement, dt, alpha]).all()
                or min(displacement, dt, alpha) <= 0 or steps != 1):
            raise ValueError("Paper FD requires positive parameters and steps=1")
        if options.variant in ("pt_cn", "pt_im"):
            raise ValueError("Paper FD does not support PT branches")
        self.displacement, self.dt, self.alpha = displacement, dt, alpha
        self.options, self.subtract_drift = options, subtract_drift

    def compute(self, execution):
        central = execution.snapshot()
        positions = execution.positions

        def branch_coupling(offset):
            endpoints = []
            for sign in (-1, 1):
                execution.restore(central)
                execution.advance(positions + sign * offset, sign * self.dt,
                    alphas=(self.alpha, self.alpha), options=self.options,
                    gauge_reference=central, forces=False)
                endpoints.append(execution.snapshot())
            _, _, coupling = overlap_coupling(execution.overlap(*endpoints), 2*self.displacement)
            return coupling[0, 1]

        raw = np.zeros_like(positions, dtype=complex)
        try:
            for atom in range(len(positions)):
                for axis in range(3):
                    offset = np.zeros_like(positions)
                    offset[atom, axis] = self.displacement
                    raw[atom, axis] = branch_coupling(offset)
            if self.subtract_drift:
                raw -= branch_coupling(np.zeros_like(positions))
        finally:
            execution.restore(central)
        norm = float(np.linalg.norm(raw.real))
        if not np.isfinite(norm) or norm <= 64 * np.finfo(float).eps:
            raise ZeroNACError("The paper real FD NAC is zero; its direction is undefined")
        return NACResult(raw, raw.real / norm, norm)

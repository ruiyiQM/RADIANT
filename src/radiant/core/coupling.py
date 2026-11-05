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
    """

    def __init__(self, *, displacement=.01, dt=4., alpha=1., steps=1,
                 options=CNOptions("cn_accelerated", 1e-6, 80)):
        if (not np.isfinite([displacement, dt, alpha]).all()
                or min(displacement, dt, alpha) <= 0 or steps != 1):
            raise ValueError("Paper FD requires positive parameters and steps=1")
        if options.variant == "pt_cn":
            raise ValueError("Paper FD does not support PT-CN branches")
        self.displacement, self.dt, self.alpha = displacement, dt, alpha
        self.options = options

    def compute(self, execution):
        central = execution.snapshot()
        positions = execution.positions
        raw = np.zeros_like(positions, dtype=complex)
        try:
            for atom in range(len(positions)):
                for axis in range(3):
                    endpoints = []
                    for sign in (-1, 1):
                        execution.restore(central)
                        end = positions.copy()
                        end[atom, axis] += sign * self.displacement
                        execution.advance(end, sign * self.dt,
                            alphas=(self.alpha, self.alpha), options=self.options,
                            gauge_reference=central, forces=False)
                        endpoints.append(execution.snapshot())
                    _, _, coupling = overlap_coupling(execution.overlap(*endpoints), 2*self.displacement)
                    raw[atom, axis] = coupling[0, 1]
        finally:
            execution.restore(central)
        norm = float(np.linalg.norm(raw.real))
        if not np.isfinite(norm) or norm <= 64 * np.finfo(float).eps:
            raise ZeroNACError("The paper real FD NAC is zero; its direction is undefined")
        return NACResult(raw, raw.real / norm, norm)

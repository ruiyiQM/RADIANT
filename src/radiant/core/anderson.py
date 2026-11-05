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

"""Anderson mixing from TDMASH v0.1.0, Anderson.py."""

import numpy as np
class AndersonAcceleration:
    def __init__(self, history_size=5, regularization=1e-4):
        self.history_size = history_size
        self.regularization = regularization
        self.reset()

    def reset(self):
        self.x_history = []
        self.f_history = []

    def compute(self, x_in, x_out):
        """
        x_in: The input parameters (e.g., F_guess)
        x_out: The output parameters (e.g., F_calculated)
        Returns: Optimized guess for next iteration.
        """
        # Flatten matrices to vectors for linear algebra
        flat_in = x_in.ravel()
        flat_out = x_out.ravel()
        residual = flat_out - flat_in

        self.x_history.append(flat_in)
        self.f_history.append(residual)

        if len(self.x_history) > self.history_size:
            self.x_history.pop(0)
            self.f_history.pop(0)

        m = len(self.x_history)
        if m < 2:
            return x_out # Not enough history, return direct update

        # Matrix of residuals: F_k = [f_k, f_{k-1}, ...]
        F = np.column_stack(self.f_history)
        # Matrix of differences: dF_i = f_i - f_{i-1}
        dF = np.diff(F, axis=1)
        # Matrix of input differences: dX_i = x_i - x_{i-1}
        dX = np.column_stack(self.x_history)
        dX = np.diff(dX, axis=1)

        # Solve least squares for mixing coefficients
        # We want to minimize || f(x) || approx || f_k + dF * gamma ||
        # solution gamma defines how to mix previous steps

        # Standard Anderson: Minimize || f_k - sum(gamma_j * (f_j - f_{j-1})) ||
        # This is equivalent to finding coefficients to make residual 0.

        latest_res = self.f_history[-1]

        # Solving linear system: dF.T @ dF @ theta = dF.T @ latest_res
        # Add regularization for stability
        ATA = dF.conj().T @ dF
        ATA += self.regularization * np.eye(ATA.shape[0])
        ATb = dF.conj().T @ latest_res

        try:
            theta = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            return x_out # Fallback if singular

        # Compute accelerated point
        x_acc = flat_in - np.dot(dX, theta) + (latest_res - np.dot(dF, theta))

        return x_acc.reshape(x_in.shape)

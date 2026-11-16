"""NumPy/CuPy dispatch without importing a GPU runtime for CPU calculations."""

import numpy as np
from scipy import linalg


def array_module(value):
    if hasattr(value, "__cuda_array_interface__"):
        import cupy
        return cupy
    return np


def to_host(value):
    xp = array_module(value)
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def max_abs(value):
    return float(array_module(value).max(abs(value))) if value.size else 0.


def eigh(value):
    xp = array_module(value)
    return linalg.eigh(value) if xp is np else xp.linalg.eigh(value)


def null_space(value):
    xp = array_module(value)
    if xp is np:
        return linalg.null_space(value)
    if not value.shape[0]:
        return xp.eye(value.shape[1], dtype=value.dtype)
    _, singular, vh = xp.linalg.svd(value, full_matrices=True)
    threshold = float(singular[0]) * max(value.shape) * np.finfo(float).eps
    rank = int(xp.sum(singular > threshold))
    return vh[rank:].conj().T


def cholesky(value, *, lower=False):
    xp = array_module(value)
    if xp is np:
        return linalg.cholesky(value, lower=lower)
    root = xp.linalg.cholesky(value)
    return root if lower else root.conj().T


def cho_factor(value):
    if array_module(value) is np:
        return linalg.cho_factor(value)
    return cholesky(value, lower=True), True


def cho_solve(factor, rhs):
    if not rhs.size:
        return rhs.copy()
    if array_module(rhs) is np:
        return linalg.cho_solve(factor, rhs)
    root, lower = factor
    left = root if lower else root.conj().T
    return solve_triangular(left.conj().T, solve_triangular(left, rhs, lower=True), lower=False)


def lu_factor(value):
    if array_module(value) is np:
        return linalg.lu_factor(value)
    from cupyx.scipy.linalg import lu_factor as gpu_factor
    return gpu_factor(value)


def lu_solve(factor, rhs):
    if array_module(rhs) is np:
        return linalg.lu_solve(factor, rhs)
    from cupyx.scipy.linalg import lu_solve as gpu_solve
    return gpu_solve(factor, rhs)


def solve_triangular(matrix, rhs, *, lower=False):
    if array_module(matrix) is np:
        return linalg.solve_triangular(matrix, rhs, lower=lower)
    from cupyx.scipy.linalg import solve_triangular as gpu_solve
    xp = array_module(matrix)
    dtype = xp.result_type(matrix, rhs)
    return gpu_solve(matrix.astype(dtype, copy=False), rhs.astype(dtype, copy=False), lower=lower)

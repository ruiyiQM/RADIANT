"""Device-only response solves and AO derivatives for resident Z-vector NACs."""

from copy import copy

import numpy as np

from ..core.arrays import null_space
from ..core.transport import lowdin


def response_model(provider, mo, occ):
    native = provider._gpu_mean_field()
    model = copy(native)
    model._numint = copy(native._numint)
    model.grids = native.grids.copy()
    if hasattr(native, "with_df"):
        model.with_df = copy(native.with_df)
    model._opt_gpu = dict(native._opt_gpu or {})
    model._opt_jengine = dict(native._opt_jengine or {})
    if provider.unrestricted:
        xp = provider.array_module
        width = max(c.shape[1] for c in mo)
        coefficients = xp.zeros((2, mo[0].shape[0], width), dtype=mo[0].dtype)
        occupations = xp.zeros((2, width))
        for spin in range(2):
            coefficients[spin, :, :mo[spin].shape[1]] = mo[spin]
            occupations[spin, :len(occ[spin])] = occ[spin]
        mo, occ = coefficients, occupations
    model.mo_coeff, model.mo_occ = mo, occ
    return model


def ao_derivative(provider, model, mo, occ, *, left=None, right=None):
    """Project occupied-column H1 directly; retain AO recovery for diagnostics."""
    if (left is None) != (right is None):
        raise ValueError("Supply both left and right derivative projection blocks")
    xp = provider.array_module
    inverse, half = lowdin(provider.overlap)
    blocks = mo if provider.unrestricted else (mo,)
    occupied = occ if provider.unrestricted else (occ,)
    occupation = 1. if provider.unrestricted else 2.
    bases, scales, fulls = [], [], []
    for block, weights in zip(blocks, occupied):
        block = block[:, weights > 0]
        scale = xp.sqrt(xp.sum(abs(half @ block)**2, axis=0))
        basis = block / scale if block.shape[1] else block
        complement = inverse @ null_space((half @ basis).T)
        basis = xp.column_stack((basis, complement))
        # shortcut: epsilon-density columns expose AO H1; replace when GPU4PySCF offers an AO interface.
        scale = xp.concatenate((scale, xp.full(complement.shape[1], np.sqrt(np.finfo(float).eps))))
        bases.append(basis); scales.append(scale); fulls.append(basis * scale)
    hmodel = copy(model)
    # GPU4PySCF drops the numerical integrator when copying an SCF object.
    hmodel._numint = model._numint
    hmodel.grids = model.grids
    hmodel.mo_coeff = xp.stack(fulls) if provider.unrestricted else fulls[0]
    occupations = [xp.full(f.shape[1], occupation) for f in fulls]
    hmodel.mo_occ = xp.stack(occupations) if provider.unrestricted else occupations[0]
    hessian = hmodel.Hessian()
    hessian.grid_response = False
    result = hessian.make_h1(hmodel.mo_coeff, hmodel.mo_occ)
    blocks = result if provider.unrestricted else (result,)
    outputs = []
    for spin, (basis, scale, block) in enumerate(zip(bases, scales, blocks)):
        if not hasattr(block, '__cuda_array_interface__'):
            raise TypeError("GPU Hessian exported its derivative matrix to the host")
        inverse_basis = basis.T @ provider.overlap
        if left is None:
            unscaled = block / scale[None, None, :, None] / scale[None, None, None, :]
            outputs.append(inverse_basis.T @ unscaled @ inverse_basis)
        else:
            l = left[spin] if provider.unrestricted else left
            r = right[spin] if provider.unrestricted else right
            l = (inverse_basis @ l) / scale[:, None]
            r = (inverse_basis @ r) / scale[:, None]
            outputs.append(l.conj().T @ (block @ r))
    return outputs if provider.unrestricted else outputs[0]


def solve_indefinite(action, target, inverse_diagonal, tolerance, max_iterations):
    """Use GPU4PySCF's device Krylov solver and verify the actual residual."""
    import cupy as xp
    from gpu4pyscf.lib.cupy_helper import krylov
    norm = xp.linalg.norm(target, axis=-1, keepdims=True)
    if not xp.any(norm):
        return xp.zeros_like(target), 0
    def shifted(vectors):
        return inverse_diagonal * action(vectors) - vectors
    z, residual = xp.zeros_like(target), target
    # Deflate roundoff-sized Krylov directions; refine against the original equation.
    for _ in range(3):
        scale = xp.linalg.norm(residual, axis=-1, keepdims=True)
        scale = xp.where(scale > 0, scale, 1.)
        z += scale * krylov(shifted, inverse_diagonal * residual / scale,
                           tol=.1*tolerance, lindep=max(1e-14, (.1*tolerance)**2),
                           max_cycle=max_iterations, verbose=0)
        residual = target-action(z)
        if xp.all(xp.linalg.norm(residual, axis=-1, keepdims=True) <= tolerance*norm):
            return z, 0
    return z, 1

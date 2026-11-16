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


def ao_derivative(provider, model, mo, occ):
    """Recover AO H1 from GPU4PySCF's occupied-column Hessian interface."""
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
    hmodel.mo_coeff = fulls if provider.unrestricted else fulls[0]
    occupations = [xp.full(f.shape[1], occupation) for f in fulls]
    hmodel.mo_occ = occupations if provider.unrestricted else occupations[0]
    hessian = hmodel.Hessian()
    hessian.grid_response = False
    result = hessian.make_h1(hmodel.mo_coeff, hmodel.mo_occ)
    blocks = result if provider.unrestricted else (result,)
    outputs = []
    for basis, scale, block in zip(bases, scales, blocks):
        if not hasattr(block, '__cuda_array_interface__'):
            raise TypeError("GPU Hessian exported its derivative matrix to the host")
        inverse_basis = basis.T @ provider.overlap
        unscaled = block / scale[None, None, :, None] / scale[None, None, None, :]
        outputs.append(inverse_basis.T @ unscaled @ inverse_basis)
    return outputs if provider.unrestricted else outputs[0]


def solve_indefinite(action, target, inverse_diagonal, tolerance, max_iterations):
    """Use GPU4PySCF's device Krylov solver and verify the actual residual."""
    import cupy as xp
    from gpu4pyscf.lib.cupy_helper import krylov
    norm = float(xp.linalg.norm(target))
    if not norm:
        return xp.zeros_like(target), 0
    def shifted(vectors):
        return xp.stack([inverse_diagonal * action(v) - v for v in xp.atleast_2d(vectors)])
    z = norm * krylov(shifted, inverse_diagonal * target / norm,
                      tol=.1*tolerance, lindep=(.1*tolerance)**2,
                      max_cycle=max_iterations, verbose=0)
    status = int(float(xp.linalg.norm(target-action(z))) > tolerance*norm)
    return z, status

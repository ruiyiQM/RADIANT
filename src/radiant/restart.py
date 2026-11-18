# Copyright (c) 2026 Ruiyi Zhou
"""Atomic, data-only restarts for the built-in PySCF/GPU and MASH components.

Pass freshly configured, matching providers to load_checkpoint. Loading never
runs SCF; saved orbitals, quadrature and DTO phases replace their transient state.
"""

from copy import deepcopy
from dataclasses import asdict
import json
import inspect
import os
from pathlib import Path
import tempfile

import numpy as np

from .core.coupling import PaperFD
from .core.execution import LocalExecution, Observation, _force_states
from .core.arrays import to_host
from .core.propagation import CNOptions
from .core.state import NuclearState, OrbitalState
from .driver import HOP_EVENT_DTYPE, Trajectory
from .methods.mash import MASH, coefficients_to_spin


_MO_FIELDS = ("mo_coeff", "mo_occ", "mo_energy")
_GRID_FIELDS = ("coords", "weights", "non0tab", "screen_index", "atm_idx", "quadrature_weights")
_FRAME_FIELDS = ("time", "positions", "momenta", "energies", "active_state", "kinetic_energy",
                 "total_energy", "spin", "overlap")


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def _model(provider):
    from pyscf import dft, scf
    from .backends.pyscf import PySCFProvider
    from .backends.gpu_pyscf import GPU4PySCFProvider
    from .backends.gpu_resident import GPUResidentProvider
    if type(provider) not in (PySCFProvider, GPU4PySCFProvider, GPUResidentProvider):
        raise ValueError("Restart requires a built-in PySCF or GPU4PySCF provider")
    mf, mol = provider.mf, provider.mf.mol
    base = mf.undo_df() if hasattr(mf, "with_df") else mf
    if type(base) not in (scf.hf.RHF, scf.uhf.UHF, dft.rks.RKS, dft.uks.UKS):
        raise ValueError("Restart requires a standard molecular mean-field class")
    if any(k != "get_occ" and (inspect.isroutine(v) or
            (callable(v) and callable(getattr(type(mf), k, None)))) for k, v in vars(mf).items()):
        raise ValueError("Restart cannot preserve custom mean-field methods")
    name = "GPU4PySCFProvider" if isinstance(provider, GPU4PySCFProvider) else type(provider).__name__
    result = {"provider": name, "mean_field": type(base).__name__,
              "density_convention": provider.density_convention,
              "symbols": [mol.atom_symbol(i) for i in range(mol.natm)],
              "charges": mol.atom_charges(), "basis": mol._basis, "cart": mol.cart,
              "charge": mol.charge, "spin": mol.spin, "nucmod": mol.nucmod, "nucprop": mol.nucprop,
              "settings": {k: getattr(mf, k, None) for k in
                           ("direct_scf", "direct_scf_tol", "small_rho_cutoff", "max_memory",
                            "xc", "nlc", "disp", "gcp")}}
    if hasattr(mf, "with_df"):
        from pyscf.df import df
        if type(mf.with_df) is not df.DF or any(inspect.isroutine(v) for v in vars(mf.with_df).values()):
            raise ValueError("Restart cannot preserve custom density fitting")
        result["density_fitting"] = {"auxbasis": mf.with_df.auxbasis, "only_dfj": mf.only_dfj}
    if isinstance(mf, dft.rks.KohnShamDFT):
        if type(mf._numint) is not dft.numint.NumInt or any(inspect.isroutine(v) for v in vars(mf._numint).values()):
            raise ValueError("Restart cannot preserve custom numerical integration")
        result["numint"] = {k: getattr(mf._numint, k, None) for k in ("omega", "cutoff")}
        result["grids"] = {k: getattr(mf.grids, k, None) for k in
                           ("level", "atom_grid", "prune", "radi_method", "radii_adjust", "becke_scheme",
                            "atomic_radii", "alignment", "cutoff", "symmetry")}
    return _plain(result)


def _nac_settings(nac):
    from .backends.pyscf_nac import ZVectorNAC
    if nac is None:
        return None
    if type(nac) is ZVectorNAC:
        return {"kind": "zvector", "options": vars(nac).copy()}
    if type(nac) is PaperFD:
        settings = vars(nac).copy()
        settings["options"] = asdict(nac.options)
        return {"kind": "paper_fd", "options": settings}
    raise ValueError("Restart requires a built-in NAC implementation")


def _write_state(arrays, prefix, state):
    for spin, (c, f) in enumerate(zip(state.coefficients, state.occupations)):
        arrays[f"{prefix}_c{spin}"] = to_host(c)
        arrays[f"{prefix}_f{spin}"] = f


def _read_state(arrays, prefix, nspin, xp=np):
    return OrbitalState(tuple(xp.asarray(arrays[f"{prefix}_c{s}"]) for s in range(nspin)),
                        tuple(arrays[f"{prefix}_f{s}"] for s in range(nspin)))


def save_checkpoint(run, path):
    """Save the last accepted trajectory state and history, replacing atomically."""
    import pyscf
    execution = run.execution
    if type(execution) is not LocalExecution:
        raise ValueError("Restart requires LocalExecution")
    for provider in execution.providers:
        if hasattr(provider, "capture_grids"):
            provider.capture_grids()
    models = [_model(p) for p in execution.providers]
    if run.method is not None and (type(run.method) is not MASH or set(vars(run.method)) != {"coefficients"}):
        raise ValueError("Restart requires the built-in MASH method")
    arrays = {"positions": run.nuclei.positions, "momenta": run.nuclei.momenta, "masses": run.nuclei.masses,
              "energies": run.observation.energies, "forces": run.observation.forces,
              "initial_energies": run.initial_energies, "hop_events": np.array(run.hop_events, dtype=HOP_EVENT_DTYPE)}
    if run.method is not None:
        coefficients_to_spin(run.method.coefficients)
        arrays["method_coefficients"] = run.method.coefficients
    for key in _FRAME_FIELDS:
        arrays[f"frame_{key}"] = np.array([f[key] for f in run.frames])
    stored = []
    for i, (provider, state) in enumerate(zip(execution.providers, execution.states)):
        _write_state(arrays, f"state{i}", state)
        if execution.gauge is not None:
            _write_state(arrays, f"gauge{i}", execution.gauge[i])
        names = []
        for key in _MO_FIELDS:
            value = getattr(provider.mf, key, None)
            if value is not None:
                arrays[f"mf{i}_{key}"] = np.asarray(value)
                names.append(key)
        for key in _GRID_FIELDS if hasattr(provider.mf, "grids") else ():
            value = getattr(provider.mf.grids, key, None)
            if value is not None:
                arrays[f"grid{i}_{key}"] = value
                names.append(f"grid_{key}")
        stored.append(names)
    force_states = tuple(np.flatnonzero(np.isfinite(run.observation.forces).all(axis=(1, 2))))
    if (run.active_state not in force_states or any(not np.isnan(row).all() for i, row in
            enumerate(run.observation.forces) if i not in force_states)):
        raise ValueError("Checkpoint forces must be finite for available states and NaN otherwise")
    info = {"format": 2, "force_states": force_states, "pyscf": pyscf.__version__, "models": models, "stored": stored,
            "gpu_forces": [getattr(p, "gpu_forces", None) for p in execution.providers],
            "options": asdict(execution.options), "alphas": execution.alphas,
            "time": execution.time, "initial_time": run.initial_time, "nuclear_dt": run.nuclear_dt,
            "mash": run.method is not None, "nac": _nac_settings(run.nac), "hop_tolerance": run.hop_tolerance,
            "frames": len(run.frames), "failure": run.failure}
    arrays["metadata"] = np.array(json.dumps(_plain(info), sort_keys=True, allow_nan=False))
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = stream.name
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, providers):
    """Build a trajectory from numeric data and matching trusted providers, without SCF."""
    import pyscf
    from .backends.pyscf_nac import ZVectorNAC
    providers = tuple(providers)
    with np.load(path, allow_pickle=False) as arrays:
        info = json.loads(str(arrays["metadata"].item()))
        if info["format"] not in (1, 2) or info["pyscf"] != pyscf.__version__:
            raise ValueError("Checkpoint format or PySCF version does not match")
        if [_model(p) for p in providers] != info["models"]:
            raise ValueError("Checkpoint provider/model settings do not match")
        nuclei = NuclearState(arrays["positions"], arrays["momenta"], arrays["masses"])
        candidates = deepcopy(providers)
        gpu_forces = info.get("gpu_forces", [False if hasattr(p, "gpu_forces") else None for p in candidates])
        if not isinstance(gpu_forces, list) or len(gpu_forces) != len(candidates):
            raise ValueError("Invalid checkpoint GPU force modes")
        for provider, mode in zip(candidates, gpu_forces):
            if hasattr(provider, "gpu_forces"):
                if type(mode) is not bool:
                    raise ValueError("Invalid checkpoint GPU force mode")
                provider.gpu_forces = mode or getattr(provider, "device_resident", False)
            elif mode is not None:
                raise ValueError("CPU provider cannot request GPU forces")
        states, gauge = [], []
        for i, provider in enumerate(candidates):
            provider.prepare_geometry(nuclei.positions)
            nspin = 2 if provider.unrestricted else 1
            xp = getattr(provider, "array_module", np)
            states.append(_read_state(arrays, f"state{i}", nspin, xp))
            if len(providers) == 2:
                gauge.append(_read_state(arrays, f"gauge{i}", nspin, xp))
            for key in _MO_FIELDS:
                value = arrays[f"mf{i}_{key}"] if key in info["stored"][i] else None
                if value is not None and not np.isfinite(value).all():
                    raise ValueError("Non-finite checkpoint mean-field data")
                setattr(provider.mf, key, value)
            c, f, e = (getattr(provider.mf, key) for key in _MO_FIELDS)
            if c is not None:
                if c.ndim != nspin + 1 or c.shape[-2] != provider.mf.mol.nao:
                    raise ValueError("Invalid checkpoint mean-field orbital shape")
                expected = (2, c.shape[-1]) if nspin == 2 else (c.shape[-1],)
                if any(v is not None and v.shape != expected for v in (f, e)):
                    raise ValueError("Invalid checkpoint mean-field occupations or energies")
            for key in _GRID_FIELDS if hasattr(provider.mf, "grids") else ():
                value = arrays[f"grid{i}_{key}"] if f"grid_{key}" in info["stored"][i] else None
                if value is not None and not np.isfinite(value).all():
                    raise ValueError("Non-finite checkpoint quadrature")
                setattr(provider.mf.grids, key, value)
            if hasattr(provider.mf, "grids"):
                coords, weights = provider.mf.grids.coords, provider.mf.grids.weights
                if (coords is None) != (weights is None) or (coords is not None and
                        (coords.ndim != 2 or coords.shape[1] != 3 or weights.shape != (len(coords),))):
                    raise ValueError("Invalid checkpoint quadrature shape")
            provider.mf._opt = {}
            provider.mf._eri = None
            if hasattr(provider.mf, "with_df"):
                provider.mf.with_df.reset(provider.mf.mol)
            if getattr(provider, "device_resident", False):
                # Rebuild from the restored quadrature, never from transient GPU caches.
                provider._gpu_mf = None
        execution = LocalExecution(candidates, states, alphas=info["alphas"],
                                   options=CNOptions(**info["options"]), time=info["time"])
        if gauge:
            xp = getattr(candidates[0], "array_module", np)
            for state, frame in zip(execution.states, gauge):
                if (frame.orthogonality_error(candidates[0].overlap) > 1e-8 or
                        not xp.allclose(frame.density(), state.density(), atol=1e-10, rtol=0)):
                    raise ValueError("Checkpoint DTO gauge does not match its state")
            execution.gauge = tuple(gauge)
        method = None
        if info["mash"]:
            coefficients = arrays["method_coefficients"]
            method = MASH(coefficients_to_spin(coefficients))
            method.coefficients = coefficients.copy()
        nac = None
        if info["nac"] is not None:
            settings = info["nac"]["options"].copy()
            if info["nac"]["kind"] == "zvector":
                nac = ZVectorNAC(**settings)
            elif info["nac"]["kind"] == "paper_fd":
                settings["options"] = CNOptions(**settings["options"])
                nac = PaperFD(**settings)
            else:
                raise ValueError("Unknown checkpoint NAC implementation")
        run = Trajectory(execution, nuclei, method=method, nac=nac, hop_tolerance=info["hop_tolerance"])
        energies, forces = arrays["energies"], arrays["forces"]
        selected = _force_states(info["force_states"] if info["format"] == 2 else True, len(providers))
        if (run.active_state not in selected or forces.shape != run.observation.forces.shape or
                not np.isfinite(forces[list(selected)]).all() or any(not np.isnan(row).all()
                for i, row in enumerate(forces) if i not in selected)):
            raise ValueError("Invalid checkpoint force availability")
        if not np.isfinite(run.observation.forces[list(selected)]).all():
            run.observation = execution.observe(forces=selected)
        if (energies.shape != run.observation.energies.shape or forces.shape != run.observation.forces.shape or
                not np.allclose(energies, run.observation.energies, atol=1e-8, rtol=0) or
                not np.allclose(forces, run.observation.forces, atol=1e-8, rtol=0, equal_nan=True)):
            raise ValueError("Checkpoint observations do not match the electronic state")
        run.observation = Observation(execution.time, energies.copy(), forces.copy())
        count = info["frames"]
        if type(count) is not int or count < 0 or not np.isfinite(info["initial_time"]):
            raise ValueError("Invalid checkpoint frame count or initial time")
        fields = {key: arrays[f"frame_{key}"] for key in _FRAME_FIELDS}
        natom, nstate = len(nuclei.masses), len(providers)
        shapes = {"time": (), "positions": (natom, 3), "momenta": (natom, 3), "energies": (nstate,),
                  "active_state": (), "kinetic_energy": (), "total_energy": (),
                  "spin": (3,) if method is not None else (0,), "overlap": (2, 2) if nstate == 2 else (0, 0)}
        for key, data in fields.items():
            if (data.shape != ((count, *shapes[key]) if count else (0,)) or not np.isfinite(data).all()):
                raise ValueError(f"Invalid checkpoint frame data: {key}")
        dt = info["nuclear_dt"]
        if dt is not None and (not np.isfinite(dt) or dt <= 0):
            raise ValueError("Invalid checkpoint nuclear timestep")
        if count and (dt is None or not np.allclose(fields["time"], info["initial_time"] +
                np.arange(1, count + 1) * dt, atol=1e-10, rtol=0) or
                abs(fields["time"][-1] - run.time) > 1e-10):
            raise ValueError("Checkpoint frame times do not match the accepted clock")
        if not count and abs(info["initial_time"] - run.time) > 1e-10:
            raise ValueError("Empty checkpoint history does not match its initial clock")
        if count and (not np.isin(fields["active_state"], range(nstate)).all() or
                not np.allclose(fields["positions"][-1], nuclei.positions, atol=1e-12, rtol=0) or
                not np.allclose(fields["momenta"][-1], nuclei.momenta, atol=1e-12, rtol=0) or
                not np.allclose(fields["energies"][-1], energies, atol=1e-12, rtol=0) or
                fields["active_state"][-1] != run.active_state or
                not np.allclose(fields["spin"][-1], method.spin if method is not None else [], atol=1e-12, rtol=0) or
                abs(fields["kinetic_energy"][-1] - nuclei.kinetic_energy) > 1e-12 or
                abs(fields["total_energy"][-1] - energies[run.active_state] - nuclei.kinetic_energy) > 1e-12):
            raise ValueError("Checkpoint history does not match the accepted state")
        run.frames = [{key: data[i].copy() for key, data in fields.items()} for i in range(count)]
        run.initial_time, run.nuclear_dt = info["initial_time"], dt
        run.initial_energies = arrays["initial_energies"].copy()
        if run.initial_energies.shape != (nstate,) or not np.isfinite(run.initial_energies).all():
            raise ValueError("Invalid checkpoint initial energies")
        events = arrays["hop_events"]
        if events.dtype != HOP_EVENT_DTYPE or events.ndim != 1:
            raise ValueError("Invalid checkpoint hop events")
        if (not np.isfinite(events['time']).all() or not np.isin(events['from_state'], (0, 1)).all() or
                not np.isin(events['to_state'], (0, 1)).all() or
                not np.isin(events['outcome'], ('accepted', 'frustrated', 'failed', 'zero_nac')).all() or
                np.any(events['step'] < 1) or np.any(events['step'] > count + 1)):
            raise ValueError("Invalid checkpoint hop event values")
        run.hop_events = events.tolist()
        run.failure = None if info["failure"] is None else tuple(info["failure"])
        if run.failure is not None and (len(run.failure) != 3 or run.failure[0] != count + 1 or
                not all(isinstance(v, str) for v in run.failure[1:])):
            raise ValueError("Invalid checkpoint failure record")
        return run

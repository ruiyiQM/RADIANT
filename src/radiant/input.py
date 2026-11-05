# Copyright (c) 2026 Ruiyi Zhou
"""Minimal TOML entry point for a single SCF-initialized RT state.

MASH currently takes explicitly prepared orbital states through the Python
API. No legacy spin-file ordering or excited SCF branch is inferred here.
"""

from pathlib import Path
import tomllib

import numpy as np

from .core.propagation import CNOptions
from .core.state import NuclearState


def run_toml(path):
    """Read atomic units explicitly, initialize once, and run single-state CN/PT-CN.

    Required sections are system, backend, state and trajectory. The system
    supplies an inline PySCF atom string, explicit Bohr units, nuclear masses
    (electron masses), and momenta (a.u.). State.initial must be 'scf'.
    """
    path = Path(path)
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    allowed = {
        "system": {"atoms", "unit", "charge", "spin", "masses", "momenta"},
        "backend": {"model", "basis", "xc", "density_convention"},
        "state": {"initial", "alpha"},
        "propagation": {"variant", "tolerance", "max_iterations"},
        "trajectory": {"dt", "steps", "n_substeps"},
        "output": {"file"},
    }
    if config.keys() - allowed.keys():
        raise ValueError("Unknown TOML section")
    for section, values in config.items():
        if not isinstance(values, dict) or values.keys() - allowed[section]:
            raise ValueError(f"Unknown field in {section}")
    system, backend = config["system"], config["backend"]
    state, trajectory = config["state"], config["trajectory"]
    if system["unit"] != "Bohr" or state["initial"] != "scf":
        raise ValueError("This entry point requires explicit Bohr units and initial='scf'")
    if (not np.isfinite([state["alpha"], trajectory["dt"]]).all()
            or state["alpha"] <= 0 or trajectory["dt"] <= 0):
        raise ValueError("alpha and nuclear dt must be finite and positive")
    for key, minimum in (("steps", 0), ("n_substeps", 1)):
        if (not isinstance(trajectory[key], int) or isinstance(trajectory[key], bool)
                or trajectory[key] < minimum):
            raise ValueError(f"{key} must be an integer >= {minimum}")
    options = CNOptions(**config.get("propagation", {}))
    # Importing TOML or the numerical tools does not load an electronic engine.
    from pyscf import dft, gto, scf
    from .backends.pyscf import PySCFProvider
    from .core.execution import LocalExecution
    from .driver import Trajectory

    model = backend["model"]
    factories = {"rhf": scf.RHF, "rks": dft.RKS, "uhf": scf.UHF, "uks": dft.UKS}
    if model not in factories or (not model.endswith("ks") and "xc" in backend):
        raise ValueError("Select rhf/rks/uhf/uks; xc applies only to KS models")
    mol = gto.M(atom=system["atoms"], basis=backend["basis"], unit="Bohr",
                charge=system.get("charge", 0), spin=system.get("spin", 0), verbose=0)
    nuclei = NuclearState(mol.atom_coords(), system["momenta"], system["masses"])
    mf = factories[model](mol)
    mf.chkfile = None
    if model.endswith("ks"):
        mf.xc = backend["xc"]
    mf.kernel()
    provider = PySCFProvider(mf, density_convention=backend.get("density_convention", "complex"))
    execution = LocalExecution([provider], alphas=[state["alpha"]], options=options)
    run = Trajectory(execution, nuclei)
    output = config.get("output", {}).get("file")
    run.run(trajectory["steps"], trajectory["dt"], n_substeps=trajectory["n_substeps"],
            output=None if output is None else path.parent / output)
    return run

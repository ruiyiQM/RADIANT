# RADIANT

**R**eal-time **ADIA**batic states for **N**onadiabatic **T**rajectories

RADIANT is a Python library for real-time electronic propagation and
nonadiabatic molecular dynamics. It combines orbital Crank–Nicolson (CN),
parallel-transport CN (PT-CN), moving-basis Löwdin transport, and a local
PySCF backend.

**Status: early development.** The API is not yet stable. Numerical regressions
cover the implemented routes; they do not certify general excited-state
stability, long-time accuracy, or an ensemble reproduction of the paper.

## Installation

Use Python 3.11 or later. Install the development branch from a source checkout:

```bash
git clone --branch develop https://github.com/ruiyiQM/RADIANT.git
cd RADIANT
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[pyscf]'
```

The distribution is called `pyradiant`; the import name is `radiant`.
NumPy and SciPy are required. The `pyscf` extra installs PySCF >=2.12 for the
molecular backend and the full regression suite. A base installation without
the molecular backend is available with `pip install -e .`.

## Regression tests

CMake >=3.16 and the PySCF extra are required:

```bash
cmake -S . -B build -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
ctest --test-dir build --output-on-failure
```

The nine CTest suites contain 74 unittest methods. They exercise numerical
utilities, CN and PT-CN, adiabatic following, moving Löwdin transport, PySCF
energies and forces, snapshots and rollback, DTO/FD/MASH, and TOML input.
The selected Table 1 fixture checks recomputed unit directions rather than
the missing original orbitals or general complex coupling magnitudes.

## Implemented scope

- Self-consistent plain and Anderson-accelerated orbital CN, with the original
  AO residual used for acceptance and failure on nonconvergence.
- Single-state PT-CN for occupied restricted columns with occupations 2
  and/or 1, or per-spin unrestricted unit blocks, without virtual columns.
  Restricted doubly and singly occupied columns form separate gauge groups:
  only rotations inside a group are removed, while their mutual coupling is
  propagated. Two-state PT-CN and PT finite-difference couplings are rejected.
  The trapezoidal PT-CN root is orthonormalized before it is returned; each
  step's `Convergence` reports the root's Gram error
  (`raw_orthogonality_error`) and the PT-CN residual of the returned orbitals
  (`returned_residual`), so this projection can be monitored.
- Molecular RHF/RKS/UHF/UKS Focks, physical energies, and forces at fixed
  symmetric Löwdin coordinates through `PySCFProvider`.
- `LocalExecution`, `follow_path`, and `Trajectory` for local electronic
  propagation, single-state Ehrenfest dynamics, or explicitly prepared
  restricted two-state MASH. Energy scaling applies to electronics only.
- Reusable in-memory snapshots and step/output rollback. Trajectory NPZ
  files contain numeric results, not persistent electronic restart states.

The explicit `paper_real` density convention is separate from native complex
densities. Large-alpha following and larger time steps require an appropriate
state and observable accuracy check; nonlinear convergence alone is insufficient.
Online adaptive alpha, FSSH, FHI-aims/DFTB+ backends, and persistent electronic
restart loading are not implemented.

## Minimal single-state run

Save the following as `input.toml`. Positions are Bohr, masses are electron
masses, and momenta and times use atomic units. This route initializes SCF
once; prepared excited states and two-state MASH use the Python API.

```toml
[system]
atoms = "H 0 0 0; H 0 0 1.4"
unit = "Bohr"
masses = [1836.0, 1836.0]
momenta = [[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]]

[backend]
model = "rhf"
basis = "sto-3g"

[state]
initial = "scf"
alpha = 2.0

[trajectory]
dt = 0.1
steps = 2
n_substeps = 2

[output]
file = "trajectory.npz"
```

```python
from radiant.input import run_toml

trajectory = run_toml("input.toml")
```

## Relation to TDMASH_pack

The method combining energy-scaled RT-TDDFT with MASH is described in

> R. Zhou and J. O. Richardson, *Surface Hopping Dynamics on Adiabatic States
> Propagated with Real-Time TDDFT*, arXiv:2610.41686 (2026).

Its reference implementation,
[TDMASH_pack](https://github.com/ruiyiQM/TDMASH_pack), remains frozen to
reproduce that paper. RADIANT is its successor for continued development.

Migrated source files retain their original MIT copyright notices.

## License

MIT; see [`LICENSE`](LICENSE).

## Contact

Ruiyi Zhou, ETH Zurich:
[ruiyi.zhou@phys.chem.ethz.ch](mailto:ruiyi.zhou@phys.chem.ethz.ch).

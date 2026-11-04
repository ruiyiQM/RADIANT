# RADIANT

**R**eal-time **ADIA**batic states for **N**onadiabatic **T**rajectories

> **Status: early development.** This release reserves the package name and
> contains only a small numerical utility. The API is not yet stable.

RADIANT will be a backend-agnostic library for nonadiabatic molecular dynamics
on electronic states obtained by real-time propagation. Energy-scaled
real-time TDDFT lets each propagated determinant follow its own adiabatic
surface; a dynamical transition orbital (DTO) gauge then provides the overlaps
and couplings between these states. Different trajectory methods are built on
the same electronic core.

Planned scope:

- **Dynamics methods:** mapping approach to surface hopping (MASH),
  fewest-switches surface hopping (FSSH), Ehrenfest dynamics, and adiabatic
  excited-state dynamics with large energy scaling.
- **Electronic propagation:** Crank–Nicolson and parallel-transport
  Crank–Nicolson (PT-CN) with Löwdin transport for moving atom-centred bases.
- **Backends:** PySCF (reference), FHI-aims, and further codes through a
  common interface.

## Installation

```bash
pip install pyradiant
```

The distribution is called `pyradiant`; the import name is `radiant`:

```python
import radiant
print(radiant.__version__)
```

## Relation to TDMASH_pack

The method combining energy-scaled RT-TDDFT with MASH is described in

> R. Zhou and J. O. Richardson, *Surface Hopping Dynamics on Adiabatic States
> Propagated with Real-Time TDDFT*, arXiv:2610.41686 (2026).

Its reference implementation,
[TDMASH_pack](https://github.com/ruiyiQM/TDMASH_pack), remains frozen to
reproduce that paper. RADIANT is its successor for continued development.

## License

MIT; see [`LICENSE`](LICENSE).

## Contact

Ruiyi Zhou, ETH Zurich:
[ruiyi.zhou@phys.chem.ethz.ch](mailto:ruiyi.zhou@phys.chem.ethz.ch).

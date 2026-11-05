# Copyright (c) 2026 Ruiyi Zhou
"""In-process electronic execution with atomic all-state advance/restore.

This endpoint combines local matrix providers and core propagation. A future
native propagator implements the small-result interface directly, without
supplying Focks or transporting its resident orbitals to this class.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .propagation import CNOptions, crank_nicolson, validate_pt_state
from .transport import transport


@dataclass(frozen=True)
class Observation:
    time: float
    energies: np.ndarray
    forces: np.ndarray | None


@dataclass(frozen=True)
class StepResult:
    left_time: float
    observation: Observation
    diagnostics: tuple
    overlap_raw: np.ndarray | None = None
    overlap_unitary: np.ndarray | None = None
    singular_values: np.ndarray | None = None
    gram_right: np.ndarray | None = None


@dataclass(frozen=True)
class _Snapshot:
    owner: object
    providers: tuple
    states: tuple
    time: float
    gauge: object


class LocalExecution:
    """One or two states; heavy arrays and provider caches remain local."""

    def __init__(self, providers, states=None, *, alphas=None, options=CNOptions(), time=0.):
        self.providers = tuple(providers)
        if len(self.providers) not in (1, 2):
            raise ValueError("Only one-state RT or two-state restricted DTO is supported")
        self.states = tuple(p.initial_state() for p in self.providers) if states is None else tuple(s.copy() for s in states)
        if len(self.states) != len(self.providers):
            raise ValueError("Provider/state counts do not match")
        self.alphas = tuple(1. for _ in self.states) if alphas is None else tuple(alphas)
        if len(self.alphas) != len(self.states) or not np.isfinite(self.alphas).all() or min(self.alphas) <= 0:
            raise ValueError("Each electronic state needs a positive finite alpha")
        self.options = options
        self._validate_propagation(options)
        self.time = float(time)
        self._identity = object()
        self.gauge = None
        if not np.isfinite(self.time):
            raise ValueError("Initial time must be finite")
        for provider, state in zip(self.providers, self.states):
            if not np.allclose(provider.positions, self.positions, atol=1e-12, rtol=0):
                raise ValueError("States must share one molecular geometry")
            if state.orthogonality_error(provider.overlap) > 1e-8:
                raise ValueError("Initial state is not S-orthonormal")
            cross = provider.cross_overlap(self.providers[0].mf.mol, provider.mf.mol)
            common = self.providers[0].overlap
            if (provider.overlap.shape != common.shape or cross.shape != common.shape
                    or not np.allclose(provider.overlap, common, atol=1e-12, rtol=0)
                    or not np.allclose(cross, common, atol=1e-12, rtol=0)):
                raise ValueError("States must share the same ordered AO basis")
        if len(self.states) == 2:
            from .overlap import dto_pair
            self.gauge = dto_pair(self.states, self.providers[0].overlap)

    @property
    def positions(self):
        return self.providers[0].positions.copy()

    def _validate_propagation(self, options):
        if options.variant == "pt_cn":
            if len(self.states) != 1:
                raise ValueError("PT-CN is validated only for single-state execution")
            validate_pt_state(self.states[0])

    def observe(self, *, forces=True):
        energies = np.array([p.energy(s) for p, s in zip(self.providers, self.states)])
        force = np.array([p.forces(s) for p, s in zip(self.providers, self.states)]) if forces else None
        if not np.isfinite(energies).all() or (force is not None and not np.isfinite(force).all()):
            raise ValueError("Non-finite endpoint observation")
        return Observation(self.time, energies, force)

    def snapshot(self):
        """Opaque reusable snapshot including geometry, all states and DTO gauge."""
        return _Snapshot(self._identity, deepcopy(self.providers),
                         tuple(s.copy() for s in self.states), self.time, deepcopy(self.gauge))

    def restore(self, snapshot):
        self._validate_snapshot(snapshot)
        self.providers = deepcopy(snapshot.providers)
        self.states = tuple(s.copy() for s in snapshot.states)
        self.time = snapshot.time
        self.gauge = deepcopy(snapshot.gauge)

    def _validate_snapshot(self, snapshot):
        if not isinstance(snapshot, _Snapshot) or snapshot.owner is not self._identity:
            raise ValueError("Snapshot belongs to another execution session")

    def overlap(self, left, right):
        """Return only the two-state cross-geometry singlet overlap matrix."""
        self._validate_snapshot(left)
        self._validate_snapshot(right)
        if len(self.states) != 2:
            raise ValueError("DTO overlap requires a restricted two-state pair")
        from .overlap import pair_overlap
        cross = self.providers[0].cross_overlap(left.providers[0].mf.mol, right.providers[0].mf.mol)
        return pair_overlap(left.gauge, right.gauge, cross)

    def advance(self, end, dt, *, midpoint=None, n_substeps=1, alphas=None,
                options=None, gauge_reference=None, forces=True):
        """Transport, propagate at a fixed midpoint, transport, then observe.

        dt is physical, including signed field-free FD intervals. No time,
        density, cache, gauge or state is accepted until every state succeeds.
        """
        if (not np.isfinite(dt) or dt == 0 or not isinstance(n_substeps, int)
                or isinstance(n_substeps, bool) or n_substeps < 1):
            raise ValueError("Use a nonzero finite dt and positive integer substep count")
        end = np.asarray(end, dtype=float)
        midpoint = (self.positions + end) / 2 if midpoint is None else np.asarray(midpoint, dtype=float)
        if end.shape != self.positions.shape or midpoint.shape != end.shape or not np.isfinite([end, midpoint]).all():
            raise ValueError("Invalid electronic path geometry")
        scales = self.alphas if alphas is None else tuple(alphas)
        if len(scales) != len(self.states) or not np.isfinite(scales).all() or min(scales) <= 0:
            raise ValueError("Each state needs a positive finite alpha")
        propagation = self.options if options is None else options
        self._validate_propagation(propagation)
        before = self.snapshot()
        anchor = before if gauge_reference is None else gauge_reference
        self._validate_snapshot(anchor)
        try:
            states = list(self.states)
            diagnostics = []
            for i, (provider, state, alpha) in enumerate(zip(self.providers, states, scales)):
                old_overlap = provider.overlap
                middle_overlap = provider.prepare_geometry(midpoint)
                state = transport(state, old_overlap, middle_overlap)
                for k in range(n_substeps):
                    state, diagnostic = crank_nicolson(state, middle_overlap, provider.fock,
                        before.time + k * dt / n_substeps, dt / n_substeps, alpha=alpha,
                        options=propagation)
                    diagnostics.append(diagnostic)
                new_overlap = provider.prepare_geometry(end)
                states[i] = transport(state, middle_overlap, new_overlap)
            self.states = tuple(states)
            self.time = before.time + dt
            raw = None
            unitary = singular = gram = None
            if len(states) == 2:
                from .overlap import dto_pair, overlap_coupling, pair_overlap
                cross = self.providers[0].cross_overlap(anchor.providers[0].mf.mol, self.providers[0].mf.mol)
                self.gauge = dto_pair(self.states, self.providers[0].overlap,
                    reference=anchor.gauge, cross=cross, align_all=gauge_reference is not None)
                raw = self.overlap(before, self.snapshot())
                unitary, singular, _ = overlap_coupling(raw, dt)
                gram = pair_overlap(self.gauge, self.gauge, self.providers[0].overlap)
            return StepResult(before.time, self.observe(forces=forces), tuple(diagnostics),
                              raw, unitary, singular, gram)
        except BaseException:
            self.restore(before)
            raise

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

"""Physical nuclear time, accepted frames and whole-step transactions.

The resident execution endpoint returns small energies/forces/overlaps. The
driver never requests a Fock matrix or an electronic state from that endpoint.
"""

from copy import deepcopy
import os
from pathlib import Path
import tempfile

import numpy as np

from .core.coupling import PaperFD, ZeroNACError


HOP_EVENT_DTYPE = np.dtype([
    ("step", "i8"), ("time", "f8"), ("from_state", "i8"), ("to_state", "i8"),
    ("outcome", "U16"), ("kinetic_energy_parallel", "f8"), ("energy_gap", "f8")])


class Trajectory:
    """Single-RT-state Ehrenfest or paper two-state CN+MASH.

    The single-state route also covers large-alpha following of an explicitly
    prepared branch; its force and clock remain physical and unscaled.
    """

    def __init__(self, execution, nuclei, *, method=None, nac=None):
        self.execution, self.nuclei = execution, deepcopy(nuclei)
        self.method = method
        self.nac = PaperFD() if nac is None and method is not None else nac
        if not np.allclose(execution.positions, nuclei.positions, atol=1e-12, rtol=0):
            raise ValueError("Electronic and nuclear geometries differ")
        self.observation = execution.observe(forces=True)
        if len(self.observation.energies) != (1 if method is None else 2):
            raise ValueError("Method/state count mismatch")
        self.time = self.observation.time
        self.initial_time = self.time
        self.nuclear_dt = None
        self.frames, self.hop_events = [], []
        self.initial_energies = self.observation.energies.copy()
        self.failure = None

    @property
    def active_state(self):
        return 0 if self.method is None else self.method.active_state

    def step(self, dt, *, n_substeps=1, output=None, requested_steps=None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Nuclear dt must be positive and finite")
        if not isinstance(n_substeps, int) or isinstance(n_substeps, bool) or n_substeps < 1:
            raise ValueError("n_substeps must be a positive integer")
        if self.nuclear_dt is not None and dt != self.nuclear_dt:
            raise ValueError("A trajectory uses one fixed nuclear dt")
        before = self.execution.snapshot()
        nuclei_before, method_before = deepcopy(self.nuclei), deepcopy(self.method)
        observation_before, time_before = self.observation, self.time
        dt_before, event_count = self.nuclear_dt, len(self.hop_events)
        active = self.active_state
        number = len(self.frames) + 1
        target_time = self.initial_time + number * dt
        event = None
        try:
            half_momentum = self.nuclei.momenta + .5 * dt * self.observation.forces[active]
            velocity = half_momentum / self.nuclei.masses[:, None]
            midpoint = self.nuclei.positions + .5 * dt * velocity
            end = self.nuclei.positions + dt * velocity
            result = self.execution.advance(end, target_time - self.time, midpoint=midpoint,
                                            n_substeps=n_substeps, forces=True)
            observation = result.observation
            momentum = half_momentum + .5 * dt * observation.forces[active]
            if self.method is not None:
                self.method.propagate(self.observation.energies, observation.energies,
                                      result.overlap_raw, dt)
                candidate = self.method.active_state
                if candidate != active:
                    event = [number, target_time, active, candidate, "failed", np.nan,
                             observation.energies[candidate] - observation.energies[active]]
                    try:
                        direction = self.nac.compute(self.execution).real_direction
                    except ZeroNACError:
                        event[4] = "zero_nac"
                        raise
                    momentum, outcome, kinetic = self.method.resolve_hop(momentum, direction,
                        observation.energies, self.nuclei.masses, initial_state=active)
                    event[4:6] = [outcome, kinetic]
            if not np.isfinite(momentum).all():
                raise ValueError("Non-finite nuclear momentum")
            self.nuclei.positions, self.nuclei.momenta = end, momentum
            frame = {
                "time": observation.time, "positions": end.copy(), "momenta": momentum.copy(),
                "energies": observation.energies.copy(), "active_state": self.active_state,
                "kinetic_energy": self.nuclei.kinetic_energy,
                "total_energy": observation.energies[self.active_state] + self.nuclei.kinetic_energy,
                "spin": self.method.spin.copy() if self.method is not None else np.empty(0),
                "overlap": result.overlap_raw.copy() if result.overlap_raw is not None else np.empty((0,0)),
            }
            self.observation, self.time = observation, observation.time
            self.nuclear_dt = dt
            self.frames.append(frame)
            if event is not None:
                self.hop_events.append(tuple(event))
            self.failure = None
            if output is not None:
                self.save(output, requested_steps=number if requested_steps is None else requested_steps)
        except BaseException as exc:
            self.execution.restore(before)
            self.nuclei = nuclei_before
            self.observation, self.time, self.nuclear_dt = observation_before, time_before, dt_before
            del self.frames[number - 1:]
            del self.hop_events[event_count:]
            if self.method is not None:
                self.method.__dict__.clear()
                self.method.__dict__.update(method_before.__dict__)
            self.failure = (number, "zero_nac" if isinstance(exc, ZeroNACError) else type(exc).__name__, str(exc))
            if event is not None:
                if event[4] != "zero_nac":
                    event[4] = "failed"
                self.hop_events.append(tuple(event))
            raise
        return frame

    def run(self, steps, dt, *, n_substeps=1, output=None):
        """Save completed frames and explicit failure status, including zero NAC."""
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if (not np.isfinite(dt) or dt <= 0 or not isinstance(n_substeps, int)
                or isinstance(n_substeps, bool) or n_substeps < 1
                or (self.nuclear_dt is not None and dt != self.nuclear_dt)):
            raise ValueError("Require the fixed positive nuclear dt and a positive integer substep count")
        requested = len(self.frames) + steps
        try:
            for _ in range(steps):
                self.step(dt, n_substeps=n_substeps, output=output, requested_steps=requested)
        except BaseException as exc:
            if output is not None and not isinstance(exc, OSError):
                self.save(output, requested_steps=requested)
            raise
        if steps == 0 and output is not None:
            self.save(output, requested_steps=requested)

    def save(self, path, *, requested_steps):
        """Atomically replace numeric output; no pickle or failed candidate frame."""
        path = Path(path)
        arrays = {key: np.array([f[key] for f in self.frames]) for key in (
            "time", "positions", "momenta", "energies", "active_state", "kinetic_energy", "total_energy", "spin", "overlap")}
        arrays.update(initial_energies=self.initial_energies,
            hop_events=np.array(self.hop_events, dtype=HOP_EVENT_DTYPE),
            completed=self.failure is None and len(self.frames) == requested_steps,
            completed_steps=len(self.frames), requested_steps=requested_steps,
            failed_step=-1 if self.failure is None else self.failure[0],
            failure_reason="" if self.failure is None else self.failure[1],
            failure_message="" if self.failure is None else self.failure[2],
            state_order=np.array(["ground", "excited"] if self.method is not None else ["rt_state"]))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                temporary = stream.name
                np.savez(stream, **arrays)
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)


def follow_path(execution, positions, times, *, midpoints=None, method=None, n_substeps=1):
    """Propagate a prescribed Bohr path; MASH variables evolve without hops.

    The first geometry/time must be the current accepted endpoint. Subsequent
    times are explicit physical times; no time grid or nuclear path is guessed.
    Optional midpoints supply one Bohr geometry per segment; otherwise the
    endpoint average is used. Archived midpoint geometries are passed unchanged.
    """
    positions, times = np.asarray(positions, dtype=float), np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 1 or positions.shape != (len(times), *execution.positions.shape)
            or not np.isfinite(positions).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isclose(times[0], execution.time, atol=1e-12, rtol=0)
            or not np.allclose(positions[0], execution.positions, atol=1e-12, rtol=0)):
        raise ValueError("Path must start at the accepted geometry/time and increase in time")
    if midpoints is not None:
        midpoints = np.asarray(midpoints, dtype=float)
        if (midpoints.shape != (len(times)-1, *execution.positions.shape)
                or not np.isfinite(midpoints).all()):
            raise ValueError("Supply one finite midpoint geometry per path segment")
    observation = execution.observe(forces=False)
    frames = []
    for index, (end, time) in enumerate(zip(positions[1:], times[1:])):
        before = execution.snapshot()
        method_before = deepcopy(method)
        try:
            result = execution.advance(end, time - observation.time,
                midpoint=None if midpoints is None else midpoints[index],
                n_substeps=n_substeps, forces=False)
            if method is not None:
                method.propagate(observation.energies, result.observation.energies,
                                 result.overlap_raw, time - observation.time)
            frames.append(result)
            observation = result.observation
        except BaseException:
            execution.restore(before)
            if method is not None:
                method.__dict__.clear()
                method.__dict__.update(method_before.__dict__)
            raise
    return frames

"""Small local Fock provider for transaction and method regressions."""

from types import SimpleNamespace

import numpy as np

from radiant.core.state import OrbitalState


class ToyProvider:
    trace = []

    def __init__(self, branch=0, *, rate=.02, force=0.):
        self.branch, self.rate, self.force = branch, rate, force
        self.mf = SimpleNamespace(mol=np.zeros((1, 3)))
        self.failure_time = None
        self.cache = None

    @property
    def positions(self):
        return self.mf.mol.copy()

    @property
    def overlap(self):
        return np.eye(2)

    def initial_state(self):
        return OrbitalState((np.eye(2, dtype=complex) if self.branch else np.eye(2, dtype=complex)[:, :1],),
                            ([1,1],) if self.branch else ([2],))

    def prepare_geometry(self, positions):
        self.mf.mol = np.array(positions, copy=True)
        return self.overlap

    def fock(self, dm, time):
        type(self).trace.append((self.branch, time, self.positions.copy(), dm.copy()))
        self.cache = dm.copy()
        if self.failure_time is not None and time >= self.failure_time:
            raise RuntimeError("Injected electronic failure")
        return np.array([[0., -1j*self.rate], [1j*self.rate, 0.]])

    def energy(self, state):
        return .3 * self.branch - self.force * self.positions[0,0]

    def forces(self, state):
        return np.array([[self.force, 0., 0.]])

    @staticmethod
    def cross_overlap(left, right):
        return np.eye(2) * np.exp(-.1 * np.sum((left-right)**2))

"""Numerical properties of the time integrators used for electronic propagation."""

import numpy as np


def cayley_frequency(omega, dt):
    """Return the frequency that Crank-Nicolson assigns to a linear mode.

    The Crank-Nicolson (Cayley) propagator maps a mode exp(-i omega t) onto a
    unit-modulus factor per step, so the amplitude is preserved exactly. The
    phase per step is 2 arctan(omega dt / 2), i.e. the propagated frequency is

        omega_eff = (2 / dt) arctan(omega dt / 2),

    which is bounded by pi / dt and has a relative error of about
    (omega dt)^2 / 12 for small omega dt.

    Parameters
    ----------
    omega : float or array_like
        Exact angular frequency (atomic units).
    dt : float
        Positive time step (atomic units).

    Returns
    -------
    float or ndarray
        Frequency represented by the discrete propagator.
    """
    dt = float(dt)
    if not dt > 0.0:
        raise ValueError("dt must be positive")
    omega = np.asarray(omega, dtype=float)
    return 2.0 / dt * np.arctan(0.5 * omega * dt)


def inverse_cayley_frequency(omega_eff, dt):
    """Recover the exact frequency from a Crank-Nicolson frequency.

    Inverts :func:`cayley_frequency` on its range |omega_eff| < pi / dt.
    """
    dt = float(dt)
    if not dt > 0.0:
        raise ValueError("dt must be positive")
    omega_eff = np.asarray(omega_eff, dtype=float)
    if np.any(np.abs(omega_eff) * dt >= np.pi):
        raise ValueError("|omega_eff| must be smaller than pi / dt")
    return 2.0 / dt * np.tan(0.5 * omega_eff * dt)

from __future__ import annotations

import numpy as np


def calculate_equivalent_full_cycles(
    time_s: list[float], current_a: list[float], nominal_capacity_ah: float
) -> list[float]:
    """Calculate EFC robustly when current samples are missing.

    If no finite current value exists, a NaN EFC vector is returned because an
    EFC axis cannot be derived from the available data. For isolated missing
    current samples, the cumulative value is held constant over that interval;
    this avoids poisoning the whole result with NaN while making no invented
    current contribution.
    """
    if nominal_capacity_ah <= 0:
        raise ValueError("nominal_capacity_ah must be > 0")

    t = np.asarray(time_s, dtype=float)
    i = np.asarray(current_a, dtype=float)
    if len(t) != len(i):
        raise ValueError("time_s and current_a must have equal length")
    if len(t) == 0:
        return []

    finite_i = np.isfinite(i)
    if not finite_i.any():
        return np.full(len(t), np.nan, dtype=float).tolist()

    dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isfinite(dt), np.maximum(dt, 0.0), 0.0)
    i_for_integration = np.where(finite_i, np.abs(i), 0.0)
    throughput_ah = np.cumsum(i_for_integration * dt / 3600.0)
    return (throughput_ah / (2.0 * nominal_capacity_ah)).tolist()

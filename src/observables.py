"""Observable computations for voter-model trajectories."""

from __future__ import annotations

import numpy as np


def magnetization(states: np.ndarray) -> float:
    """Compute magnetization m = (1/N) sum_i x_i."""
    return float(np.mean(states))


def _post_burn_in(series: np.ndarray, burn_in: int) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if burn_in >= arr.size:
        raise ValueError("burn_in must be smaller than series length.")
    return arr[burn_in:]


def time_average(series: np.ndarray, burn_in: int) -> float:
    """Mean of a series after burn-in."""
    return float(np.mean(_post_burn_in(series, burn_in)))


def variance(series: np.ndarray, burn_in: int) -> float:
    """Variance of a series after burn-in."""
    return float(np.var(_post_burn_in(series, burn_in)))

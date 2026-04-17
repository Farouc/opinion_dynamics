"""Observable computations for voter-model trajectories."""

from __future__ import annotations

import numpy as np


def magnetization(states: np.ndarray) -> float:
    """Compute magnetization m = (1/N) sum_i x_i."""
    return float(np.mean(states))


def positive_fraction(states: np.ndarray) -> float:
    """Fraction of nodes in state +1."""
    return float(np.mean(np.asarray(states) == 1))


def negative_fraction(states: np.ndarray) -> float:
    """Fraction of nodes in state -1."""
    return float(np.mean(np.asarray(states) == -1))


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


def dominance_sign(value: float, eps: float = 1e-12) -> int:
    """Return dominance sign: +1, -1, or 0 for near-tie."""
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def victory_indicator(mean_m: float, eps: float = 1e-12) -> int:
    """Positive-camp victory indicator: 1 if mean_m > 0 else 0."""
    return int(mean_m > eps)


def time_to_first_sign_change(
    series: np.ndarray,
    reference_sign: int | None = None,
    eps: float = 1e-12,
) -> int | None:
    """First 1-based time index where sign differs from the reference sign."""
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError("series must be one-dimensional.")
    if arr.size == 0:
        return None

    if reference_sign is None:
        reference_sign = 0
        for val in arr:
            s = dominance_sign(float(val), eps=eps)
            if s != 0:
                reference_sign = s
                break
        if reference_sign == 0:
            return None

    for i, val in enumerate(arr):
        s = dominance_sign(float(val), eps=eps)
        if s != 0 and s != int(reference_sign):
            return int(i + 1)

    return None


def time_to_threshold_crossing(
    series: np.ndarray,
    threshold: float = 0.0,
    direction: str = "above",
    strict: bool = True,
) -> int | None:
    """First 1-based time index crossing a threshold in the requested direction."""
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError("series must be one-dimensional.")
    if arr.size == 0:
        return None

    direction_key = str(direction).strip().lower()
    if direction_key not in {"above", "below"}:
        raise ValueError("direction must be one of {'above', 'below'}.")

    if direction_key == "above":
        mask = arr > threshold if strict else arr >= threshold
    else:
        mask = arr < threshold if strict else arr <= threshold

    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return int(idx[0] + 1)


def fraction_time_above_threshold(
    series: np.ndarray,
    burn_in: int = 0,
    threshold: float = 0.0,
    strict: bool = True,
) -> float:
    """Fraction of post-burn-in time spent above a threshold."""
    arr = _post_burn_in(series, burn_in=burn_in)
    return float(np.mean(arr > threshold if strict else arr >= threshold))


def fraction_time_below_threshold(
    series: np.ndarray,
    burn_in: int = 0,
    threshold: float = 0.0,
    strict: bool = True,
) -> float:
    """Fraction of post-burn-in time spent below a threshold."""
    arr = _post_burn_in(series, burn_in=burn_in)
    return float(np.mean(arr < threshold if strict else arr <= threshold))


def asymptotic_observables(
    magnetization_series: np.ndarray,
    burn_in: int,
    positive_series: np.ndarray | None = None,
    negative_series: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Asymptotic observables used for tipping analysis."""
    mean_m = time_average(magnetization_series, burn_in=burn_in)
    out: dict[str, float | int] = {
        "mean_magnetization": float(mean_m),
        "var_magnetization": variance(magnetization_series, burn_in=burn_in),
        "dominance_sign": int(dominance_sign(mean_m, eps=eps)),
        "victory_indicator": int(victory_indicator(mean_m, eps=eps)),
    }

    if positive_series is not None:
        out["mean_positive_fraction"] = float(time_average(positive_series, burn_in=burn_in))
    if negative_series is not None:
        out["mean_negative_fraction"] = float(time_average(negative_series, burn_in=burn_in))

    return out


def tipping_observables(
    magnetization_series: np.ndarray,
    burn_in: int,
    threshold: float = 0.0,
    eps: float = 1e-12,
) -> dict[str, float | int | None]:
    """Compute tipping-oriented observables from m(t)."""
    mean_m = time_average(magnetization_series, burn_in=burn_in)
    ref_sign = dominance_sign(float(magnetization_series[0]), eps=eps)
    if ref_sign == 0:
        ref_sign = None

    return {
        "mean_magnetization": float(mean_m),
        "var_magnetization": variance(magnetization_series, burn_in=burn_in),
        "dominance_sign": int(dominance_sign(mean_m, eps=eps)),
        "victory_indicator": int(victory_indicator(mean_m, eps=eps)),
        "time_to_first_sign_change": time_to_first_sign_change(
            magnetization_series, reference_sign=ref_sign, eps=eps
        ),
        "time_to_first_crossing_above_threshold": time_to_threshold_crossing(
            magnetization_series, threshold=threshold, direction="above", strict=True
        ),
        "fraction_time_above_threshold": fraction_time_above_threshold(
            magnetization_series, burn_in=burn_in, threshold=threshold, strict=True
        ),
        "fraction_time_below_threshold": fraction_time_below_threshold(
            magnetization_series, burn_in=burn_in, threshold=threshold, strict=True
        ),
    }


def free_node_flip_activity(
    trajectory: np.ndarray,
    zealot_mask: np.ndarray,
) -> dict[str, float | int]:
    """Estimate post-update flipping activity among free nodes."""
    traj = np.asarray(trajectory, dtype=np.int8)
    if traj.ndim != 2:
        raise ValueError("trajectory must be a 2D array of shape (T, N).")

    zealot_mask = np.asarray(zealot_mask, dtype=bool)
    if traj.shape[1] != zealot_mask.shape[0]:
        raise ValueError("trajectory and zealot_mask have inconsistent node dimensions.")

    free_mask = np.logical_not(zealot_mask)
    if traj.shape[0] < 2:
        free_count = int(np.sum(free_mask))
        return {
            "total_flips": 0,
            "mean_flips_per_free_node": 0.0,
            "active_free_fraction": 0.0 if free_count > 0 else 0.0,
        }

    flips = np.sum(traj[1:] != traj[:-1], axis=0)
    free_flips = flips[free_mask]
    if free_flips.size == 0:
        return {
            "total_flips": 0,
            "mean_flips_per_free_node": 0.0,
            "active_free_fraction": 0.0,
        }

    return {
        "total_flips": int(np.sum(free_flips)),
        "mean_flips_per_free_node": float(np.mean(free_flips)),
        "active_free_fraction": float(np.mean(free_flips > 0)),
    }

"""Utilities for estimating convergence/stopping times from magnetization trajectories."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from src.observables import magnetization
from src.utils import assign_zealots
from src.voter_model import step_voter_with_delta


def estimate_convergence_time(
    series,
    tol: float = 1e-12,
    min_plateau: int = 100,
) -> int | None:
    """Estimate the first time index where magnetization stops changing.

    We define convergence time as the earliest time t such that:
    - for all s >= t, |m(s) - m(T)| <= tol
    - and the terminal plateau has length at least `min_plateau`

    Returns
    -------
    int | None
        1-based stopping time in update steps, or None if not detected.
    """
    arr = np.asarray(series, dtype=float)

    if arr.ndim != 1:
        raise ValueError("series must be one-dimensional.")
    if arr.size == 0:
        return None
    if min_plateau <= 0:
        raise ValueError("min_plateau must be strictly positive.")
    if tol < 0.0:
        raise ValueError("tol must be non-negative.")

    n = arr.size
    if n < min_plateau:
        return None

    final_val = float(arr[-1])
    dev = np.abs(arr - final_val)
    suffix_max_dev = np.maximum.accumulate(dev[::-1])[::-1]

    max_start = n - min_plateau
    valid_idx = np.where(
        (suffix_max_dev <= tol)
        & (np.arange(n) <= max_start)
    )[0]

    if valid_idx.size == 0:
        return None

    # Convert to 1-based steps for easier interpretation in logs/plots.
    return int(valid_idx[0] + 1)


def summarize_stopping_times(stopping_times: list[int | None]) -> dict[str, Any]:
    """Summarize stopping-time samples (possibly containing missing values)."""
    valid = np.array([x for x in stopping_times if x is not None], dtype=float)
    n_total = len(stopping_times)
    n_conv = int(valid.size)

    summary: dict[str, Any] = {
        "n_total": int(n_total),
        "n_converged": n_conv,
        "converged_fraction": float(n_conv / n_total) if n_total > 0 else 0.0,
        "mean_stopping_time": None,
        "std_stopping_time": None,
        "median_stopping_time": None,
        "min_stopping_time": None,
        "max_stopping_time": None,
    }

    if valid.size == 0:
        return summary

    summary.update(
        {
            "mean_stopping_time": float(np.mean(valid)),
            "std_stopping_time": float(np.std(valid, ddof=1 if valid.size > 1 else 0)),
            "median_stopping_time": float(np.median(valid)),
            "min_stopping_time": float(np.min(valid)),
            "max_stopping_time": float(np.max(valid)),
        }
    )
    return summary


def run_until_convergence(
    G,
    rho: float,
    T_max: int,
    seed: int | None = None,
    tol: float = 1e-12,
    stable_window: int = 500,
    zealot_state: int = +1,
) -> dict[str, Any]:
    """Run dynamics until convergence or max horizon.

    Convergence criterion:
    Magnetization remains unchanged (within `tol`) for `stable_window` consecutive
    update steps.

    Returns
    -------
    dict
        {
            "converged": bool,
            "stopping_time": int | None,  # 1-based plateau start
            "steps_simulated": int,
            "final_magnetization": float,
            "seed": int | None,
        }
    """
    if T_max <= 0:
        raise ValueError("T_max must be strictly positive.")
    if stable_window <= 0:
        raise ValueError("stable_window must be strictly positive.")
    if tol < 0.0:
        raise ValueError("tol must be non-negative.")
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be in [0, 1].")

    if set(G.nodes()) != set(range(G.number_of_nodes())):
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    rng = np.random.default_rng(seed)
    zealot_seed = int(rng.integers(np.iinfo(np.int32).max))
    zealot_mask, states = assign_zealots(
        G,
        rho=rho,
        state=zealot_state,
        seed=zealot_seed,
    )

    m_t = float(magnetization(states))
    plateau_count = 0

    for t in range(1, T_max + 1):
        delta_m = float(step_voter_with_delta(G, states, zealot_mask, rng))
        m_t += delta_m

        if abs(delta_m) <= tol:
            plateau_count += 1
        else:
            plateau_count = 0

        if plateau_count >= stable_window:
            stopping_time = t - stable_window + 1
            return {
                "converged": True,
                "stopping_time": int(stopping_time),
                "steps_simulated": int(t),
                "final_magnetization": float(m_t),
                "seed": seed,
            }

    return {
        "converged": False,
        "stopping_time": None,
        "steps_simulated": int(T_max),
        "final_magnetization": float(m_t),
        "seed": seed,
    }

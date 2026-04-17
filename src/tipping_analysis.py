"""Experiment orchestration for two-camp zealot tipping analysis."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
from tqdm import tqdm

from src.convergence_time import estimate_convergence_time
from src.effective_strength import compute_effective_strength_candidates
from src.graph_features import (
    compute_graph_context,
    compute_two_camp_comparison_features,
)
from src.observables import asymptotic_observables, tipping_observables
from src.two_zealot_voter_model import run_two_zealot_simulation
from src.wl_features import compute_two_camp_wl_features

LOGGER = logging.getLogger(__name__)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _nanstd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    if finite.size <= 1:
        return float(0.0)
    return float(np.std(finite, ddof=1))


def _to_float_or_nan(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _aggregate_numeric_fields(records: list[dict], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        vals = [_to_float_or_nan(rec.get(key)) for rec in records]
        out[f"mean_{key}"] = _nanmean(vals)
        out[f"std_{key}"] = _nanstd(vals)
    return out


def run_two_camp_configuration(
    G: nx.Graph,
    n_pos: int,
    n_neg: int,
    T: int,
    burn_in: int,
    n_runs: int,
    seed: int | None = None,
    strategy_pos: str = "random",
    strategy_neg: str = "random",
    strategy_kwargs_pos: dict | None = None,
    strategy_kwargs_neg: dict | None = None,
    threshold: float = 0.0,
    wl_n_iter: int = 3,
    show_progress: bool = True,
    compute_flip_activity: bool = False,
    store_run_records: bool = True,
) -> dict:
    """Run repeated simulations for one (n_pos, n_neg, strategies) configuration."""
    if n_runs <= 0:
        raise ValueError("n_runs must be strictly positive.")

    strategy_kwargs_pos = dict(strategy_kwargs_pos or {})
    strategy_kwargs_neg = dict(strategy_kwargs_neg or {})

    context = compute_graph_context(
        G,
        wl_n_iter=wl_n_iter,
        include_betweenness=False,
        include_eigenvector=False,
        include_communities=True,
    )

    graph_features = context["graph_features"]
    n_nodes = int(graph_features["n_nodes"])

    rng = np.random.default_rng(seed)
    run_records: list[dict] = []

    iterator = tqdm(
        range(n_runs),
        desc=f"two-camp n+={n_pos} n-={n_neg}",
        disable=not show_progress,
        leave=False,
    )

    for rep in iterator:
        run_seed = int(rng.integers(np.iinfo(np.int32).max))

        sim = run_two_zealot_simulation(
            G=G,
            n_pos=int(n_pos),
            n_neg=int(n_neg),
            T=int(T),
            burn_in=int(burn_in),
            seed=run_seed,
            strategy_pos=strategy_pos,
            strategy_neg=strategy_neg,
            strategy_kwargs_pos=strategy_kwargs_pos,
            strategy_kwargs_neg=strategy_kwargs_neg,
            show_progress=False,
            record=False,
            compute_flip_activity=compute_flip_activity,
        )

        m = np.asarray(sim["magnetization"], dtype=float)
        p = np.asarray(sim["positive_fraction"], dtype=float)
        q = np.asarray(sim["negative_fraction"], dtype=float)

        asym = asymptotic_observables(
            magnetization_series=m,
            positive_series=p,
            negative_series=q,
            burn_in=int(burn_in),
        )
        tip = tipping_observables(
            magnetization_series=m,
            burn_in=int(burn_in),
            threshold=float(threshold),
        )

        min_plateau = max(10, int(0.02 * T))
        stabilization_time = estimate_convergence_time(m, tol=1e-12, min_plateau=min_plateau)

        pos_nodes = sim["pos_nodes"]
        neg_nodes = sim["neg_nodes"]

        wl_features = compute_two_camp_wl_features(
            G,
            pos_nodes=pos_nodes,
            neg_nodes=neg_nodes,
            n_iter=wl_n_iter,
            wl_labels=context["wl_labels"],
        )

        comparison_features = compute_two_camp_comparison_features(
            G,
            pos_nodes=pos_nodes,
            neg_nodes=neg_nodes,
            context=context,
        )

        strength = compute_effective_strength_candidates(
            G,
            pos_nodes=pos_nodes,
            neg_nodes=neg_nodes,
            feature_bundle={
                "graph_features": graph_features,
                "comparison_features": comparison_features,
                "wl_features": wl_features,
            },
        )

        run_record = {
            "replicate": int(rep),
            "seed": int(run_seed),
            "n_nodes": int(n_nodes),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "rho_pos": float(n_pos / n_nodes),
            "rho_neg": float(n_neg / n_nodes),
            "strategy_pos": str(strategy_pos),
            "strategy_neg": str(strategy_neg),
            "stabilization_time": stabilization_time,
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in asym.items()},
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in tip.items()},
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in comparison_features.items()},
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in wl_features.items() if not isinstance(v, dict)},
            **{k: float(v) for k, v in strength.items()},
            "pos_nodes": pos_nodes,
            "neg_nodes": neg_nodes,
        }

        flip_activity = sim.get("flip_activity")
        if isinstance(flip_activity, dict):
            for key, val in flip_activity.items():
                run_record[f"flip_{key}"] = float(val)

        run_records.append(run_record)

    n_pos_wins = int(np.sum([int(rec["victory_indicator"]) for rec in run_records]))
    pos_win_prob = float(n_pos_wins / n_runs)

    aggregate = {
        "n_runs": int(n_runs),
        "n_positive_wins": int(n_pos_wins),
        "positive_win_probability": float(pos_win_prob),
    }

    observable_keys = [
        "mean_magnetization",
        "var_magnetization",
        "mean_positive_fraction",
        "mean_negative_fraction",
        "time_to_first_sign_change",
        "time_to_first_crossing_above_threshold",
        "fraction_time_above_threshold",
        "fraction_time_below_threshold",
        "stabilization_time",
    ]
    aggregate.update(_aggregate_numeric_fields(run_records, observable_keys))

    strength_keys = sorted([k for k in run_records[0].keys() if k.startswith("delta_psi_") or k.startswith("pos_psi_") or k.startswith("neg_psi_")])
    aggregate.update(_aggregate_numeric_fields(run_records, strength_keys))

    feature_keys = [
        "diff_degree_sum",
        "diff_pagerank_sum",
        "diff_dispersion_score",
        "distance_advantage_pos",
        "wl_overlap_fraction_union",
        "diff_wl_coverage_fraction",
    ]
    aggregate.update(_aggregate_numeric_fields(run_records, feature_keys))

    out = {
        "config": {
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "rho_pos": float(n_pos / n_nodes),
            "rho_neg": float(n_neg / n_nodes),
            "T": int(T),
            "burn_in": int(burn_in),
            "n_runs": int(n_runs),
            "seed": seed,
            "strategy_pos": str(strategy_pos),
            "strategy_neg": str(strategy_neg),
            "strategy_kwargs_pos": strategy_kwargs_pos,
            "strategy_kwargs_neg": strategy_kwargs_neg,
            "threshold": float(threshold),
            "wl_n_iter": int(wl_n_iter),
        },
        "graph_features": {k: float(v) for k, v in graph_features.items()},
        "aggregate": aggregate,
    }

    if store_run_records:
        out["run_records"] = run_records

    return out


def run_tipping_grid(
    G: nx.Graph,
    n_pos_values: list[int],
    n_neg_values: list[int],
    T: int,
    burn_in: int,
    n_runs: int,
    seed: int | None = None,
    strategy_pos: str = "random",
    strategy_neg: str = "random",
    strategy_kwargs_pos: dict | None = None,
    strategy_kwargs_neg: dict | None = None,
    threshold: float = 0.0,
    wl_n_iter: int = 3,
    show_progress: bool = True,
    store_run_records: bool = False,
) -> dict:
    """Run a phase-diagram style grid over (n_pos, n_neg)."""
    pos_vals = [int(v) for v in n_pos_values]
    neg_vals = [int(v) for v in n_neg_values]
    if len(pos_vals) == 0 or len(neg_vals) == 0:
        raise ValueError("n_pos_values and n_neg_values must be non-empty.")

    mean_m_grid = np.empty((len(pos_vals), len(neg_vals)), dtype=float)
    pos_win_grid = np.empty((len(pos_vals), len(neg_vals)), dtype=float)

    records: list[dict] = []
    rng = np.random.default_rng(seed)

    outer = tqdm(pos_vals, desc="tipping grid n_pos", disable=not show_progress)
    for i, n_pos in enumerate(outer):
        inner = tqdm(neg_vals, desc=f"n_neg (n_pos={n_pos})", leave=False, disable=not show_progress)
        for j, n_neg in enumerate(inner):
            config_seed = int(rng.integers(np.iinfo(np.int32).max))
            res = run_two_camp_configuration(
                G=G,
                n_pos=int(n_pos),
                n_neg=int(n_neg),
                T=int(T),
                burn_in=int(burn_in),
                n_runs=int(n_runs),
                seed=config_seed,
                strategy_pos=strategy_pos,
                strategy_neg=strategy_neg,
                strategy_kwargs_pos=strategy_kwargs_pos,
                strategy_kwargs_neg=strategy_kwargs_neg,
                threshold=float(threshold),
                wl_n_iter=int(wl_n_iter),
                show_progress=False,
                store_run_records=store_run_records,
            )

            mean_m = float(res["aggregate"]["mean_mean_magnetization"])
            p_win = float(res["aggregate"]["positive_win_probability"])

            mean_m_grid[i, j] = mean_m
            pos_win_grid[i, j] = p_win

            row = {
                **res["config"],
                **res["aggregate"],
                **{f"graph_{k}": v for k, v in res["graph_features"].items()},
            }
            records.append(row)

    boundary_points = extract_phase_boundary(
        n_pos_values=pos_vals,
        n_neg_values=neg_vals,
        mean_m_grid=mean_m_grid,
        epsilon=0.05,
    )

    return {
        "n_pos_values": np.asarray(pos_vals, dtype=int),
        "n_neg_values": np.asarray(neg_vals, dtype=int),
        "mean_m_grid": mean_m_grid,
        "positive_win_probability_grid": pos_win_grid,
        "records": records,
        "boundary_points": boundary_points,
        "strategy_pos": str(strategy_pos),
        "strategy_neg": str(strategy_neg),
        "T": int(T),
        "burn_in": int(burn_in),
        "n_runs": int(n_runs),
    }


def extract_phase_boundary(
    n_pos_values: list[int],
    n_neg_values: list[int],
    mean_m_grid: np.ndarray,
    epsilon: float = 0.05,
) -> list[dict[str, float]]:
    """Extract approximate boundary points where mean magnetization is near zero."""
    out: list[dict[str, float]] = []
    arr = np.asarray(mean_m_grid, dtype=float)
    for i, n_pos in enumerate(n_pos_values):
        for j, n_neg in enumerate(n_neg_values):
            if abs(float(arr[i, j])) <= float(epsilon):
                out.append(
                    {
                        "n_pos": float(n_pos),
                        "n_neg": float(n_neg),
                        "mean_m": float(arr[i, j]),
                    }
                )
    return out


def run_minimal_sanity_checks(seed: int = 123) -> dict:
    """Minimal sanity checks required by the two-camp extension brief."""
    rng = np.random.default_rng(seed)

    # Check 1: symmetry under equal camp sizes and random placement on complete graph.
    G_sym = nx.complete_graph(60)
    sym = run_two_camp_configuration(
        G=G_sym,
        n_pos=6,
        n_neg=6,
        T=6000,
        burn_in=1000,
        n_runs=20,
        seed=int(rng.integers(np.iinfo(np.int32).max)),
        strategy_pos="random",
        strategy_neg="random",
        show_progress=False,
        store_run_records=False,
    )

    # Check 2: positive hub placement should often outperform random placement.
    G_adv = nx.barabasi_albert_graph(150, 3, seed=int(rng.integers(np.iinfo(np.int32).max)))
    base = run_two_camp_configuration(
        G=G_adv,
        n_pos=8,
        n_neg=10,
        T=8000,
        burn_in=1500,
        n_runs=16,
        seed=int(rng.integers(np.iinfo(np.int32).max)),
        strategy_pos="random",
        strategy_neg="random",
        show_progress=False,
        store_run_records=False,
    )
    hub = run_two_camp_configuration(
        G=G_adv,
        n_pos=8,
        n_neg=10,
        T=8000,
        burn_in=1500,
        n_runs=16,
        seed=int(rng.integers(np.iinfo(np.int32).max)),
        strategy_pos="highest_degree",
        strategy_neg="random",
        show_progress=False,
        store_run_records=False,
    )

    return {
        "equal_size_random_balance": {
            "positive_win_probability": float(sym["aggregate"]["positive_win_probability"]),
            "mean_mean_magnetization": float(sym["aggregate"]["mean_mean_magnetization"]),
        },
        "hub_advantage_check": {
            "baseline_positive_win_probability": float(base["aggregate"]["positive_win_probability"]),
            "hub_positive_win_probability": float(hub["aggregate"]["positive_win_probability"]),
        },
    }

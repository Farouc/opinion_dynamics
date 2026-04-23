"""Shared utilities for two-camp experiment scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path("/tmp/matplotlib")
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

sns.set_style("whitegrid")

PALETTE = {
    "random": "#7B8EC8",
    "highest_degree": "#E05C5C",
    "hub_then_spread": "#E8A838",
    "wl_cover": "#5BAD8F",
    "farthest_spread": "#9B59B6",
    "wl_top_class": "#2ECC71",
    "highest_eigenvector": "#4E79A7",
    "highest_betweenness": "#F28E2B",
    "highest_pagerank": "#76B7B2",
}

GRAPH_COLORS = {
    "fully_connected": "#34495E",
    "erdos_renyi": "#2980B9",
    "barabasi_albert": "#C0392B",
    "grid_lattice": "#16A085",
}


def setup_output_dirs(experiment_name: str) -> dict[str, Path]:
    """Creates results/{name}/raw/ and results/{name}/figures/, returns paths dict."""
    base = ROOT / "results" / str(experiment_name)
    raw = base / "raw"
    figures = base / "figures"
    raw.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return {"base": base, "raw": raw, "figures": figures}


def save_figure(fig, path: str | Path, dpi: int = 300):
    """Saves figure with tight layout and prints confirmation."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    print(f"[saved figure] {out}")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def graph_label(graph_type: str) -> str:
    key = str(graph_type).lower()
    mapping = {
        "fully_connected": "FC",
        "erdos_renyi": "ER",
        "barabasi_albert": "BA",
        "grid_lattice": "GRID",
    }
    return mapping.get(key, key.upper())


def slug_float(x: float, ndigits: int = 4) -> str:
    return f"{float(x):.{ndigits}f}".replace(".", "")


def sanitize_token(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


def make_graph(graph_type: str | None = None, **kwargs) -> nx.Graph:
    """Factory for reproducible graph generation."""
    from src.graph_generation import (
        generate_barabasi_albert,
        generate_erdos_renyi,
        generate_fully_connected,
        generate_grid_lattice,
    )

    if graph_type is None:
        if "graph_type" not in kwargs:
            raise ValueError("make_graph requires graph_type either as argument or keyword.")
        graph_type = kwargs.pop("graph_type")
    else:
        kwargs.pop("graph_type", None)

    gtype = str(graph_type).strip().lower()
    seed = kwargs.get("seed")

    if gtype == "fully_connected":
        return generate_fully_connected(n=int(kwargs["n"]))
    if gtype == "erdos_renyi":
        return generate_erdos_renyi(n=int(kwargs["n"]), p=float(kwargs["p"]), seed=seed)
    if gtype == "barabasi_albert":
        return generate_barabasi_albert(n=int(kwargs["n"]), m=int(kwargs["m"]), seed=seed)
    if gtype == "grid_lattice":
        return generate_grid_lattice(L=int(kwargs["L"]))

    raise ValueError(f"Unsupported graph_type={graph_type}")


def run_config_multirun(
    graph,
    pos_nodes,
    neg_nodes,
    T,
    burn_in,
    n_runs,
    base_seed,
):
    """Runs n_runs simulations, returns dict with mean_m, std_m, P_plus_win, trajectories."""
    from src.observables import time_average
    from src.two_zealot_voter_model import run_two_zealot_simulation

    pos_nodes = [int(x) for x in pos_nodes]
    neg_nodes = [int(x) for x in neg_nodes]

    rng = np.random.default_rng(base_seed)

    magnetization = np.empty((int(n_runs), int(T)), dtype=float)
    positive_fraction = np.empty((int(n_runs), int(T)), dtype=float)
    mean_m_runs = np.empty(int(n_runs), dtype=float)

    for r in range(int(n_runs)):
        seed_r = int(rng.integers(np.iinfo(np.int32).max))
        sim = run_two_zealot_simulation(
            G=graph,
            n_pos=len(pos_nodes),
            n_neg=len(neg_nodes),
            T=int(T),
            burn_in=int(burn_in),
            seed=seed_r,
            pos_nodes=pos_nodes,
            neg_nodes=neg_nodes,
            show_progress=False,
            record=False,
        )
        magnetization[r] = np.asarray(sim["magnetization"], dtype=float)
        positive_fraction[r] = np.asarray(sim["positive_fraction"], dtype=float)
        mean_m_runs[r] = float(time_average(sim["magnetization"], burn_in=int(burn_in)))

    return {
        "magnetization": magnetization,
        "positive_fraction": positive_fraction,
        "mean_m_runs": mean_m_runs,
        "mean_m": float(np.mean(mean_m_runs)),
        "std_m": float(np.std(mean_m_runs, ddof=1 if int(n_runs) > 1 else 0)),
        "P_plus_win": float(np.mean(mean_m_runs > 0.0)),
    }


def run_strategy_multirun(
    G: nx.Graph,
    n_pos: int,
    n_neg: int,
    strategy_pos: str,
    strategy_neg: str,
    T: int,
    burn_in: int,
    n_runs: int,
    base_seed: int,
    threshold: float = 0.0,
    record: bool = False,
) -> dict[str, Any]:
    """Run repeated simulations for one strategy pair and return trajectories + summaries."""
    from src.observables import time_average
    from src.two_zealot_voter_model import run_two_zealot_simulation

    rng = np.random.default_rng(base_seed)

    magnetization = np.empty((int(n_runs), int(T)), dtype=float)
    positive_fraction = np.empty((int(n_runs), int(T)), dtype=float)
    mean_m_runs = np.empty(int(n_runs), dtype=float)
    crossing_times = np.empty(int(n_runs), dtype=float)

    sample_run: dict[str, Any] | None = None

    for r in range(int(n_runs)):
        seed_r = int(rng.integers(np.iinfo(np.int32).max))
        sim = run_two_zealot_simulation(
            G=G,
            n_pos=int(n_pos),
            n_neg=int(n_neg),
            T=int(T),
            burn_in=int(burn_in),
            seed=seed_r,
            strategy_pos=str(strategy_pos),
            strategy_neg=str(strategy_neg),
            show_progress=False,
            record=bool(record),
        )
        if sample_run is None:
            sample_run = sim

        m = np.asarray(sim["magnetization"], dtype=float)
        p = np.asarray(sim["positive_fraction"], dtype=float)

        magnetization[r] = m
        positive_fraction[r] = p
        mean_m_runs[r] = float(time_average(m, burn_in=int(burn_in)))

        idx = np.where(m > float(threshold))[0]
        crossing_times[r] = float(idx[0] + 1) if idx.size > 0 else np.nan

    return {
        "magnetization": magnetization,
        "positive_fraction": positive_fraction,
        "mean_m_runs": mean_m_runs,
        "crossing_times": crossing_times,
        "mean_m": float(np.mean(mean_m_runs)),
        "std_m": float(np.std(mean_m_runs, ddof=1 if int(n_runs) > 1 else 0)),
        "P_plus_win": float(np.mean(mean_m_runs > 0.0)),
        "mean_crossing_time": float(np.nanmean(crossing_times)) if np.any(np.isfinite(crossing_times)) else float("nan"),
        "sample_run": sample_run,
    }


def draw_network_opinion(
    G,
    zealot_plus,
    zealot_minus,
    state,
    ax,
    title="",
    seed=0,
    pos=None,
):
    """NetworkX visualization: red=Z+, blue=Z-, lightcoral=free+1, lightblue=free-1.
    Node size proportional to degree. Draws on provided matplotlib Axes.
    """
    z_pos = set(int(x) for x in zealot_plus)
    z_neg = set(int(x) for x in zealot_minus)
    st = np.asarray(state, dtype=int)

    if pos is None:
        pos = nx.spring_layout(G, seed=int(seed))

    deg = dict(G.degree())
    max_deg = max(deg.values()) if len(deg) > 0 else 1

    node_colors = []
    node_sizes = []
    for node in G.nodes():
        node_i = int(node)
        if node_i in z_pos:
            color = "red"
        elif node_i in z_neg:
            color = "blue"
        else:
            color = "lightcoral" if int(st[node_i]) == 1 else "lightblue"
        node_colors.append(color)
        node_sizes.append(80 + 320 * (float(deg[node_i]) / max_deg if max_deg > 0 else 0.0))

    nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.25, width=0.7, edge_color="#999999")
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.3,
        edgecolors="#222222",
    )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def summarize_to_stdout(experiment_name: str, payload: dict[str, Any]) -> None:
    """Print short standardized run summary."""
    print(f"\n[{experiment_name}] summary")
    for key, val in payload.items():
        print(f"- {key}: {val}")

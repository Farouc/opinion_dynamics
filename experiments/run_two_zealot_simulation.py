"""Run a single two-camp zealot voter simulation and save outputs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path("/tmp/matplotlib")
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph_generation import create_graph_from_config
from src.observables import asymptotic_observables, tipping_observables
from src.two_zealot_voter_model import run_two_zealot_simulation
from src.utils import ensure_dir, load_config, setup_logging

LOGGER = logging.getLogger(__name__)


def _merge_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    merged = dict(cfg)
    for key in ("graph_type", "n", "p", "m", "L", "T", "burn_in", "seed"):
        val = getattr(args, key, None)
        if val is not None:
            merged[key] = val
    return merged


def _log_graph_details(G: nx.Graph, graph_type: str) -> None:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    avg_degree = (2.0 * e / n) if n > 0 else 0.0
    density = nx.density(G) if n > 1 else 0.0
    n_components = nx.number_connected_components(G) if n > 0 else 0

    LOGGER.info("Graph details:")
    LOGGER.info("  graph_type=%s", graph_type)
    LOGGER.info("  n_nodes=%d", n)
    LOGGER.info("  n_edges=%d", e)
    LOGGER.info("  average_degree=%.4f", avg_degree)
    LOGGER.info("  density=%.6f", density)
    LOGGER.info("  connected_components=%d", n_components)


def _parse_json_dict(raw: str | None) -> dict:
    if raw is None or str(raw).strip() == "":
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Strategy kwargs must be a JSON object.")
    return parsed


def _parse_node_list(raw: str | None) -> list[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _resolve_count(n_total: int, count: int | None, rho: float | None, name: str) -> int:
    if count is not None:
        out = int(count)
    elif rho is not None:
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"{name} rho must be in [0,1].")
        out = int(round(float(rho) * n_total))
    else:
        raise ValueError(f"Specify either --{name}-count or --{name}-rho.")

    if out < 0:
        raise ValueError(f"{name} count must be non-negative.")
    if out > n_total:
        raise ValueError(f"{name} count cannot exceed graph size ({n_total}).")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")

    parser.add_argument(
        "--graph-type",
        type=str,
        choices=["fully_connected", "erdos_renyi", "barabasi_albert", "grid_lattice"],
        default=None,
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--p", type=float, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--L", type=int, default=None)

    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--pos-count", type=int, default=None)
    parser.add_argument("--neg-count", type=int, default=None)
    parser.add_argument("--pos-rho", type=float, default=None)
    parser.add_argument("--neg-rho", type=float, default=None)

    parser.add_argument("--strategy-pos", type=str, default="random")
    parser.add_argument("--strategy-neg", type=str, default="random")
    parser.add_argument("--strategy-kwargs-pos", type=str, default=None)
    parser.add_argument("--strategy-kwargs-neg", type=str, default=None)

    parser.add_argument("--pos-nodes", type=str, default=None)
    parser.add_argument("--neg-nodes", type=str, default=None)

    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--print-graph-info", action="store_true")
    parser.add_argument("--print-magnetization", action="store_true")
    parser.add_argument("--magnetization-interval", type=int, default=1000)
    parser.add_argument("--compute-flip-activity", action="store_true")

    parser.add_argument(
        "--save-figure",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    cfg = _merge_cli_overrides(load_config(args.config), args)
    graph_seed = int(cfg.get("seed", 0))
    G = create_graph_from_config(cfg, seed=graph_seed)

    if args.print_graph_info:
        _log_graph_details(G, str(cfg.get("graph_type", "erdos_renyi")))

    n_total = G.number_of_nodes()
    n_pos = _resolve_count(n_total, args.pos_count, args.pos_rho, name="pos")
    n_neg = _resolve_count(n_total, args.neg_count, args.neg_rho, name="neg")
    if n_pos + n_neg > n_total:
        raise ValueError("pos + neg zealot counts exceed number of nodes.")

    strategy_kwargs_pos = _parse_json_dict(args.strategy_kwargs_pos)
    strategy_kwargs_neg = _parse_json_dict(args.strategy_kwargs_neg)

    pos_nodes = _parse_node_list(args.pos_nodes)
    neg_nodes = _parse_node_list(args.neg_nodes)

    sim = run_two_zealot_simulation(
        G=G,
        n_pos=n_pos,
        n_neg=n_neg,
        T=int(cfg["T"]),
        burn_in=int(cfg["burn_in"]),
        seed=int(cfg.get("seed", 0)),
        strategy_pos=str(args.strategy_pos),
        strategy_neg=str(args.strategy_neg),
        strategy_kwargs_pos=strategy_kwargs_pos,
        strategy_kwargs_neg=strategy_kwargs_neg,
        pos_nodes=pos_nodes,
        neg_nodes=neg_nodes,
        record=bool(args.record),
        show_progress=not args.no_progress,
        print_magnetization=bool(args.print_magnetization),
        magnetization_interval=int(args.magnetization_interval),
        compute_flip_activity=bool(args.compute_flip_activity),
    )

    asym = asymptotic_observables(
        sim["magnetization"],
        burn_in=int(cfg["burn_in"]),
        positive_series=sim["positive_fraction"],
        negative_series=sim["negative_fraction"],
    )
    tip = tipping_observables(
        sim["magnetization"],
        burn_in=int(cfg["burn_in"]),
        threshold=float(args.threshold),
    )

    raw_dir = ensure_dir(Path(args.results_dir) / "raw")
    fig_dir = ensure_dir(Path(args.results_dir) / "figures")

    tag = (
        f"npos_{n_pos}_nneg_{n_neg}_"
        f"spos_{args.strategy_pos}_sneg_{args.strategy_neg}_seed_{int(cfg.get('seed', 0))}"
    )

    npz_path = raw_dir / f"two_camp_simulation_{tag}.npz"
    payload = {
        "magnetization": np.asarray(sim["magnetization"], dtype=float),
        "positive_fraction": np.asarray(sim["positive_fraction"], dtype=float),
        "negative_fraction": np.asarray(sim["negative_fraction"], dtype=float),
        "pos_zealot_mask": np.asarray(sim["pos_zealot_mask"], dtype=bool),
        "neg_zealot_mask": np.asarray(sim["neg_zealot_mask"], dtype=bool),
        "initial_states": np.asarray(sim["initial_states"], dtype=np.int8),
        "final_states": np.asarray(sim["final_states"], dtype=np.int8),
        "n_pos": np.array([sim["n_pos"]]),
        "n_neg": np.array([sim["n_neg"]]),
        "rho_pos": np.array([sim["rho_pos"]]),
        "rho_neg": np.array([sim["rho_neg"]]),
    }
    if sim["trajectory"] is not None:
        payload["trajectory"] = np.asarray(sim["trajectory"], dtype=np.int8)

    np.savez(npz_path, **payload)
    LOGGER.info("Saved simulation arrays to %s", npz_path)

    summary = {
        "config": {
            "graph_type": str(cfg.get("graph_type", "erdos_renyi")),
            "n_nodes": int(n_total),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "rho_pos": float(n_pos / n_total),
            "rho_neg": float(n_neg / n_total),
            "strategy_pos": str(args.strategy_pos),
            "strategy_neg": str(args.strategy_neg),
            "strategy_kwargs_pos": strategy_kwargs_pos,
            "strategy_kwargs_neg": strategy_kwargs_neg,
            "T": int(cfg["T"]),
            "burn_in": int(cfg["burn_in"]),
            "seed": int(cfg.get("seed", 0)),
            "threshold": float(args.threshold),
        },
        "asymptotic_observables": asym,
        "tipping_observables": tip,
        "flip_activity": sim.get("flip_activity"),
        "pos_nodes": sim["pos_nodes"],
        "neg_nodes": sim["neg_nodes"],
    }

    json_path = raw_dir / f"two_camp_summary_{tag}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    LOGGER.info("Saved simulation summary to %s", json_path)

    if args.save_figure:
        t = np.arange(1, int(cfg["T"]) + 1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.0), sharex=True)

        ax1.plot(t, sim["magnetization"], linewidth=1.1, label="m(t)")
        ax1.axhline(float(args.threshold), linestyle="--", linewidth=1.0, alpha=0.8, label="threshold")
        ax1.axvline(int(cfg["burn_in"]), linestyle=":", linewidth=1.0, alpha=0.8, label="burn-in")
        ax1.set_ylabel("magnetization")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="best")

        ax2.plot(t, sim["positive_fraction"], linewidth=1.1, label="positive fraction")
        ax2.plot(t, sim["negative_fraction"], linewidth=1.1, label="negative fraction")
        ax2.axvline(int(cfg["burn_in"]), linestyle=":", linewidth=1.0, alpha=0.8)
        ax2.set_xlabel("time step")
        ax2.set_ylabel("fraction")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="best")

        fig.suptitle(
            f"Two-camp zealot dynamics (n+={n_pos}, n-={n_neg}, "
            f"s+={args.strategy_pos}, s-={args.strategy_neg})"
        )
        fig.tight_layout()

        fig_path = fig_dir / f"two_camp_timeseries_{tag}.png"
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        LOGGER.info("Saved time-series figure to %s", fig_path)


if __name__ == "__main__":
    main()

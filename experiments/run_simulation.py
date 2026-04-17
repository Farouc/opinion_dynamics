"""Run a single zealot voter simulation and plot m(t)."""

from __future__ import annotations

import argparse
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
from src.simulation import run_simulation
from src.utils import ensure_dir, load_config, parse_rho_values, setup_logging


LOGGER = logging.getLogger(__name__)


def _merge_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply optional CLI overrides on top of config values."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        choices=["fully_connected", "erdos_renyi", "barabasi_albert", "grid_lattice"],
        default=None,
        help="Override graph type from config.",
    )
    parser.add_argument("--n", type=int, default=None, help="Override number of nodes.")
    parser.add_argument(
        "--p",
        type=float,
        default=None,
        help="Override Erdos-Renyi edge probability p.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="Override Barabasi-Albert attachment parameter m.",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=None,
        help="Override grid side length L (graph has L*L nodes).",
    )
    parser.add_argument("--T", type=int, default=None, help="Override number of time steps.")
    parser.add_argument(
        "--burn-in",
        type=int,
        default=None,
        help="Override burn-in value saved with outputs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Zealot fraction for this run (defaults to first value in rho_values).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record full state trajectory (can be memory-heavy).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--print-graph-info",
        action="store_true",
        help="Print graph statistics before simulation starts.",
    )
    parser.add_argument(
        "--print-magnetization",
        action="store_true",
        help="Print m(t) periodically during simulation.",
    )
    parser.add_argument(
        "--magnetization-interval",
        type=int,
        default=1000,
        help="Interval in steps for printing magnetization when --print-magnetization is enabled.",
    )
    parser.add_argument(
        "--save-figure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save magnetization figure to results/figures (default: true).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Base directory where raw/ and figures/ outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    cfg = _merge_cli_overrides(load_config(args.config), args)
    rho_values = parse_rho_values(cfg["rho_values"])
    rho = float(args.rho) if args.rho is not None else float(rho_values[0])

    seed = int(cfg.get("seed", 0))
    G = create_graph_from_config(cfg, seed=seed)

    if args.print_graph_info:
        _log_graph_details(G, str(cfg.get("graph_type", "erdos_renyi")))

    sim = run_simulation(
        G=G,
        rho=rho,
        T=int(cfg["T"]),
        burn_in=int(cfg["burn_in"]),
        seed=seed,
        record=args.record,
        show_progress=not args.no_progress,
        print_magnetization=args.print_magnetization,
        magnetization_interval=int(args.magnetization_interval),
    )

    raw_dir = ensure_dir(Path(args.results_dir) / "raw")
    fig_dir = ensure_dir(Path(args.results_dir) / "figures")

    rho_tag = f"{rho:.3f}".replace(".", "p")
    npz_path = raw_dir / f"simulation_rho_{rho_tag}.npz"

    to_save = {
        "magnetization": sim["magnetization"],
        "zealot_mask": sim["zealot_mask"],
        "initial_states": sim["initial_states"],
        "final_states": sim["final_states"],
        "rho": np.array([sim["rho"]]),
        "T": np.array([sim["T"]]),
        "burn_in": np.array([sim["burn_in"]]),
        "seed": np.array([sim["seed"] if sim["seed"] is not None else -1]),
    }
    if sim["trajectory"] is not None:
        to_save["trajectory"] = sim["trajectory"]

    np.savez(npz_path, **to_save)
    LOGGER.info("Saved simulation data to %s", npz_path)

    if args.save_figure:
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        ax.plot(sim["magnetization"], linewidth=1.0)
        ax.set_xlabel("time step")
        ax.set_ylabel("m(t)")
        ax.set_title(f"Zealot Voter Model: magnetization trajectory (rho={rho:.3f})")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        fig_path = fig_dir / f"magnetization_time_series_rho_{rho_tag}.png"
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        LOGGER.info("Saved time-series figure to %s", fig_path)
    else:
        LOGGER.info("Skipping figure save (--no-save-figure).")


if __name__ == "__main__":
    main()

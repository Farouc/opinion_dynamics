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
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph_generation import create_graph_from_config
from src.simulation import run_simulation
from src.utils import ensure_dir, load_config, parse_rho_values, setup_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to YAML configuration file.",
    )
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
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Base directory where raw/ and figures/ outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    cfg = load_config(args.config)
    rho_values = parse_rho_values(cfg["rho_values"])
    rho = float(args.rho) if args.rho is not None else float(rho_values[0])

    seed = int(cfg.get("seed", 0))
    G = create_graph_from_config(cfg, seed=seed)

    sim = run_simulation(
        G=G,
        rho=rho,
        T=int(cfg["T"]),
        burn_in=int(cfg["burn_in"]),
        seed=seed,
        record=args.record,
        show_progress=not args.no_progress,
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


if __name__ == "__main__":
    main()

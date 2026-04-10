"""Run EVT analysis over zealot densities and plot xi(rho)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evt import evt_analysis, plot_evt_results
from src.graph_generation import create_graph_from_config
from src.utils import ensure_dir, load_config, parse_rho_values, save_dict_csv, setup_logging

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

    seed = int(cfg.get("seed", 0))
    G = create_graph_from_config(cfg, seed=seed)

    results = evt_analysis(
        G=G,
        rho_values=rho_values,
        T=int(cfg["T"]),
        block_size=int(cfg["block_size"]),
        burn_in=int(cfg.get("burn_in", 0)),
        seed=seed,
        show_progress=not args.no_progress,
    )

    raw_dir = ensure_dir(Path(args.results_dir) / "raw")
    fig_dir = ensure_dir(Path(args.results_dir) / "figures")

    npz_path = raw_dir / "evt_results.npz"
    np.savez(npz_path, **results)
    LOGGER.info("Saved EVT arrays to %s", npz_path)

    csv_path = raw_dir / "evt_results.csv"
    save_dict_csv(csv_path, results, keys=["rho_values", "xi", "mu", "sigma", "n_blocks"])
    LOGGER.info("Saved EVT summary CSV to %s", csv_path)

    plot_evt_results(results, output_dir=fig_dir)


if __name__ == "__main__":
    main()

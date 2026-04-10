"""Run phase-transition sweep over zealot density rho."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph_generation import create_graph_from_config
from src.phase_transition import estimate_critical_rho, plot_phase_transition, sweep_rho
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    cfg = load_config(args.config)
    rho_values = parse_rho_values(cfg["rho_values"])

    seed = int(cfg.get("seed", 0))
    G = create_graph_from_config(cfg, seed=seed)

    results = sweep_rho(
        G=G,
        rho_values=rho_values,
        T=int(cfg["T"]),
        burn_in=int(cfg["burn_in"]),
        n_runs=int(cfg["n_runs"]),
        seed=seed,
        show_progress=not args.no_progress,
    )
    critical = estimate_critical_rho(results)

    raw_dir = ensure_dir(ROOT / "results" / "raw")
    fig_dir = ensure_dir(ROOT / "results" / "figures")

    npz_path = raw_dir / "phase_transition_results.npz"
    np.savez(npz_path, **results, **critical)
    LOGGER.info("Saved phase-transition arrays to %s", npz_path)

    csv_data = {
        "rho": results["rho_values"],
        "mean_m": results["mean_m"],
        "var_m": results["var_m"],
        "mean_m_sem": results["mean_m_sem"],
        "var_m_sem": results["var_m_sem"],
    }
    csv_path = raw_dir / "phase_transition_results.csv"
    save_dict_csv(csv_path, csv_data, keys=["rho", "mean_m", "var_m", "mean_m_sem", "var_m_sem"])
    LOGGER.info("Saved phase-transition summary CSV to %s", csv_path)

    crit_path = raw_dir / "critical_rho_estimate.json"
    with crit_path.open("w", encoding="utf-8") as f:
        json.dump(critical, f, indent=2)
    LOGGER.info("Saved critical rho estimate to %s", crit_path)

    plot_phase_transition(results, output_dir=fig_dir)

    LOGGER.info("Estimated rho_c = %.6f", critical["rho_c_estimate"])


if __name__ == "__main__":
    main()

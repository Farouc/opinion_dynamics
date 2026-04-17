"""Run a phase-diagram style (n_pos, n_neg) tipping grid for two zealot camps."""

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
from src.symbolic_features_dataset import save_dataset_rows
from src.tipping_analysis import run_tipping_grid
from src.utils import ensure_dir, load_config, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("Expected at least one integer value.")
    return sorted(set(vals))


def parse_float_list(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("Expected at least one float value.")
    return sorted(set(vals))


def _merge_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    merged = dict(cfg)
    for key in ("graph_type", "n", "p", "m", "L", "T", "burn_in", "seed"):
        val = getattr(args, key, None)
        if val is not None:
            merged[key] = val
    return merged


def _parse_json_dict(raw: str | None) -> dict:
    if raw is None or str(raw).strip() == "":
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Strategy kwargs must be a JSON object.")
    return parsed


def _resolve_count_grid(
    n_total: int,
    n_values_raw: str | None,
    rho_values_raw: str | None,
    label: str,
) -> list[int]:
    if rho_values_raw is not None and str(rho_values_raw).strip() != "":
        rho_vals = parse_float_list(rho_values_raw)
        out = [int(round(float(r) * n_total)) for r in rho_vals]
    elif n_values_raw is not None and str(n_values_raw).strip() != "":
        out = parse_int_list(n_values_raw)
    else:
        raise ValueError(f"Specify either --{label}-values or --rho-{label}-values.")

    if any(v < 0 for v in out):
        raise ValueError(f"All {label} values must be non-negative.")
    if any(v > n_total for v in out):
        raise ValueError(f"Some {label} values exceed graph size {n_total}.")

    return sorted(set(out))


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

    parser.add_argument("--n-pos-values", type=str, default="5,10,15,20")
    parser.add_argument("--n-neg-values", type=str, default="5,10,15,20")
    parser.add_argument("--rho-pos-values", type=str, default=None)
    parser.add_argument("--rho-neg-values", type=str, default=None)

    parser.add_argument("--strategy-pos", type=str, default="random")
    parser.add_argument("--strategy-neg", type=str, default="random")
    parser.add_argument("--strategy-kwargs-pos", type=str, default=None)
    parser.add_argument("--strategy-kwargs-neg", type=str, default=None)

    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--wl-n-iter", type=int, default=3)

    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--store-run-records", action="store_true")

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
    graph_seed = int(cfg.get("seed", 0))
    G = create_graph_from_config(cfg, seed=graph_seed)
    n_total = G.number_of_nodes()

    n_pos_values = _resolve_count_grid(
        n_total=n_total,
        n_values_raw=args.n_pos_values,
        rho_values_raw=args.rho_pos_values,
        label="pos",
    )
    n_neg_values = _resolve_count_grid(
        n_total=n_total,
        n_values_raw=args.n_neg_values,
        rho_values_raw=args.rho_neg_values,
        label="neg",
    )

    strategy_kwargs_pos = _parse_json_dict(args.strategy_kwargs_pos)
    strategy_kwargs_neg = _parse_json_dict(args.strategy_kwargs_neg)

    LOGGER.info("Graph type: %s", cfg.get("graph_type", "erdos_renyi"))
    LOGGER.info("n_pos grid: %s", n_pos_values)
    LOGGER.info("n_neg grid: %s", n_neg_values)
    LOGGER.info("Strategies: pos=%s, neg=%s", args.strategy_pos, args.strategy_neg)

    results = run_tipping_grid(
        G=G,
        n_pos_values=n_pos_values,
        n_neg_values=n_neg_values,
        T=int(cfg["T"]),
        burn_in=int(cfg["burn_in"]),
        n_runs=int(args.n_runs),
        seed=int(cfg.get("seed", 0)),
        strategy_pos=str(args.strategy_pos),
        strategy_neg=str(args.strategy_neg),
        strategy_kwargs_pos=strategy_kwargs_pos,
        strategy_kwargs_neg=strategy_kwargs_neg,
        threshold=float(args.threshold),
        wl_n_iter=int(args.wl_n_iter),
        show_progress=not args.no_progress,
        store_run_records=bool(args.store_run_records),
    )

    raw_dir = ensure_dir(Path(args.results_dir) / "raw")

    npz_path = raw_dir / "tipping_grid_results.npz"
    np.savez(
        npz_path,
        n_pos_values=np.asarray(results["n_pos_values"], dtype=int),
        n_neg_values=np.asarray(results["n_neg_values"], dtype=int),
        mean_m_grid=np.asarray(results["mean_m_grid"], dtype=float),
        positive_win_probability_grid=np.asarray(results["positive_win_probability_grid"], dtype=float),
        T=np.array([int(results["T"])]),
        burn_in=np.array([int(results["burn_in"])]),
        n_runs=np.array([int(results["n_runs"])]),
    )
    LOGGER.info("Saved tipping-grid arrays to %s", npz_path)

    records_json = raw_dir / "tipping_grid_records.json"
    records_csv = raw_dir / "tipping_grid_records.csv"
    save_dataset_rows(results["records"], json_path=records_json, csv_path=records_csv)
    LOGGER.info("Saved tipping-grid records to %s and %s", records_json, records_csv)

    boundary_path = raw_dir / "tipping_grid_boundary.json"
    with boundary_path.open("w", encoding="utf-8") as f:
        json.dump({"boundary_points": results["boundary_points"]}, f, indent=2)
    LOGGER.info("Saved phase-boundary points to %s", boundary_path)


if __name__ == "__main__":
    main()

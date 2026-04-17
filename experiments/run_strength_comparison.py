"""Compare placement strategies and effective-strength candidates for two camps."""

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
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph_generation import create_graph_from_config
from src.symbolic_features_dataset import configuration_result_to_dataset_row, save_dataset_rows
from src.tipping_analysis import run_two_camp_configuration
from src.utils import ensure_dir, load_config, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("Expected at least one integer value.")
    return sorted(set(vals))


def parse_str_list(raw: str) -> list[str]:
    vals = [str(x).strip() for x in str(raw).split(",") if str(x).strip()]
    if len(vals) == 0:
        raise ValueError("Expected at least one string value.")
    return vals


def _merge_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    merged = dict(cfg)
    for key in ("graph_type", "n", "p", "m", "L", "T", "burn_in", "seed"):
        val = getattr(args, key, None)
        if val is not None:
            merged[key] = val
    return merged


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

    parser.add_argument("--n-pos-values", type=str, default="8")
    parser.add_argument("--n-neg-values", type=str, default="10,12")

    parser.add_argument(
        "--strategy-pos-list",
        type=str,
        default="random,highest_degree,highest_eigenvector,farthest_spread,hub_then_spread,wl_cover",
    )
    parser.add_argument(
        "--strategy-neg-list",
        type=str,
        default="random,highest_degree",
    )

    parser.add_argument("--n-runs", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--wl-n-iter", type=int, default=3)

    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser.parse_args()


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(spearmanr(x, y, nan_policy="omit").correlation)


def main() -> None:
    args = parse_args()
    setup_logging()

    cfg = _merge_cli_overrides(load_config(args.config), args)
    G = create_graph_from_config(cfg, seed=int(cfg.get("seed", 0)))

    n_pos_values = parse_int_list(args.n_pos_values)
    n_neg_values = parse_int_list(args.n_neg_values)
    strategy_pos_list = parse_str_list(args.strategy_pos_list)
    strategy_neg_list = parse_str_list(args.strategy_neg_list)

    jobs: list[tuple[int, int, str, str]] = []
    for n_pos in n_pos_values:
        for n_neg in n_neg_values:
            for spos in strategy_pos_list:
                for sneg in strategy_neg_list:
                    jobs.append((n_pos, n_neg, spos, sneg))

    rows: list[dict] = []
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    iterator = tqdm(jobs, desc="strength comparison", disable=args.no_progress)
    for n_pos, n_neg, strategy_pos, strategy_neg in iterator:
        res = run_two_camp_configuration(
            G=G,
            n_pos=int(n_pos),
            n_neg=int(n_neg),
            T=int(cfg["T"]),
            burn_in=int(cfg["burn_in"]),
            n_runs=int(args.n_runs),
            seed=int(rng.integers(np.iinfo(np.int32).max)),
            strategy_pos=strategy_pos,
            strategy_neg=strategy_neg,
            threshold=float(args.threshold),
            wl_n_iter=int(args.wl_n_iter),
            show_progress=False,
            store_run_records=False,
        )

        row = configuration_result_to_dataset_row(
            res,
            graph_type=str(cfg.get("graph_type", "erdos_renyi")),
            graph_params={
                "n": cfg.get("n"),
                "p": cfg.get("p"),
                "m": cfg.get("m"),
                "L": cfg.get("L"),
            },
        )
        rows.append(row)

    raw_dir = ensure_dir(Path(args.results_dir) / "raw")
    fig_dir = ensure_dir(Path(args.results_dir) / "figures")

    rows_json = raw_dir / "strength_comparison_rows.json"
    rows_csv = raw_dir / "strength_comparison_rows.csv"
    save_dataset_rows(rows, json_path=rows_json, csv_path=rows_csv)

    target = np.asarray([float(r["positive_win_probability"]) for r in rows], dtype=float)
    candidate_cols = sorted([k for k in rows[0].keys() if k.startswith("mean_delta_psi_")])

    ranking = []
    for col in candidate_cols:
        x = np.asarray([float(r[col]) for r in rows], dtype=float)
        pearson = _pearson(x, target)
        spear = _spearman(x, target)
        ranking.append(
            {
                "candidate": col,
                "pearson_corr": pearson,
                "spearman_corr": spear,
                "abs_spearman": abs(spear) if np.isfinite(spear) else float("nan"),
            }
        )

    ranking = sorted(
        ranking,
        key=lambda d: (-(d["abs_spearman"] if np.isfinite(d["abs_spearman"]) else -np.inf), d["candidate"]),
    )

    ranking_path = raw_dir / "strength_predictive_rankings.json"
    with ranking_path.open("w", encoding="utf-8") as f:
        json.dump({"ranking": ranking}, f, indent=2, default=str)

    top = ranking[: min(8, len(ranking))]
    if len(top) > 0:
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        labels = [r["candidate"].replace("mean_", "") for r in top]
        vals = [r["spearman_corr"] for r in top]
        ax.bar(np.arange(len(vals)), vals)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Spearman correlation with P(positive win)")
        ax.set_title("Effective-strength candidate ranking")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        corr_path = fig_dir / "strength_candidate_correlations.png"
        fig.savefig(corr_path, dpi=180)
        plt.close(fig)

    if len(ranking) > 0:
        best = ranking[0]["candidate"]
        x = np.asarray([float(r[best]) for r in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.scatter(x, target, alpha=0.8)
        ax.set_xlabel(best)
        ax.set_ylabel("positive_win_probability")
        ax.set_title("Best candidate vs positive victory probability")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        best_path = fig_dir / "strength_best_candidate_scatter.png"
        fig.savefig(best_path, dpi=180)
        plt.close(fig)

    LOGGER.info("Saved strength-comparison rows to %s and %s", rows_json, rows_csv)
    LOGGER.info("Saved predictive ranking to %s", ranking_path)


if __name__ == "__main__":
    main()

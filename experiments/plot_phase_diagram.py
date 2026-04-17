"""Plot phase-diagram heatmaps from tipping-grid results."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=ROOT / "results" / "raw" / "tipping_grid_results.npz",
    )
    parser.add_argument(
        "--boundary-json",
        type=Path,
        default=ROOT / "results" / "raw" / "tipping_grid_boundary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "figures",
    )
    parser.add_argument("--title-prefix", type=str, default="Two-Camp Tipping")
    return parser.parse_args()


def _load_boundary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    points = payload.get("boundary_points", [])
    if not isinstance(points, list):
        return []
    return points


def main() -> None:
    args = parse_args()
    setup_logging()

    data = np.load(args.input_npz)
    n_pos = np.asarray(data["n_pos_values"], dtype=float)
    n_neg = np.asarray(data["n_neg_values"], dtype=float)
    mean_m = np.asarray(data["mean_m_grid"], dtype=float)
    pos_win = np.asarray(data["positive_win_probability_grid"], dtype=float)

    if mean_m.shape != (n_pos.size, n_neg.size):
        raise ValueError("mean_m_grid shape does not match (len(n_pos_values), len(n_neg_values)).")

    output_dir = ensure_dir(args.output_dir)
    boundary_points = _load_boundary(args.boundary_json)

    X, Y = np.meshgrid(n_neg, n_pos)

    fig1, ax1 = plt.subplots(figsize=(7.2, 5.6))
    c1 = ax1.pcolormesh(X, Y, mean_m, shading="auto", cmap="coolwarm")
    fig1.colorbar(c1, ax=ax1, label=r"$\bar{m}$")
    cs1 = ax1.contour(X, Y, mean_m, levels=[0.0], colors="k", linewidths=1.5)
    ax1.clabel(cs1, fmt={0.0: "m=0"}, inline=True, fontsize=8)

    if boundary_points:
        bx = [float(p["n_neg"]) for p in boundary_points]
        by = [float(p["n_pos"]) for p in boundary_points]
        ax1.scatter(bx, by, s=16, c="k", alpha=0.6, label="|m| <= eps boundary")
        ax1.legend(loc="best")

    ax1.set_xlabel("n_- (negative zealots)")
    ax1.set_ylabel("n_+ (positive zealots)")
    ax1.set_title(f"{args.title_prefix}: Mean Magnetization")
    ax1.grid(alpha=0.2)
    fig1.tight_layout()
    out1 = output_dir / "phase_diagram_mean_m.png"
    fig1.savefig(out1, dpi=180)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7.2, 5.6))
    c2 = ax2.pcolormesh(X, Y, pos_win, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    fig2.colorbar(c2, ax=ax2, label="P(positive victory)")
    cs2 = ax2.contour(X, Y, pos_win, levels=[0.5], colors="w", linewidths=1.5)
    ax2.clabel(cs2, fmt={0.5: "P=0.5"}, inline=True, fontsize=8)

    ax2.set_xlabel("n_- (negative zealots)")
    ax2.set_ylabel("n_+ (positive zealots)")
    ax2.set_title(f"{args.title_prefix}: Positive Victory Probability")
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    out2 = output_dir / "phase_diagram_positive_win_probability.png"
    fig2.savefig(out2, dpi=180)
    plt.close(fig2)

    LOGGER.info("Saved phase-diagram figures to %s", output_dir)


if __name__ == "__main__":
    main()

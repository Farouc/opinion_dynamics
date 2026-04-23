"""Experiment 2: phase diagrams and tipping boundaries for two-camp competition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.utils_exp import (
    GRAPH_COLORS,
    graph_label,
    make_graph,
    run_strategy_multirun,
    save_figure,
    save_json,
    setup_output_dirs,
    summarize_to_stdout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=25000)
    parser.add_argument("--burn-in", type=int, default=5000)
    parser.add_argument("--n-runs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pos-values", type=str, default="5,10,15,20,25,30,35,40")
    parser.add_argument("--n-neg-values", type=str, default="5,10,15,20,25,30,35,40,50")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="exp2_phase_diagram_tipping")
    return parser.parse_args()


def _parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("expected at least one integer")
    return sorted(set(vals))


def _graph_specs() -> list[dict]:
    return [
        {"graph_type": "barabasi_albert", "n": 800, "m": 3},
        {"graph_type": "erdos_renyi", "n": 800, "p": 0.015},
    ]


def _strategy_pairs() -> list[tuple[str, str]]:
    return [
        ("random", "random"),
        ("highest_degree", "random"),
        ("hub_then_spread", "random"),
    ]


def _graph_token(spec: dict) -> str:
    g = str(spec["graph_type"])
    n = int(spec["n"])
    if g == "erdos_renyi":
        return f"{g}_N{n}_p{str(float(spec['p'])).replace('.', '')}"
    if g == "barabasi_albert":
        return f"{g}_N{n}_m{int(spec['m'])}"
    return f"{g}_N{n}"


def _first_crossing_toward_positive(series: np.ndarray) -> float:
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return float("nan")

    if arr[0] > 0.0:
        return 1.0

    for t in range(1, arr.size):
        if arr[t] > 0.0 and arr[t - 1] <= 0.0:
            return float(t + 1)

    return float("nan")


def _build_grid(
    G,
    n_pos_values: list[int],
    n_neg_values: list[int],
    strategy_pos: str,
    strategy_neg: str,
    T: int,
    burn_in: int,
    n_runs: int,
    seed: int,
) -> dict:
    mean_m_grid = np.full((len(n_pos_values), len(n_neg_values)), np.nan, dtype=float)
    pwin_grid = np.full_like(mean_m_grid, np.nan)
    crossing_grid = np.full_like(mean_m_grid, np.nan)

    rows: list[dict] = []
    rng = np.random.default_rng(seed)

    for i, n_pos in enumerate(n_pos_values):
        for j, n_neg in enumerate(n_neg_values):
            if n_pos + n_neg > G.number_of_nodes():
                continue

            run_out = run_strategy_multirun(
                G=G,
                n_pos=int(n_pos),
                n_neg=int(n_neg),
                strategy_pos=strategy_pos,
                strategy_neg=strategy_neg,
                T=int(T),
                burn_in=int(burn_in),
                n_runs=int(n_runs),
                base_seed=int(rng.integers(np.iinfo(np.int32).max)),
                threshold=0.0,
                record=False,
            )

            m_runs = np.asarray(run_out["mean_m_runs"], dtype=float)
            m_series = np.asarray(run_out["magnetization"], dtype=float)

            crossing = np.array([_first_crossing_toward_positive(m_series[r]) for r in range(m_series.shape[0])], dtype=float)

            mean_m = float(np.mean(m_runs))
            pwin = float(np.mean(m_runs > 0.0))
            mean_cross = float(np.nanmean(crossing)) if np.any(np.isfinite(crossing)) else float("nan")

            mean_m_grid[i, j] = mean_m
            pwin_grid[i, j] = pwin
            crossing_grid[i, j] = mean_cross

            rows.append(
                {
                    "n_pos": int(n_pos),
                    "n_neg": int(n_neg),
                    "strategy_pos": str(strategy_pos),
                    "strategy_neg": str(strategy_neg),
                    "mean_magnetization": mean_m,
                    "P_plus_win": pwin,
                    "mean_crossing_time": mean_cross,
                    "is_minority_positive": int(n_pos < n_neg),
                    "minority_dominance": int((n_pos < n_neg) and (pwin > 0.5)),
                }
            )

    return {
        "mean_m_grid": mean_m_grid,
        "pwin_grid": pwin_grid,
        "crossing_grid": crossing_grid,
        "rows": rows,
    }


def _extract_contour_boundary(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, level: float = 0.5) -> list[list[list[float]]]:
    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, Z, levels=[level])
    segments: list[list[list[float]]] = []
    if hasattr(cs, "allsegs") and len(cs.allsegs) > 0:
        for seg in cs.allsegs[0]:
            arr = np.asarray(seg, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
                segments.append([[float(x), float(y)] for x, y in arr])
    plt.close(fig)
    return segments


def _plot_heatmap_pwin(
    X,
    Y,
    P,
    n_pos_values,
    n_neg_values,
    minority_star_mask,
    title,
    out_path: Path,
) -> list[list[list[float]]]:
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    c = ax.pcolormesh(X, Y, P, cmap="RdBu", vmin=0.0, vmax=1.0, shading="auto")
    fig.colorbar(c, ax=ax, label="P_plus_win")

    cs = ax.contour(X, Y, P, levels=[0.5], colors="k", linewidths=1.8)
    ax.clabel(cs, inline=True, fmt={0.5: "P=0.5"}, fontsize=9)

    for i, n_pos in enumerate(n_pos_values):
        for j, n_neg in enumerate(n_neg_values):
            if minority_star_mask[i, j]:
                ax.text(float(n_neg), float(n_pos), "★", color="black", fontsize=11, ha="center", va="center")

    ax.set_xlabel("Number of negative zealots n-")
    ax.set_ylabel("Number of positive zealots n+")
    ax.set_title(title)
    ax.grid(alpha=0.2)

    save_figure(fig, out_path, dpi=300)

    segments = []
    if hasattr(cs, "allsegs") and len(cs.allsegs) > 0:
        for seg in cs.allsegs[0]:
            arr = np.asarray(seg, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
                segments.append([[float(x), float(y)] for x, y in arr])

    plt.close(fig)
    return segments


def _plot_heatmap_mean_m(X, Y, M, title, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    vmax = float(np.nanmax(np.abs(M))) if np.any(np.isfinite(M)) else 1.0
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    c = ax.pcolormesh(X, Y, M, cmap="coolwarm", norm=norm, shading="auto")
    fig.colorbar(c, ax=ax, label="mean_magnetization")

    ax.set_xlabel("Number of negative zealots n-")
    ax.set_ylabel("Number of positive zealots n+")
    ax.set_title(title)
    ax.grid(alpha=0.2)

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_heatmap_crossing(X, Y, C, title, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    cmap = plt.cm.magma.copy()
    cmap.set_bad("white")
    c = ax.pcolormesh(X, Y, C, cmap=cmap, shading="auto")
    fig.colorbar(c, ax=ax, label="mean crossing time to + side")

    ax.set_xlabel("Number of negative zealots n-")
    ax.set_ylabel("Number of positive zealots n+")
    ax.set_title(title)
    ax.grid(alpha=0.2)

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_boundary_overlay(
    n_pos_values: list[int],
    n_neg_values: list[int],
    boundaries: dict[tuple[str, str], list[list[list[float]]]],
    graph_title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    for (s_pos, s_neg), segments in boundaries.items():
        color = GRAPH_COLORS.get("barabasi_albert", "#C0392B") if s_pos == "hub_then_spread" else (
            GRAPH_COLORS.get("erdos_renyi", "#2980B9") if s_pos == "highest_degree" else "#444444"
        )
        linestyle = "-" if s_pos == "random" else ("--" if s_pos == "highest_degree" else "-.")
        label = f"{s_pos} vs {s_neg}"
        drawn = False
        for seg in segments:
            arr = np.asarray(seg, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue
            ax.plot(arr[:, 0], arr[:, 1], color=color, linestyle=linestyle, linewidth=2.1, label=label if not drawn else None)
            drawn = True

    ax.set_xlim(min(n_neg_values) - 1, max(n_neg_values) + 1)
    ax.set_ylim(min(n_pos_values) - 1, max(n_pos_values) + 1)
    ax.set_xlabel("Number of negative zealots n-")
    ax.set_ylabel("Number of positive zealots n+")
    ax.set_title(f"Tipping boundary comparison (P_plus_win=0.5) | {graph_title}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dirs = setup_output_dirs(args.experiment_name)

    n_pos_values = _parse_int_list(args.n_pos_values)
    n_neg_values = _parse_int_list(args.n_neg_values)

    graph_specs = _graph_specs()
    strategy_pairs = _strategy_pairs()

    total_jobs = len(graph_specs) * len(strategy_pairs)
    pbar = tqdm(total=total_jobs, desc="exp2", disable=args.no_progress)

    n_figures = 0
    n_raw_files = 0

    for gspec in graph_specs:
        gtype = str(gspec["graph_type"])
        n = int(gspec["n"])
        G = make_graph(**gspec, seed=int(args.seed))

        X, Y = np.meshgrid(np.asarray(n_neg_values, dtype=float), np.asarray(n_pos_values, dtype=float))

        boundaries: dict[tuple[str, str], list[list[list[float]]]] = {}

        for strategy_pos, strategy_neg in strategy_pairs:
            out = _build_grid(
                G=G,
                n_pos_values=n_pos_values,
                n_neg_values=n_neg_values,
                strategy_pos=strategy_pos,
                strategy_neg=strategy_neg,
                T=int(args.T),
                burn_in=int(args.burn_in),
                n_runs=int(args.n_runs),
                seed=int(args.seed),
            )

            M = np.asarray(out["mean_m_grid"], dtype=float)
            P = np.asarray(out["pwin_grid"], dtype=float)
            C = np.asarray(out["crossing_grid"], dtype=float)
            rows = out["rows"]

            star_mask = np.zeros_like(P, dtype=bool)
            for i, n_pos in enumerate(n_pos_values):
                for j, n_neg in enumerate(n_neg_values):
                    star_mask[i, j] = bool((n_pos < n_neg) and (P[i, j] > 0.5))

            gtoken = _graph_token(gspec)
            pair_token = f"{strategy_pos}_vs_{strategy_neg}"

            fig2a_name = f"phase_diagram_Pwin_{gtoken}_{pair_token}.png"
            segs = _plot_heatmap_pwin(
                X=X,
                Y=Y,
                P=P,
                n_pos_values=n_pos_values,
                n_neg_values=n_neg_values,
                minority_star_mask=star_mask,
                title=(
                    f"P_plus_win heatmap | {graph_label(gtype)} N={n} | {strategy_pos} vs {strategy_neg}\n"
                    "Can minority positive camp win via structural placement?"
                ),
                out_path=dirs["figures"] / fig2a_name,
            )
            n_figures += 1
            boundaries[(strategy_pos, strategy_neg)] = segs

            fig2b_name = f"phase_diagram_mean_m_{gtoken}_{pair_token}.png"
            _plot_heatmap_mean_m(
                X=X,
                Y=Y,
                M=M,
                title=f"Mean magnetization heatmap | {graph_label(gtype)} N={n} | {strategy_pos} vs {strategy_neg}",
                out_path=dirs["figures"] / fig2b_name,
            )
            n_figures += 1

            fig2d_name = f"crossing_time_heatmap_{gtoken}_{pair_token}.png"
            _plot_heatmap_crossing(
                X=X,
                Y=Y,
                C=C,
                title=f"Mean crossing time to + side | {graph_label(gtype)} N={n} | {strategy_pos} vs {strategy_neg}",
                out_path=dirs["figures"] / fig2d_name,
            )
            n_figures += 1

            npz_name = f"phase_grid_{gtoken}_{pair_token}.npz"
            np.savez(
                dirs["raw"] / npz_name,
                n_pos_values=np.asarray(n_pos_values, dtype=int),
                n_neg_values=np.asarray(n_neg_values, dtype=int),
                mean_magnetization=M,
                P_plus_win=P,
                mean_crossing_time=C,
            )
            n_raw_files += 1

            csv_name = f"phase_grid_{gtoken}_{pair_token}.csv"
            pd.DataFrame(rows).to_csv(dirs["raw"] / csv_name, index=False)
            n_raw_files += 1

            json_name = f"tipping_boundary_{gtoken}_{pair_token}.json"
            save_json(
                dirs["raw"] / json_name,
                {
                    "graph_spec": gspec,
                    "strategy_pos": strategy_pos,
                    "strategy_neg": strategy_neg,
                    "boundary_segments": segs,
                },
            )
            n_raw_files += 1

            pbar.update(1)

        fig2c_name = f"tipping_boundary_comparison_{_graph_token(gspec)}.png"
        _plot_boundary_overlay(
            n_pos_values=n_pos_values,
            n_neg_values=n_neg_values,
            boundaries=boundaries,
            graph_title=f"{graph_label(gtype)} N={n}",
            out_path=dirs["figures"] / fig2c_name,
        )
        n_figures += 1

    summarize_to_stdout(
        "exp2_phase_diagram_tipping",
        {
            "output_base": str(dirs["base"]),
            "n_figures": int(n_figures),
            "n_raw_files": int(n_raw_files),
            "n_graphs": len(graph_specs),
            "n_strategy_pairs": len(strategy_pairs),
            "grid_shape": f"{len(n_pos_values)}x{len(n_neg_values)}",
        },
    )


if __name__ == "__main__":
    main()

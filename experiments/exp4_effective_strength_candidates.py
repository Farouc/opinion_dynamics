"""Experiment 4: evaluate effective-strength candidates against tipping outcomes."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.utils_exp import (
    GRAPH_COLORS,
    PALETTE,
    graph_label,
    make_graph,
    save_figure,
    save_json,
    setup_output_dirs,
    summarize_to_stdout,
)
from src.tipping_analysis import run_two_camp_configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=20000)
    parser.add_argument("--burn-in", type=int, default=4000)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--graph-types",
        type=str,
        default="barabasi_albert,erdos_renyi,fully_connected",
    )
    parser.add_argument("--n-pos-values", type=str, default="8,12,16,20,24")
    parser.add_argument("--n-neg-values", type=str, default="8,12,16,20,24,28")
    parser.add_argument(
        "--strategy-pos-list",
        type=str,
        default="random,highest_degree,farthest_spread,wl_cover,hub_then_spread",
    )
    parser.add_argument("--strategy-neg-list", type=str, default="random,highest_degree")
    parser.add_argument("--max-configs", type=int, default=0, help="Optional cap for quick smoke runs.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="exp4_effective_strength_candidates")
    return parser.parse_args()


def _parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("expected at least one int value")
    return sorted(set(vals))


def _parse_str_list(raw: str) -> list[str]:
    vals = [str(x).strip() for x in str(raw).split(",") if str(x).strip()]
    if len(vals) == 0:
        raise ValueError("expected at least one string value")
    return vals


def _graph_specs(graph_types: list[str]) -> list[dict]:
    all_specs = [
        {"graph_type": "barabasi_albert", "n": 400, "m": 3},
        {"graph_type": "erdos_renyi", "n": 400, "p": 0.02},
        {"graph_type": "fully_connected", "n": 200},
    ]
    wanted = set(graph_types)
    return [s for s in all_specs if str(s["graph_type"]) in wanted]


def _graph_token(spec: dict) -> str:
    g = str(spec["graph_type"])
    n = int(spec["n"])
    if g == "erdos_renyi":
        return f"{g}_N{n}_p{str(float(spec['p'])).replace('.', '')}"
    if g == "barabasi_albert":
        return f"{g}_N{n}_m{int(spec['m'])}"
    return f"{g}_N{n}"


def _logistic(x, a):
    return 1.0 / (1.0 + np.exp(-a * (x - 0.5)))


def _fit_logistic_steepness(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 4:
        return float("nan"), False

    try:
        popt, _ = curve_fit(_logistic, x, y, p0=[8.0], bounds=([-200.0], [200.0]), maxfev=20000)
        return float(popt[0]), True
    except Exception:
        return float("nan"), False


def _auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]

    if y.size == 0:
        return float("nan")

    pos = (y == 1)
    neg = (y == 0)
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)

    sum_ranks_pos = float(np.sum(ranks[pos]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _collect_dataset(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []

    graph_specs = _graph_specs(_parse_str_list(args.graph_types))
    n_pos_vals = _parse_int_list(args.n_pos_values)
    n_neg_vals = _parse_int_list(args.n_neg_values)
    spos_vals = _parse_str_list(args.strategy_pos_list)
    sneg_vals = _parse_str_list(args.strategy_neg_list)

    if len(graph_specs) == 0:
        raise ValueError("No graph specs selected; check --graph-types.")

    total = len(graph_specs) * len(n_pos_vals) * len(n_neg_vals) * len(spos_vals) * len(sneg_vals)
    pbar = tqdm(total=total, desc="exp4 dataset", disable=args.no_progress)

    rng = np.random.default_rng(int(args.seed))
    n_done = 0

    for spec in graph_specs:
        gtype = str(spec["graph_type"])
        G = make_graph(**spec, seed=int(rng.integers(np.iinfo(np.int32).max)))

        for n_pos in n_pos_vals:
            for n_neg in n_neg_vals:
                if n_pos + n_neg > G.number_of_nodes():
                    for _ in range(len(spos_vals) * len(sneg_vals)):
                        pbar.update(1)
                    continue

                for strategy_pos in spos_vals:
                    for strategy_neg in sneg_vals:
                        res = run_two_camp_configuration(
                            G=G,
                            n_pos=int(n_pos),
                            n_neg=int(n_neg),
                            T=int(args.T),
                            burn_in=int(args.burn_in),
                            n_runs=int(args.n_runs),
                            seed=int(rng.integers(np.iinfo(np.int32).max)),
                            strategy_pos=strategy_pos,
                            strategy_neg=strategy_neg,
                            threshold=0.0,
                            wl_n_iter=3,
                            show_progress=False,
                            store_run_records=False,
                        )

                        agg = res["aggregate"]
                        gf = res["graph_features"]

                        row = {
                            "graph_type": gtype,
                            "graph_token": _graph_token(spec),
                            "n_nodes": int(spec["n"]),
                            "graph_p": float(spec.get("p", np.nan)),
                            "graph_m": float(spec.get("m", np.nan)),
                            "n_pos": int(n_pos),
                            "n_neg": int(n_neg),
                            "strategy_pos": strategy_pos,
                            "strategy_neg": strategy_neg,
                            "P_plus_win": float(agg["positive_win_probability"]),
                            "mean_m": float(agg["mean_mean_magnetization"]),
                            "std_m": float(agg["std_mean_magnetization"]),
                            "degree_heterogeneity": float(gf.get("degree_variance", np.nan))
                            / max(float(gf.get("average_degree", 1.0)), 1e-12),
                        }

                        metrics = [
                            "psi_size",
                            "psi_rho",
                            "psi_degree",
                            "psi_degree_norm",
                            "psi_centrality",
                            "psi_wl",
                            "psi_dispersion",
                            "psi_hybrid",
                        ]
                        for m in metrics:
                            pos_val = float(agg.get(f"mean_pos_{m}", np.nan))
                            neg_val = float(agg.get(f"mean_neg_{m}", np.nan))
                            delta_val = float(agg.get(f"mean_delta_{m}", np.nan))
                            row[f"pos_{m}"] = pos_val
                            row[f"neg_{m}"] = neg_val
                            row[f"Delta_{m}"] = delta_val
                            denom = pos_val + neg_val
                            row[f"ratio_{m}"] = float(pos_val / denom) if np.isfinite(denom) and abs(denom) > 1e-15 else np.nan

                        rows.append(row)
                        n_done += 1
                        pbar.update(1)

                        if int(args.max_configs) > 0 and n_done >= int(args.max_configs):
                            pbar.close()
                            return pd.DataFrame(rows)

    pbar.close()
    return pd.DataFrame(rows)


def _plot_metric_scatter(df: pd.DataFrame, metric_col: str, out_path: Path) -> tuple[float, bool]:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    marker_map = {
        "random": "o",
        "highest_degree": "s",
        "farthest_spread": "^",
        "wl_cover": "D",
        "hub_then_spread": "P",
    }

    for gtype in sorted(df["graph_type"].unique()):
        sub_g = df[df["graph_type"] == gtype]
        for spos in sorted(sub_g["strategy_pos"].unique()):
            sub = sub_g[sub_g["strategy_pos"] == spos]
            ax.scatter(
                sub[metric_col],
                sub["P_plus_win"],
                color=GRAPH_COLORS.get(gtype, "#555555"),
                marker=marker_map.get(spos, "o"),
                alpha=0.78,
                s=36,
                edgecolors="none",
                label=f"{gtype}|{spos}",
            )

    x = df[metric_col].to_numpy(dtype=float)
    y = df["P_plus_win"].to_numpy(dtype=float)
    a_hat, ok = _fit_logistic_steepness(x, y)

    if ok and np.isfinite(a_hat):
        x_grid = np.linspace(0.0, 1.0, 250)
        y_grid = _logistic(x_grid, a_hat)
        ax.plot(x_grid, y_grid, color="#111111", linewidth=2.2, label=f"logistic fit a={a_hat:.2f}")

    ax.axvline(0.5, linestyle="--", color="#444444", linewidth=1.0)
    ax.axhline(0.5, linestyle="--", color="#444444", linewidth=1.0)

    ax.set_xlabel(metric_col)
    ax.set_ylabel("P_plus_win")
    ax.set_title(f"Metric vs P_plus_win: {metric_col}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # compact legend
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for h, l in zip(handles, labels):
        dedup[l] = h
    ax.legend(list(dedup.values())[:18], list(dedup.keys())[:18], loc="best", fontsize=7)

    ax.grid(alpha=0.25)
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)

    return a_hat, ok


def _plot_corr_bar(df: pd.DataFrame, metric_cols: list[str], graph_type: str, out_path: Path) -> dict[str, float]:
    sub = df[df["graph_type"] == graph_type]
    y = sub["P_plus_win"].to_numpy(dtype=float)

    corr = {}
    for col in metric_cols:
        x = sub[col].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 3:
            corr[col] = float("nan")
            continue
        x = x[mask]
        yy = y[mask]
        if np.std(x) < 1e-12 or np.std(yy) < 1e-12:
            corr[col] = float("nan")
        else:
            corr[col] = float(np.corrcoef(x, yy)[0, 1])

    corr_sorted = sorted(corr.items(), key=lambda kv: abs(kv[1]) if np.isfinite(kv[1]) else -1.0, reverse=True)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    xlab = [k for k, _ in corr_sorted]
    vals = [v for _, v in corr_sorted]
    colors = ["#2ECC71" if (np.isfinite(v) and v >= 0) else "#E74C3C" for v in vals]

    ax.bar(np.arange(len(vals)), vals, color=colors)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(xlab, rotation=35, ha="right")
    ax.set_ylabel("Pearson corr with P_plus_win")
    ax.set_title(f"Metric correlation ranking | {graph_label(graph_type)}")
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)

    return {k: float(v) for k, v in corr.items()}


def _plot_critical_region(df: pd.DataFrame, best_metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    for gtype in sorted(df["graph_type"].unique()):
        sub = df[df["graph_type"] == gtype]
        ax.scatter(
            sub[best_metric],
            sub["mean_m"],
            s=25 + 180 * np.clip(sub["std_m"].to_numpy(dtype=float), 0.0, 1.0),
            alpha=0.72,
            color=GRAPH_COLORS.get(gtype, "#555555"),
            label=gtype,
            edgecolors="none",
        )

    ax.axvspan(0.45, 0.55, color="#F9E79F", alpha=0.45, label="critical band |ratio-0.5|<0.05")
    ax.axvline(0.5, linestyle="--", color="#444444", linewidth=1.0)
    ax.axhline(0.0, linestyle="--", color="#444444", linewidth=1.0)

    ax.set_xlabel(best_metric)
    ax.set_ylabel("mean_m")
    ax.set_title(f"Critical region scan using {best_metric}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_minority_dominance(df: pd.DataFrame, graph_type: str, out_path: Path) -> int:
    sub = df[(df["graph_type"] == graph_type) & (df["n_pos"] < df["n_neg"]) & (df["P_plus_win"] > 0.6)].copy()
    if sub.shape[0] == 0:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.text(0.5, 0.5, "No minority-dominance cases found", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(f"Minority dominance cases | {graph_label(graph_type)}")
        save_figure(fig, out_path, dpi=300)
        plt.close(fig)
        return 0

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    for spos in sorted(sub["strategy_pos"].unique()):
        ss = sub[sub["strategy_pos"] == spos]
        ax.scatter(
            ss["ratio_psi_degree_norm"],
            ss["ratio_psi_wl"],
            s=120 * ss["P_plus_win"],
            alpha=0.8,
            color=PALETTE.get(spos, "#555555"),
            label=spos,
            edgecolors="#222222",
            linewidths=0.3,
        )

    ax.set_xlabel("ratio_psi_degree_norm")
    ax.set_ylabel("ratio_psi_wl")
    ax.set_title(f"Minority dominance structure map | {graph_label(graph_type)}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)
    return int(sub.shape[0])


def _plot_auc_summary(df: pd.DataFrame, metric_cols: list[str], out_path: Path) -> dict[str, dict[str, float]]:
    graph_types = sorted(df["graph_type"].unique())
    fig, axes = plt.subplots(1, len(graph_types), figsize=(6.8 * len(graph_types), 4.8), sharey=True)
    if len(graph_types) == 1:
        axes = [axes]

    auc_map: dict[str, dict[str, float]] = {}

    for ax, gtype in zip(axes, graph_types):
        sub = df[df["graph_type"] == gtype]
        y_bin = (sub["P_plus_win"].to_numpy(dtype=float) > 0.5).astype(int)

        vals = []
        for col in metric_cols:
            auc = _auc_binary(y_bin, sub[col].to_numpy(dtype=float))
            vals.append(auc)
            auc_map.setdefault(gtype, {})[col] = float(auc)

        ax.bar(np.arange(len(metric_cols)), vals, color=GRAPH_COLORS.get(gtype, "#555555"), alpha=0.85)
        ax.set_xticks(np.arange(len(metric_cols)))
        ax.set_xticklabels(metric_cols, rotation=35, ha="right")
        ax.set_title(graph_label(gtype))
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("AUC")

    fig.suptitle("Metric comparison summary: AUC for predicting I[P_plus_win > 0.5]")
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)

    return auc_map


def main() -> None:
    args = parse_args()
    dirs = setup_output_dirs(args.experiment_name)

    df = _collect_dataset(args)

    csv_path = dirs["raw"] / "strength_dataset_all_configs.csv"
    df.to_csv(csv_path, index=False)

    ratio_metrics = [
        "ratio_psi_rho",
        "ratio_psi_degree_norm",
        "ratio_psi_centrality",
        "ratio_psi_wl",
        "ratio_psi_dispersion",
        "ratio_psi_hybrid",
    ]

    logistic_fits: dict[str, dict[str, float | bool]] = {}
    n_figures = 0

    for metric in ratio_metrics:
        out_path = dirs["figures"] / f"metric_vs_Pwin_scatter_{metric}.png"
        a_hat, ok = _plot_metric_scatter(df=df, metric_col=metric, out_path=out_path)
        logistic_fits[metric] = {"a": float(a_hat), "fit_ok": bool(ok)}
        n_figures += 1

    metric_correlations: dict[str, dict[str, float]] = {}
    for gtype in sorted(df["graph_type"].unique()):
        out_path = dirs["figures"] / f"metric_correlation_with_Pwin_{gtype}.png"
        metric_correlations[gtype] = _plot_corr_bar(
            df=df,
            metric_cols=ratio_metrics,
            graph_type=gtype,
            out_path=out_path,
        )
        n_figures += 1

    # best metric by average absolute correlation across graph families
    metric_scores = {}
    for metric in ratio_metrics:
        vals = []
        for gtype in metric_correlations:
            v = metric_correlations[gtype].get(metric, np.nan)
            if np.isfinite(v):
                vals.append(abs(float(v)))
        metric_scores[metric] = float(np.mean(vals)) if len(vals) > 0 else -np.inf

    best_metric = max(metric_scores.keys(), key=lambda k: metric_scores[k])

    fig4c_path = dirs["figures"] / f"critical_region_scan_{best_metric}_all_graphs.png"
    _plot_critical_region(df=df, best_metric=best_metric, out_path=fig4c_path)
    n_figures += 1

    minority_counts = {}
    for gtype in sorted(df["graph_type"].unique()):
        out_path = dirs["figures"] / f"minority_dominance_cases_{gtype}.png"
        minority_counts[gtype] = _plot_minority_dominance(df=df, graph_type=gtype, out_path=out_path)
        n_figures += 1

    fig4e_path = dirs["figures"] / "metric_comparison_summary.png"
    auc_map = _plot_auc_summary(df=df, metric_cols=ratio_metrics, out_path=fig4e_path)
    n_figures += 1

    save_json(dirs["raw"] / "metric_correlations.json", metric_correlations)
    save_json(dirs["raw"] / "logistic_fits.json", {"best_metric": best_metric, "fits": logistic_fits, "auc": auc_map})

    summarize_to_stdout(
        "exp4_effective_strength_candidates",
        {
            "output_base": str(dirs["base"]),
            "dataset_rows": int(df.shape[0]),
            "n_figures": int(n_figures),
            "best_metric": best_metric,
            "minority_dominance_counts": minority_counts,
            "csv": str(csv_path),
        },
    )


if __name__ == "__main__":
    main()

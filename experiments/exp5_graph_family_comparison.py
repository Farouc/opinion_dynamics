"""Experiment 5: graph-family comparison for two-camp zealot dynamics."""

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
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.utils_exp import (
    GRAPH_COLORS,
    PALETTE,
    graph_label,
    make_graph,
    run_strategy_multirun,
    save_figure,
    save_json,
    setup_output_dirs,
    summarize_to_stdout,
)
from src.tipping_analysis import run_two_camp_configuration
from src.zealot_assignment import assign_two_zealot_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=30000)
    parser.add_argument("--burn-in", type=int, default=6000)
    parser.add_argument("--n-runs", type=int, default=15)
    parser.add_argument("--n-pos", type=int, default=20)
    parser.add_argument("--n-neg", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="exp5_graph_family_comparison")
    return parser.parse_args()


def _graph_specs() -> list[dict]:
    specs = [{"graph_type": "fully_connected", "n": 200, "family": "fully_connected", "config_label": "FC_N200"}]
    for p in [0.008, 0.015, 0.03]:
        specs.append(
            {
                "graph_type": "erdos_renyi",
                "n": 500,
                "p": p,
                "family": "erdos_renyi",
                "config_label": f"ER_p{str(p).replace('.', '')}",
            }
        )
    for m in [2, 3, 5]:
        specs.append(
            {
                "graph_type": "barabasi_albert",
                "n": 500,
                "m": m,
                "family": "barabasi_albert",
                "config_label": f"BA_m{m}",
            }
        )
    return specs


def _strategy_pairs() -> list[tuple[str, str]]:
    return [("highest_degree", "random"), ("random", "random")]


def _time_to_abs_threshold(series: np.ndarray, thr: float = 0.3) -> float:
    arr = np.asarray(series, dtype=float)
    idx = np.where(np.abs(arr) > float(thr))[0]
    return float(idx[0] + 1) if idx.size > 0 else float("nan")


def _degree_heterogeneity(G) -> float:
    deg = np.asarray([float(d) for _, d in G.degree()], dtype=float)
    if deg.size == 0:
        return float("nan")
    mean = float(np.mean(deg))
    if abs(mean) < 1e-15:
        return float("nan")
    return float(np.std(deg) / mean)


def _run_experiment_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    dist_rows = []
    degree_dist = {}

    specs = _graph_specs()
    pairs = _strategy_pairs()

    total = len(specs) * len(pairs)
    pbar = tqdm(total=total, desc="exp5", disable=args.no_progress)

    rng = np.random.default_rng(int(args.seed))

    for spec in specs:
        gtype = str(spec["graph_type"])
        family = str(spec["family"])
        config_label = str(spec["config_label"])

        G = make_graph(**spec, seed=int(rng.integers(np.iinfo(np.int32).max)))
        hetero = _degree_heterogeneity(G)

        degrees = [int(d) for _, d in G.degree()]
        degree_dist[config_label] = {
            "graph_type": gtype,
            "family": family,
            "config_label": config_label,
            "degrees": degrees,
            "k_max": int(np.max(degrees)) if len(degrees) > 0 else 0,
            "k_mean": float(np.mean(degrees)) if len(degrees) > 0 else 0.0,
        }

        # structural metrics under highest_degree vs random
        struct_out = run_two_camp_configuration(
            G=G,
            n_pos=int(args.n_pos),
            n_neg=int(args.n_neg),
            T=int(args.T),
            burn_in=int(args.burn_in),
            n_runs=max(4, int(min(args.n_runs, 8))),
            seed=int(rng.integers(np.iinfo(np.int32).max)),
            strategy_pos="highest_degree",
            strategy_neg="random",
            threshold=0.0,
            wl_n_iter=3,
            show_progress=False,
            store_run_records=False,
        )
        struct_agg = struct_out["aggregate"]
        psi_degree_norm = float(struct_agg.get("mean_pos_psi_degree_norm", np.nan))
        psi_wl = float(struct_agg.get("mean_pos_psi_wl", np.nan))

        for strategy_pos, strategy_neg in pairs:
            run_out = run_strategy_multirun(
                G=G,
                n_pos=int(args.n_pos),
                n_neg=int(args.n_neg),
                strategy_pos=strategy_pos,
                strategy_neg=strategy_neg,
                T=int(args.T),
                burn_in=int(args.burn_in),
                n_runs=int(args.n_runs),
                base_seed=int(rng.integers(np.iinfo(np.int32).max)),
                threshold=0.0,
                record=False,
            )

            m_all = np.asarray(run_out["magnetization"], dtype=float)
            conv_times = np.array([_time_to_abs_threshold(m_all[r], thr=0.3) for r in range(m_all.shape[0])], dtype=float)

            mean_m_runs = np.asarray(run_out["mean_m_runs"], dtype=float)
            pwin = float(np.mean(mean_m_runs > 0.0))

            rows.append(
                {
                    "graph_type": gtype,
                    "family": family,
                    "config_label": config_label,
                    "strategy_pos": strategy_pos,
                    "strategy_neg": strategy_neg,
                    "n_nodes": int(spec["n"]),
                    "n_pos": int(args.n_pos),
                    "n_neg": int(args.n_neg),
                    "degree_heterogeneity": hetero,
                    "P_plus_win": pwin,
                    "mean_m": float(np.mean(mean_m_runs)),
                    "std_m": float(np.std(mean_m_runs, ddof=1 if mean_m_runs.size > 1 else 0)),
                    "mean_time_abs_m_gt_03": float(np.nanmean(conv_times)) if np.any(np.isfinite(conv_times)) else np.nan,
                    "std_time_abs_m_gt_03": float(np.nanstd(conv_times, ddof=1)) if np.sum(np.isfinite(conv_times)) > 1 else 0.0,
                    "psi_degree_norm_pos_highdeg": psi_degree_norm,
                    "psi_wl_pos_highdeg": psi_wl,
                }
            )

            for ridx, val in enumerate(mean_m_runs):
                dist_rows.append(
                    {
                        "graph_type": gtype,
                        "family": family,
                        "config_label": config_label,
                        "strategy_pair": f"{strategy_pos}_vs_{strategy_neg}",
                        "seed_idx": int(ridx),
                        "mean_m": float(val),
                    }
                )

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows), pd.DataFrame(dist_rows), degree_dist


def _plot_convergence_speed(df: pd.DataFrame, out_path: Path) -> None:
    summary = (
        df.groupby(["config_label", "strategy_pos", "strategy_neg"], as_index=False)
        .agg(mean_conv=("mean_time_abs_m_gt_03", "mean"), std_conv=("mean_time_abs_m_gt_03", "std"))
        .fillna(0.0)
    )

    order = ["FC_N200", "ER_p0008", "ER_p0015", "ER_p003", "BA_m2", "BA_m3", "BA_m5"]
    summary["config_label"] = pd.Categorical(summary["config_label"], categories=order, ordered=True)
    summary = summary.sort_values("config_label")

    fig, ax = plt.subplots(figsize=(11.5, 5.0))

    x_labels = list(dict.fromkeys(summary["config_label"].tolist()))
    x = np.arange(len(x_labels))
    w = 0.38

    for idx, (s_pos, s_neg) in enumerate(_strategy_pairs()):
        sub = summary[(summary["strategy_pos"] == s_pos) & (summary["strategy_neg"] == s_neg)]
        sub = sub.set_index("config_label").reindex(x_labels).reset_index()
        means = sub["mean_conv"].to_numpy(dtype=float)
        stds = sub["std_conv"].to_numpy(dtype=float)

        ax.bar(
            x + (idx - 0.5) * w,
            means,
            width=w,
            yerr=stds,
            capsize=3,
            label=f"{s_pos} vs {s_neg}",
            color=PALETTE.get(s_pos, "#777777"),
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel("mean time to |m| > 0.3")
    ax.set_title("Convergence speed by graph configuration")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_pwin_vs_heterogeneity(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 5.6))

    for s_pos, s_neg in _strategy_pairs():
        sub = df[(df["strategy_pos"] == s_pos) & (df["strategy_neg"] == s_neg)]
        ax.scatter(
            sub["degree_heterogeneity"],
            sub["P_plus_win"],
            s=80,
            alpha=0.85,
            color=PALETTE.get(s_pos, "#555555"),
            label=f"{s_pos} vs {s_neg}",
            edgecolors="#222222",
            linewidths=0.4,
        )
        for _, row in sub.iterrows():
            ax.text(float(row["degree_heterogeneity"]), float(row["P_plus_win"]) + 0.02, str(row["config_label"]), fontsize=7)

    ax.set_xlabel("degree heterogeneity std(k)/mean(k)")
    ax.set_ylabel("P_plus_win")
    ax.set_title("P_plus_win vs degree heterogeneity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_magnetization_distributions(df_dist: pd.DataFrame, out_dir: Path) -> int:
    n = 0
    for family in sorted(df_dist["family"].unique()):
        sub = df_dist[df_dist["family"] == family]
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        sns.histplot(
            data=sub,
            x="mean_m",
            hue="strategy_pair",
            bins=24,
            stat="density",
            common_norm=False,
            alpha=0.45,
            ax=ax,
        )
        ax.set_xlabel("mean_m over post-burn-in")
        ax.set_ylabel("density")
        ax.set_title(f"Final magnetization distribution | {graph_label(family)}")
        ax.grid(alpha=0.25)

        out_path = out_dir / f"magnetization_distribution_final_{family}.png"
        save_figure(fig, out_path, dpi=300)
        plt.close(fig)
        n += 1
    return n


def _plot_network_family_comparison(seed: int, out_path: Path) -> None:
    small_specs = [
        {"graph_type": "fully_connected", "n": 70, "title": "FC"},
        {"graph_type": "erdos_renyi", "n": 100, "p": 0.04, "title": "ER"},
        {"graph_type": "barabasi_albert", "n": 100, "m": 3, "title": "BA"},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))

    for ax, spec in zip(axes, small_specs):
        G = make_graph(**spec, seed=seed)
        n_pos = 8
        n_neg = 8
        pos_mask, neg_mask, _ = assign_two_zealot_sets(
            G=G,
            n_pos=n_pos,
            n_neg=n_neg,
            strategy_pos="highest_degree",
            strategy_neg="random",
            seed=seed,
        )
        pos_nodes = set(np.where(pos_mask)[0].astype(int).tolist())
        neg_nodes = set(np.where(neg_mask)[0].astype(int).tolist())

        pos = __import__("networkx").spring_layout(G, seed=seed)
        deg = dict(G.degree())
        max_deg = max(deg.values()) if len(deg) > 0 else 1

        node_colors = []
        node_sizes = []
        for node in G.nodes():
            i = int(node)
            if i in pos_nodes:
                node_colors.append("#E74C3C")
            elif i in neg_nodes:
                node_colors.append("#3498DB")
            else:
                node_colors.append("#BFC9CA")
            node_sizes.append(80 + 360 * (deg[i] / max_deg if max_deg > 0 else 0.0))

        __import__("networkx").draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.7)
        __import__("networkx").draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            linewidths=0.4,
            edgecolors="#222222",
        )

        k_max = int(np.max(list(deg.values()))) if len(deg) > 0 else 0
        k_mean = float(np.mean(list(deg.values()))) if len(deg) > 0 else 0.0
        ax.set_title(f"{spec['title']} | k_max={k_max}, k_mean={k_mean:.1f}")
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        Patch(color="#E74C3C", label="positive zealot"),
        Patch(color="#3498DB", label="negative zealot"),
        Patch(color="#BFC9CA", label="free node"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    fig.suptitle("Network visualizations comparison (highest_degree vs random)")

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_structural_metrics(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df["strategy_pos"] == "highest_degree"].copy()
    agg = (
        sub.groupby("family", as_index=False)
        .agg(
            psi_degree_norm=("psi_degree_norm_pos_highdeg", "mean"),
            psi_wl=("psi_wl_pos_highdeg", "mean"),
        )
        .sort_values("family")
    )

    x = np.arange(agg.shape[0])
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar(x - w / 2, agg["psi_degree_norm"], width=w, label="psi_degree_norm(Z_plus)", color="#D35400")
    ax.bar(x + w / 2, agg["psi_wl"], width=w, label="psi_wl(Z_plus)", color="#16A085")

    ax.set_xticks(x)
    ax.set_xticklabels([graph_label(v) for v in agg["family"].tolist()])
    ax.set_ylabel("metric value")
    ax.set_title("Structural metrics by graph family (highest_degree placement)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dirs = setup_output_dirs(args.experiment_name)

    df, df_dist, degree_dist = _run_experiment_dataset(args)

    n_figures = 0
    n_raw_files = 0

    for s_pos, s_neg in _strategy_pairs():
        sub = df[(df["strategy_pos"] == s_pos) & (df["strategy_neg"] == s_neg)].copy()
        out_csv = dirs["raw"] / f"graph_family_comparison_{s_pos}_vs_{s_neg}.csv"
        sub.to_csv(out_csv, index=False)
        n_raw_files += 1

    for key, payload in degree_dist.items():
        out_json = dirs["raw"] / f"degree_distributions_{key}.json"
        save_json(out_json, payload)
        n_raw_files += 1

    fig5a = dirs["figures"] / "convergence_speed_by_graph_strategy_comparison.png"
    _plot_convergence_speed(df, fig5a)
    n_figures += 1

    fig5b = dirs["figures"] / "Pwin_vs_degree_heterogeneity_strategy_comparison.png"
    _plot_pwin_vs_heterogeneity(df, fig5b)
    n_figures += 1

    n_figures += _plot_magnetization_distributions(df_dist, dirs["figures"])

    fig5d = dirs["figures"] / "network_visualizations_comparison.png"
    _plot_network_family_comparison(seed=int(args.seed), out_path=fig5d)
    n_figures += 1

    fig5e = dirs["figures"] / "structural_metrics_by_graph_highest_degree_vs_random.png"
    _plot_structural_metrics(df, fig5e)
    n_figures += 1

    summarize_to_stdout(
        "exp5_graph_family_comparison",
        {
            "output_base": str(dirs["base"]),
            "n_rows": int(df.shape[0]),
            "n_distribution_rows": int(df_dist.shape[0]),
            "n_figures": int(n_figures),
            "n_raw_files": int(n_raw_files),
        },
    )


if __name__ == "__main__":
    main()

"""Experiment 3: WL structural placement and strategy comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.utils_exp import (
    PALETTE,
    draw_network_opinion,
    graph_label,
    make_graph,
    save_figure,
    save_json,
    setup_output_dirs,
    summarize_to_stdout,
)
from src.tipping_analysis import run_two_camp_configuration
from src.wl_features import wl_color_refinement
from src.zealot_assignment import assign_two_zealot_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=20000)
    parser.add_argument("--burn-in", type=int, default=4000)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--n-pos", type=int, default=15)
    parser.add_argument("--n-neg", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wl-n-iter", type=int, default=6)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="exp3_wl_cluster_placement")
    return parser.parse_args()


def _graph_specs() -> list[dict]:
    return [
        {"graph_type": "barabasi_albert", "n": 300, "m": 2},
        {"graph_type": "erdos_renyi", "n": 300, "p": 0.02},
    ]


def _graph_token(spec: dict) -> str:
    g = str(spec["graph_type"])
    n = int(spec["n"])
    if g == "erdos_renyi":
        return f"{g}_N{n}_p{str(float(spec['p'])).replace('.', '')}"
    if g == "barabasi_albert":
        return f"{g}_N{n}_m{int(spec['m'])}"
    return f"{g}_N{n}"


def _build_class_maps(labels: dict[int, int]) -> dict[int, list[int]]:
    class_to_nodes: dict[int, list[int]] = {}
    for node, cls in labels.items():
        class_to_nodes.setdefault(int(cls), []).append(int(node))
    for cls in class_to_nodes:
        class_to_nodes[cls] = sorted(class_to_nodes[cls])
    return class_to_nodes


def _plot_wl_partition_evolution(
    G: nx.Graph,
    labels_final: dict[int, int],
    class_to_nodes: dict[int, list[int]],
    pos_nodes: list[int],
    neg_nodes: list[int],
    graph_title: str,
    out_path: Path,
    seed: int,
) -> None:
    counts = {cls: len(nodes) for cls, nodes in class_to_nodes.items()}
    sorted_classes = sorted(counts.keys(), key=lambda c: (-counts[c], c))

    # max 10 colors total: top 9 classes + "other"
    top_classes = sorted_classes[:9]
    class_palette = sns.color_palette("tab10", 10)

    node_class_compact = {}
    for node, cls in labels_final.items():
        node_class_compact[int(node)] = int(cls) if int(cls) in top_classes else -1

    class_color = {cls: class_palette[i] for i, cls in enumerate(top_classes)}
    class_color[-1] = class_palette[9]

    pos_layout = nx.spring_layout(G, seed=int(seed))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    ax0, ax1 = axes

    node_colors = [class_color[node_class_compact[int(node)]] for node in G.nodes()]
    node_sizes = [60 + 220 * (G.degree(int(node)) / max(1, max(dict(G.degree()).values()))) for node in G.nodes()]

    nx.draw_networkx_edges(G, pos_layout, ax=ax0, alpha=0.2, width=0.6)
    nx.draw_networkx_nodes(
        G,
        pos_layout,
        ax=ax0,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.2,
        edgecolors="#222222",
    )
    ax0.set_title(f"Final WL classes | {graph_title}")
    ax0.set_xticks([])
    ax0.set_yticks([])

    bar_classes = sorted_classes[: min(25, len(sorted_classes))]
    x = np.arange(len(bar_classes))
    heights = [counts[c] for c in bar_classes]

    pos_class_set = {labels_final[int(node)] for node in pos_nodes}
    neg_class_set = {labels_final[int(node)] for node in neg_nodes}

    bar_colors = []
    for c in bar_classes:
        in_pos = c in pos_class_set
        in_neg = c in neg_class_set
        if in_pos and in_neg:
            bar_colors.append("#8E44AD")
        elif in_pos:
            bar_colors.append("#E74C3C")
        elif in_neg:
            bar_colors.append("#3498DB")
        else:
            bar_colors.append("#95A5A6")

    ax1.bar(x, heights, color=bar_colors)
    ax1.set_xlabel("WL class index (sorted by class size)")
    ax1.set_ylabel("number of nodes")
    ax1.set_title("WL class-size distribution")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in bar_classes], rotation=90)

    legend_handles = [
        Patch(color="#E74C3C", label="contains positive zealot"),
        Patch(color="#3498DB", label="contains negative zealot"),
        Patch(color="#8E44AD", label="contains both"),
        Patch(color="#95A5A6", label="no zealot"),
    ]
    ax1.legend(handles=legend_handles, loc="best")

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_wl_iteration_snapshots(
    G: nx.Graph,
    history: list[dict[int, int]],
    graph_title: str,
    out_path: Path,
    seed: int,
) -> None:
    pos_layout = nx.spring_layout(G, seed=int(seed))
    idxs = [0, 1, 2, len(history) - 1]
    idxs = [min(i, len(history) - 1) for i in idxs]

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2))

    for ax, idx in zip(axes, idxs):
        labels = history[idx]
        classes = sorted(set(int(v) for v in labels.values()))
        cmap = sns.color_palette("tab20", max(3, min(20, len(classes))))
        class_to_color = {c: cmap[i % len(cmap)] for i, c in enumerate(classes)}

        node_colors = [class_to_color[int(labels[int(node)])] for node in G.nodes()]
        nx.draw_networkx_edges(G, pos_layout, ax=ax, alpha=0.18, width=0.6)
        nx.draw_networkx_nodes(
            G,
            pos_layout,
            ax=ax,
            node_color=node_colors,
            node_size=55,
            linewidths=0.2,
            edgecolors="#222222",
        )
        title = f"iter {idx}" if idx < len(history) - 1 else "final"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"WL color refinement snapshots | {graph_title}")
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _strategy_rows_for_graph(
    G: nx.Graph,
    graph_type: str,
    n_pos: int,
    n_neg: int,
    T: int,
    burn_in: int,
    n_runs: int,
    seed: int,
    wl_n_iter: int,
) -> list[dict]:
    strategies = ["random", "highest_degree", "wl_cover", "wl_top_class", "farthest_spread"]

    rows: list[dict] = []
    rng = np.random.default_rng(seed)

    for strategy_pos in strategies:
        res = run_two_camp_configuration(
            G=G,
            n_pos=int(n_pos),
            n_neg=int(n_neg),
            T=int(T),
            burn_in=int(burn_in),
            n_runs=int(n_runs),
            seed=int(rng.integers(np.iinfo(np.int32).max)),
            strategy_pos=strategy_pos,
            strategy_neg="random",
            threshold=0.1,
            wl_n_iter=int(wl_n_iter),
            show_progress=False,
            store_run_records=True,
        )

        agg = res["aggregate"]
        run_records = res.get("run_records", [])

        p_win = float(agg["positive_win_probability"])
        p_win_std = float(np.sqrt(max(p_win * (1.0 - p_win), 0.0) / max(1, n_runs)))

        mean_m = float(agg["mean_mean_magnetization"])
        std_m = float(agg["std_mean_magnetization"])

        rows.append(
            {
                "graph_type": graph_type,
                "strategy_pos": strategy_pos,
                "strategy_neg": "random",
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "P_plus_win": p_win,
                "P_plus_win_std": p_win_std,
                "mean_m": mean_m,
                "std_m": std_m,
                "mean_time_to_positive_crossing": float(agg.get("mean_time_to_first_crossing_above_threshold", np.nan)),
                "psi_degree_norm": float(agg.get("mean_pos_psi_degree_norm", np.nan)),
                "psi_wl_coverage": float(agg.get("mean_pos_psi_wl", np.nan)),
                "psi_dispersion": float(agg.get("mean_pos_psi_dispersion", np.nan)),
                "psi_pagerank": float(agg.get("mean_pos_psi_centrality", np.nan)),
                "run_records_count": int(len(run_records)),
            }
        )

    return rows


def _plot_strategy_comparison_bars(df: pd.DataFrame, graph_title: str, out_path: Path) -> None:
    x = np.arange(df.shape[0])
    w = 0.36

    pwin = df["P_plus_win"].to_numpy(dtype=float)
    pwin_std = df["P_plus_win_std"].to_numpy(dtype=float)

    m_norm = (df["mean_m"].to_numpy(dtype=float) + 1.0) / 2.0
    m_norm_std = np.abs(df["std_m"].to_numpy(dtype=float)) / 2.0

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    ax.bar(x - w / 2, pwin, width=w, yerr=pwin_std, capsize=3, label="P_plus_win", color="#1f77b4")
    ax.bar(x + w / 2, m_norm, width=w, yerr=m_norm_std, capsize=3, label="normalized mean_m", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(df["strategy_pos"].tolist(), rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("value")
    ax.set_title(f"Strategy comparison | {graph_title}")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_strategy_vs_features(df: pd.DataFrame, graph_title: str, out_path: Path) -> None:
    features = [
        ("psi_degree_norm", "psi_degree_norm"),
        ("psi_wl_coverage", "psi_wl_coverage"),
        ("psi_dispersion", "psi_dispersion"),
        ("psi_pagerank", "psi_pagerank"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    axes = axes.ravel()

    for ax, (col, label) in zip(axes, features):
        x = df[col].to_numpy(dtype=float)
        y = df["P_plus_win"].to_numpy(dtype=float)

        for idx, row in df.iterrows():
            strat = row["strategy_pos"]
            ax.scatter(
                float(row[col]),
                float(row["P_plus_win"]),
                color=PALETTE.get(strat, "#444444"),
                s=70,
                alpha=0.9,
            )
            ax.text(float(row[col]), float(row["P_plus_win"]) + 0.015, str(strat), fontsize=8)

        ax.set_xlabel(label)
        ax.set_ylabel("P_plus_win")
        ax.set_title(f"{label} vs P_plus_win")
        ax.grid(alpha=0.25)

    fig.suptitle(f"Structural features vs strategy performance | {graph_title}")
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_strategy_placement_graph(
    G: nx.Graph,
    strategy_pos: str,
    n_pos: int,
    n_neg: int,
    seed: int,
    wl_labels: dict[int, int],
    graph_title: str,
    out_path: Path,
) -> None:
    pos_mask, neg_mask, _ = assign_two_zealot_sets(
        G=G,
        n_pos=int(n_pos),
        n_neg=int(n_neg),
        strategy_pos=strategy_pos,
        strategy_neg="random",
        seed=int(seed),
    )

    pos_nodes = set(np.where(pos_mask)[0].astype(int).tolist())
    neg_nodes = set(np.where(neg_mask)[0].astype(int).tolist())

    pos_layout = nx.spring_layout(G, seed=int(seed))

    classes = np.asarray([int(wl_labels[int(node)]) for node in G.nodes()], dtype=int)
    cmin = int(np.min(classes)) if classes.size > 0 else 0
    cmax = int(np.max(classes)) if classes.size > 0 else 1
    denom = max(1, cmax - cmin)

    node_colors = []
    node_sizes = []
    lw = []

    deg = dict(G.degree())
    max_deg = max(deg.values()) if len(deg) > 0 else 1

    for node in G.nodes():
        node_i = int(node)
        if node_i in pos_nodes:
            node_colors.append("#E74C3C")
        elif node_i in neg_nodes:
            node_colors.append("#3498DB")
        else:
            node_colors.append("#D3D3D3")

        node_sizes.append(70 + 320 * (deg[node_i] / max_deg if max_deg > 0 else 0.0))
        lw.append(0.4 + 1.6 * ((int(wl_labels[node_i]) - cmin) / denom))

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    nx.draw_networkx_edges(G, pos_layout, ax=ax, alpha=0.22, width=0.7)
    nx.draw_networkx_nodes(
        G,
        pos_layout,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=lw,
        edgecolors="#222222",
    )

    legend_handles = [
        Patch(color="#E74C3C", label="positive zealot"),
        Patch(color="#3498DB", label="negative zealot"),
        Patch(color="#D3D3D3", label="free node"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    ax.set_title(f"{graph_title} | placement: {strategy_pos} vs random")
    ax.set_xticks([])
    ax.set_yticks([])

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dirs = setup_output_dirs(args.experiment_name)

    graph_specs = _graph_specs()
    pbar = tqdm(total=len(graph_specs), desc="exp3", disable=args.no_progress)

    n_figures = 0
    n_raw_files = 0

    for spec in graph_specs:
        gtype = str(spec["graph_type"])
        gtoken = _graph_token(spec)

        G = make_graph(**spec, seed=int(args.seed))

        wl = wl_color_refinement(G, n_iter=int(args.wl_n_iter))
        labels_final = {int(k): int(v) for k, v in wl["labels"].items()}
        history = [{int(k): int(v) for k, v in h.items()} for h in wl["history"]]
        class_to_nodes = _build_class_maps(labels_final)

        # Use one assignment for class-highlight panel.
        pos_mask, neg_mask, _ = assign_two_zealot_sets(
            G=G,
            n_pos=int(args.n_pos),
            n_neg=int(args.n_neg),
            strategy_pos="wl_cover",
            strategy_neg="random",
            seed=int(args.seed),
        )
        pos_nodes = np.where(pos_mask)[0].astype(int).tolist()
        neg_nodes = np.where(neg_mask)[0].astype(int).tolist()

        title = f"{graph_label(gtype)} N={int(spec['n'])}"

        fig3a = dirs["figures"] / f"wl_partition_evolution_{gtoken}.png"
        _plot_wl_partition_evolution(
            G=G,
            labels_final=labels_final,
            class_to_nodes=class_to_nodes,
            pos_nodes=pos_nodes,
            neg_nodes=neg_nodes,
            graph_title=title,
            out_path=fig3a,
            seed=int(args.seed),
        )
        n_figures += 1

        fig3b = dirs["figures"] / f"wl_iteration_snapshots_{gtoken}.png"
        _plot_wl_iteration_snapshots(
            G=G,
            history=history,
            graph_title=title,
            out_path=fig3b,
            seed=int(args.seed),
        )
        n_figures += 1

        rows = _strategy_rows_for_graph(
            G=G,
            graph_type=gtype,
            n_pos=int(args.n_pos),
            n_neg=int(args.n_neg),
            T=int(args.T),
            burn_in=int(args.burn_in),
            n_runs=int(args.n_runs),
            seed=int(args.seed),
            wl_n_iter=int(args.wl_n_iter),
        )
        df = pd.DataFrame(rows)

        fig3c = dirs["figures"] / f"strategy_comparison_bar_{gtoken}.png"
        _plot_strategy_comparison_bars(df=df, graph_title=title, out_path=fig3c)
        n_figures += 1

        fig3d = dirs["figures"] / f"strategy_vs_structural_features_{gtoken}.png"
        _plot_strategy_vs_features(df=df, graph_title=title, out_path=fig3d)
        n_figures += 1

        for strategy in ["random", "highest_degree", "wl_cover", "wl_top_class", "farthest_spread"]:
            fig3e = dirs["figures"] / f"wl_zealot_placement_graph_{gtoken}_{strategy}.png"
            _plot_strategy_placement_graph(
                G=G,
                strategy_pos=strategy,
                n_pos=int(args.n_pos),
                n_neg=int(args.n_neg),
                seed=int(args.seed),
                wl_labels=labels_final,
                graph_title=title,
                out_path=fig3e,
            )
            n_figures += 1

        csv_path = dirs["raw"] / f"wl_strategy_comparison_{gtoken}.csv"
        df.to_csv(csv_path, index=False)
        n_raw_files += 1

        json_path = dirs["raw"] / f"wl_features_{gtoken}.json"
        save_json(
            json_path,
            {
                "graph_spec": spec,
                "n_classes": int(wl["n_classes"]),
                "class_counts": wl["class_counts"],
                "strategy_rows": rows,
            },
        )
        n_raw_files += 1

        pbar.update(1)

    summarize_to_stdout(
        "exp3_wl_cluster_placement",
        {
            "output_base": str(dirs["base"]),
            "n_figures": int(n_figures),
            "n_raw_files": int(n_raw_files),
            "n_graphs": int(len(graph_specs)),
        },
    )


if __name__ == "__main__":
    main()

"""Experiment 1: time-series dynamics and convergence sanity checks."""

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
import seaborn as sns
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.utils_exp import (
    PALETTE,
    draw_network_opinion,
    graph_label,
    make_graph,
    run_strategy_multirun,
    save_figure,
    setup_output_dirs,
    summarize_to_stdout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=20000)
    parser.add_argument("--burn-in", type=int, default=4000)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-T", type=int, default=3000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="exp1_timeseries_and_convergence")
    return parser.parse_args()


def _graph_specs() -> list[dict]:
    return [
        {"graph_type": "fully_connected", "n": 200},
        {"graph_type": "erdos_renyi", "n": 500, "p": 0.02},
        {"graph_type": "barabasi_albert", "n": 500, "m": 3},
    ]


def _regimes() -> dict[str, tuple[int, int]]:
    return {
        "regimeA": (15, 30),
        "regimeB": (25, 25),
    }


def _strategy_pairs() -> list[tuple[str, str]]:
    return [
        ("random", "random"),
        ("highest_degree", "random"),
        ("hub_then_spread", "highest_degree"),
    ]


def _graph_param_token(spec: dict) -> str:
    g = str(spec["graph_type"])
    if g == "erdos_renyi":
        return f"p{str(float(spec['p'])).replace('.', '')}"
    if g == "barabasi_albert":
        return f"m{int(spec['m'])}"
    return ""


def _save_raw_trajectories(raw_dir: Path, fname_stem: str, payload: dict) -> None:
    npz_path = raw_dir / f"{fname_stem}.npz"
    np.savez(
        npz_path,
        magnetization=np.asarray(payload["magnetization"], dtype=float),
        positive_fraction=np.asarray(payload["positive_fraction"], dtype=float),
    )

    json_path = raw_dir / f"{fname_stem}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload["params"], f, indent=2)


def _plot_timeseries(
    series: np.ndarray,
    burn_in: int,
    title: str,
    ylabel: str,
    out_path: Path,
    hline_zero: bool = False,
) -> None:
    n_runs, T = series.shape
    t = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = sns.color_palette("husl", n_runs)
    for r in range(n_runs):
        ax.plot(t, series[r], linewidth=0.8, alpha=0.35, color=colors[r])

    mean_series = np.mean(series, axis=0)
    ax.plot(t, mean_series, linewidth=2.6, color="#1f1f1f", label="mean across seeds")

    if hline_zero:
        ax.axhline(0.0, linestyle="--", color="#333333", linewidth=1.1, alpha=0.8)

    ax.axvline(int(burn_in), linestyle="--", color="#A23B72", linewidth=1.2, alpha=0.9, label="burn-in")

    ax.set_xlabel("time step t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_convergence_comparison(
    magnetization_by_pair: dict[tuple[str, str], np.ndarray],
    graph_title: str,
    burn_in: int,
    out_path: Path,
) -> None:
    pairs = list(magnetization_by_pair.keys())
    n_pairs = len(pairs)

    y_min = min(float(np.min(magnetization_by_pair[p])) for p in pairs)
    y_max = max(float(np.max(magnetization_by_pair[p])) for p in pairs)

    fig, axes = plt.subplots(1, n_pairs, figsize=(5.8 * n_pairs, 4.6), sharey=True)
    if n_pairs == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        s_pos, s_neg = pair
        arr = magnetization_by_pair[pair]
        t = np.arange(1, arr.shape[1] + 1)

        for r in range(arr.shape[0]):
            ax.plot(t, arr[r], linewidth=0.7, alpha=0.22, color=PALETTE.get(s_pos, "#7B8EC8"))

        mean_series = np.mean(arr, axis=0)
        std_series = np.std(arr, axis=0)
        ax.plot(t, mean_series, linewidth=2.4, color=PALETTE.get(s_pos, "#7B8EC8"), label=f"{s_pos} vs {s_neg}")
        ax.fill_between(t, mean_series - std_series, mean_series + std_series, alpha=0.15, color=PALETTE.get(s_pos, "#7B8EC8"))

        ax.axhline(0.0, linestyle="--", color="#333333", linewidth=1.0, alpha=0.8)
        ax.axvline(int(burn_in), linestyle="--", color="#A23B72", linewidth=1.0, alpha=0.8)

        ax.set_ylim(y_min - 0.05, y_max + 0.05)
        ax.set_xlabel("time step t")
        ax.set_title(f"{s_pos} vs {s_neg}")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("m(t)")
    fig.suptitle(f"Convergence comparison across strategies | {graph_title}")
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_small_network_snapshots(
    figures_dir: Path,
    seed: int,
    T: int,
) -> Path:
    graph_type = "erdos_renyi"
    G = make_graph(graph_type, n=50, p=0.1, seed=seed)

    n_pos_small = 6
    n_neg_small = 10
    strategy_pos = "highest_degree"
    strategy_neg = "random"

    out = run_strategy_multirun(
        G=G,
        n_pos=n_pos_small,
        n_neg=n_neg_small,
        strategy_pos=strategy_pos,
        strategy_neg=strategy_neg,
        T=int(T),
        burn_in=max(1, int(0.2 * T)),
        n_runs=1,
        base_seed=seed,
        record=True,
    )

    sim = out["sample_run"]
    assert sim is not None

    trajectory = np.asarray(sim["trajectory"], dtype=np.int8)
    initial = np.asarray(sim["initial_states"], dtype=np.int8)
    m_series = np.asarray(sim["magnetization"], dtype=float)

    t_points = [0, int(T / 5), int(2 * T / 5), int(3 * T / 5), int(4 * T / 5), int(T)]
    t_points = [max(0, min(int(T), int(v))) for v in t_points]

    pos_layout = __import__("networkx").spring_layout(G, seed=int(seed))

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0))
    axes = axes.ravel()

    z_pos = sim["pos_nodes"]
    z_neg = sim["neg_nodes"]

    for ax, t_v in zip(axes, t_points):
        if t_v == 0:
            state = initial
            m_t = float(np.mean(initial))
        else:
            idx = int(t_v - 1)
            state = trajectory[idx]
            m_t = float(m_series[idx])

        draw_network_opinion(
            G=G,
            zealot_plus=z_pos,
            zealot_minus=z_neg,
            state=state,
            ax=ax,
            title=f"t={t_v}, m={m_t:.2f}",
            seed=seed,
            pos=pos_layout,
        )

    fig.suptitle("Small-network snapshots (ER N=50, p=0.1)")

    out_path = figures_dir / f"network_snapshots_small_N50_seed{seed}.png"
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    dirs = setup_output_dirs(args.experiment_name)

    graph_specs = _graph_specs()
    regimes = _regimes()
    strategy_pairs = _strategy_pairs()

    total_jobs = len(graph_specs) * len(regimes) * len(strategy_pairs)
    pbar = tqdm(total=total_jobs, desc="exp1", disable=args.no_progress)

    n_figures = 0
    n_raw_files = 0

    for gspec in graph_specs:
        gtype = str(gspec["graph_type"])
        n = int(gspec["n"])
        gparam_token = _graph_param_token(gspec)

        G = make_graph(**gspec, seed=args.seed)

        for regime_name, (n_pos, n_neg) in regimes.items():
            mag_for_panel: dict[tuple[str, str], np.ndarray] = {}

            for strategy_pos, strategy_neg in strategy_pairs:
                run_out = run_strategy_multirun(
                    G=G,
                    n_pos=n_pos,
                    n_neg=n_neg,
                    strategy_pos=strategy_pos,
                    strategy_neg=strategy_neg,
                    T=int(args.T),
                    burn_in=int(args.burn_in),
                    n_runs=int(args.n_runs),
                    base_seed=int(args.seed),
                    threshold=0.0,
                    record=False,
                )

                mag = np.asarray(run_out["magnetization"], dtype=float)
                pfrac = np.asarray(run_out["positive_fraction"], dtype=float)
                mag_for_panel[(strategy_pos, strategy_neg)] = mag

                key_params = f"npos{n_pos}_nneg{n_neg}_{regime_name}"
                graph_token = f"{gtype}_N{n}"
                if gparam_token:
                    graph_token = f"{graph_token}_{gparam_token}"

                pair_token = f"{strategy_pos}_vs_{strategy_neg}"

                stem = f"trajectories_{graph_token}_{key_params}_{pair_token}_allseeds"
                _save_raw_trajectories(
                    dirs["raw"],
                    fname_stem=stem,
                    payload={
                        "magnetization": mag,
                        "positive_fraction": pfrac,
                        "params": {
                            "graph_spec": gspec,
                            "regime": regime_name,
                            "n_pos": int(n_pos),
                            "n_neg": int(n_neg),
                            "strategy_pos": strategy_pos,
                            "strategy_neg": strategy_neg,
                            "T": int(args.T),
                            "burn_in": int(args.burn_in),
                            "n_runs": int(args.n_runs),
                            "seed": int(args.seed),
                            "mean_m": float(run_out["mean_m"]),
                            "P_plus_win": float(run_out["P_plus_win"]),
                        },
                    },
                )
                n_raw_files += 2

                title_base = (
                    f"{graph_label(gtype)} N={n}"
                    + (f" ({gparam_token})" if gparam_token else "")
                    + f" | n+={n_pos} n-={n_neg} | {strategy_pos} vs {strategy_neg}"
                )

                fig1a_name = (
                    f"magnetization_timeseries_{graph_token}_{key_params}_{pair_token}.png"
                )
                _plot_timeseries(
                    series=mag,
                    burn_in=int(args.burn_in),
                    title=title_base,
                    ylabel="m(t)",
                    out_path=dirs["figures"] / fig1a_name,
                    hline_zero=True,
                )
                n_figures += 1

                fig1b_name = (
                    f"positive_fraction_evolution_{graph_token}_{key_params}_{pair_token}.png"
                )
                _plot_timeseries(
                    series=pfrac,
                    burn_in=int(args.burn_in),
                    title=title_base,
                    ylabel="fraction at +1",
                    out_path=dirs["figures"] / fig1b_name,
                    hline_zero=False,
                )
                n_figures += 1

                pbar.update(1)

            graph_token = f"{gtype}_N{n}"
            if gparam_token:
                graph_token = f"{graph_token}_{gparam_token}"

            comp_name = (
                f"convergence_comparison_all_strategies_{graph_token}_"
                f"npos{n_pos}_nneg{n_neg}_{regime_name}.png"
            )
            _plot_convergence_comparison(
                magnetization_by_pair=mag_for_panel,
                graph_title=f"{graph_label(gtype)} N={n} | n+={n_pos} n-={n_neg}",
                burn_in=int(args.burn_in),
                out_path=dirs["figures"] / comp_name,
            )
            n_figures += 1

    snap_path = _plot_small_network_snapshots(
        figures_dir=dirs["figures"],
        seed=int(args.seed),
        T=int(args.small_T),
    )
    n_figures += 1

    summarize_to_stdout(
        "exp1_timeseries_and_convergence",
        {
            "output_base": str(dirs["base"]),
            "n_figures": int(n_figures),
            "n_raw_files": int(n_raw_files),
            "small_snapshot": str(snap_path),
            "T": int(args.T),
            "n_runs": int(args.n_runs),
        },
    )


if __name__ == "__main__":
    main()

"""Multi-N symbolic regression for convergence time tau(N, rho) on configurable graphs.

Protocol:
1) For each (N, rho), run multiple seeds with early stopping up to T_max.
2) Aggregate mean stopping time per (N, rho) and fit PySR on aggregated points.
3) Evaluate model on held-out N values and compare prediction vs real trials.
4) Save all datasets/models as JSON and figures for held-out comparisons.
"""

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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.convergence_time import run_until_convergence, summarize_stopping_times
from src.graph_generation import (
    generate_barabasi_albert,
    generate_erdos_renyi,
    generate_fully_connected,
)
from src.symbolic_regression import (
    fit_pysr_multivariate_symbolic_regression,
    predict_pysr_multivariate_model,
)
from src.utils import ensure_dir, setup_logging

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


def parse_operator_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--graph-type",
        choices=["fully_connected", "erdos_renyi", "barabasi_albert"],
        default="fully_connected",
        help="Graph family used in this experiment.",
    )
    parser.add_argument(
        "--erdos-p",
        type=float,
        default=0.01,
        help="Erdos-Renyi edge probability p (used if graph-type=erdos_renyi).",
    )
    parser.add_argument(
        "--barabasi-m",
        type=int,
        default=3,
        help="Barabasi-Albert attachment parameter m (used if graph-type=barabasi_albert).",
    )

    parser.add_argument(
        "--n-values",
        type=str,
        default="100,200,300,400,500",
        help="Comma-separated training N values.",
    )
    parser.add_argument(
        "--rho-values",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        help="Comma-separated rho values.",
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of seed runs per training (N, rho).",
    )
    parser.add_argument(
        "--test-n-values",
        type=str,
        default="150,250",
        help="Comma-separated held-out N values for evaluation plots.",
    )
    parser.add_argument(
        "--test-seeds",
        type=int,
        default=5,
        help="Number of seed runs per held-out (N, rho).",
    )

    parser.add_argument(
        "--T-max",
        type=int,
        default=100000,
        help="Maximum horizon (simulation stops earlier on convergence).",
    )
    parser.add_argument(
        "--stable-window",
        type=int,
        default=500,
        help="Consecutive unchanged-magnetization steps needed for convergence.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help="Tolerance for unchanged magnetization steps.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Global seed.")

    parser.add_argument("--pysr-niterations", type=int, default=400)
    parser.add_argument("--pysr-population-size", type=int, default=50)
    parser.add_argument("--pysr-maxsize", type=int, default=20)
    parser.add_argument("--pysr-binary-operators", type=str, default="+,-,*,/")
    parser.add_argument("--pysr-unary-operators", type=str, default="log,exp")
    parser.add_argument(
        "--pysr-model-selection",
        choices=["best", "accuracy", "score"],
        default="best",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "fc_symbolic_pysr_multin",
        help="Output directory with raw/ and figures/.",
    )
    parser.add_argument("--no-progress", action="store_true")

    return parser.parse_args()


def run_grid(
    n_values: list[int],
    rho_values: list[float],
    n_seeds: int,
    T_max: int,
    stable_window: int,
    tol: float,
    rng: np.random.Generator,
    show_progress: bool,
    phase: str,
    graph_type: str,
    erdos_p: float,
    barabasi_m: int,
) -> tuple[list[dict], list[dict]]:
    """Run experiments on a (N, rho) grid.

    Returns
    -------
    seed_records, aggregated_records
    """
    seed_records: list[dict] = []
    aggregated_records: list[dict] = []

    outer = tqdm(n_values, desc=f"{phase} N", disable=not show_progress)
    for N in outer:
        graph_seed = None
        if graph_type == "fully_connected":
            G = generate_fully_connected(N)
        elif graph_type == "erdos_renyi":
            graph_seed = int(rng.integers(np.iinfo(np.int32).max))
            G = generate_erdos_renyi(n=N, p=float(erdos_p), seed=graph_seed)
        elif graph_type == "barabasi_albert":
            graph_seed = int(rng.integers(np.iinfo(np.int32).max))
            G = generate_barabasi_albert(n=N, m=int(barabasi_m), seed=graph_seed)
        else:
            raise ValueError(f"Unsupported graph_type={graph_type}")

        mid = tqdm(rho_values, desc=f"{phase} rho (N={N})", leave=False, disable=not show_progress)
        for rho in mid:
            stopping_times: list[int | None] = []

            inner = tqdm(
                range(n_seeds),
                desc=f"{phase} seeds N={N} rho={rho:.3f}",
                leave=False,
                disable=not show_progress,
            )
            for rep in inner:
                run_seed = int(rng.integers(np.iinfo(np.int32).max))
                out = run_until_convergence(
                    G=G,
                    rho=float(rho),
                    T_max=int(T_max),
                    seed=run_seed,
                    tol=float(tol),
                    stable_window=int(stable_window),
                )

                stopping_times.append(out["stopping_time"])
                seed_records.append(
                    {
                        "phase": phase,
                        "graph_type": graph_type,
                        "graph_seed": graph_seed,
                        "N": int(N),
                        "rho": float(rho),
                        "replicate": int(rep),
                        "erdos_p": float(erdos_p) if graph_type == "erdos_renyi" else None,
                        "barabasi_m": int(barabasi_m) if graph_type == "barabasi_albert" else None,
                        **out,
                    }
                )

            summary = summarize_stopping_times(stopping_times)
            aggregated_records.append(
                {
                    "phase": phase,
                    "graph_type": graph_type,
                    "graph_seed": graph_seed,
                    "N": int(N),
                    "rho": float(rho),
                    "erdos_p": float(erdos_p) if graph_type == "erdos_renyi" else None,
                    "barabasi_m": int(barabasi_m) if graph_type == "barabasi_albert" else None,
                    "stopping_times": [int(x) if x is not None else None for x in stopping_times],
                    **summary,
                }
            )

    return seed_records, aggregated_records


def main() -> None:
    args = parse_args()
    setup_logging()

    n_values = parse_int_list(args.n_values)
    rho_values = parse_float_list(args.rho_values)
    test_n_values = parse_int_list(args.test_n_values)

    if args.n_seeds <= 0 or args.test_seeds <= 0:
        raise ValueError("n_seeds and test_seeds must be strictly positive.")
    if args.T_max <= 0:
        raise ValueError("T-max must be strictly positive.")
    if args.graph_type == "erdos_renyi" and not (0.0 < args.erdos_p <= 1.0):
        raise ValueError("For erdos_renyi, --erdos-p must be in (0,1].")
    if args.graph_type == "barabasi_albert":
        if args.barabasi_m <= 0:
            raise ValueError("For barabasi_albert, --barabasi-m must be strictly positive.")
        min_n = min(n_values + test_n_values)
        if args.barabasi_m >= min_n:
            raise ValueError("--barabasi-m must be smaller than all N values.")

    for rho in rho_values:
        if not (0.0 < rho <= 1.0):
            raise ValueError("All rho values must be in (0, 1].")

    raw_dir = ensure_dir(Path(args.results_dir) / "raw")
    fig_dir = ensure_dir(Path(args.results_dir) / "figures")

    rng = np.random.default_rng(args.seed)

    LOGGER.info("Graph type: %s", args.graph_type)
    if args.graph_type == "erdos_renyi":
        LOGGER.info("Erdos-Renyi p: %.6f", args.erdos_p)
    if args.graph_type == "barabasi_albert":
        LOGGER.info("Barabasi-Albert m: %d", args.barabasi_m)
    LOGGER.info("Training grid N values: %s", n_values)
    LOGGER.info("Rho values: %s", rho_values)
    LOGGER.info("Training seeds per (N,rho): %d", args.n_seeds)
    LOGGER.info("T_max=%d, stable_window=%d, tol=%g", args.T_max, args.stable_window, args.tol)

    train_seed_records, train_agg_records = run_grid(
        n_values=n_values,
        rho_values=rho_values,
        n_seeds=int(args.n_seeds),
        T_max=int(args.T_max),
        stable_window=int(args.stable_window),
        tol=float(args.tol),
        rng=rng,
        show_progress=not args.no_progress,
        phase="train",
        graph_type=str(args.graph_type),
        erdos_p=float(args.erdos_p),
        barabasi_m=int(args.barabasi_m),
    )

    training_payload = {
        "experiment": f"{args.graph_type}_tau_multin_training",
        "config": {
            "graph_type": str(args.graph_type),
            "erdos_p": float(args.erdos_p) if args.graph_type == "erdos_renyi" else None,
            "barabasi_m": int(args.barabasi_m) if args.graph_type == "barabasi_albert" else None,
            "n_values": n_values,
            "rho_values": rho_values,
            "n_seeds": int(args.n_seeds),
            "T_max": int(args.T_max),
            "stable_window": int(args.stable_window),
            "tol": float(args.tol),
            "seed": int(args.seed),
        },
        "seed_records": train_seed_records,
        "aggregated_records": train_agg_records,
    }

    training_path = raw_dir / f"{args.graph_type}_tau_multin_training_dataset.json"
    save_json(training_path, training_payload)
    LOGGER.info("Saved training dataset to %s", training_path)

    # Build PySR training matrix from aggregated means.
    X_train = []
    y_train = []
    for rec in train_agg_records:
        if rec["mean_stopping_time"] is None:
            continue
        X_train.append([float(rec["rho"]), float(rec["N"])])
        y_train.append(float(rec["mean_stopping_time"]))

    if len(X_train) < 5:
        raise RuntimeError("Not enough converged training points for PySR fitting.")

    X_train_arr = np.asarray(X_train, dtype=float)
    y_train_arr = np.asarray(y_train, dtype=float)

    model_summary, pysr_model = fit_pysr_multivariate_symbolic_regression(
        features=X_train_arr,
        target=y_train_arr,
        feature_names=["rho", "N"],
        seed=int(args.seed),
        niterations=int(args.pysr_niterations),
        population_size=int(args.pysr_population_size),
        maxsize=int(args.pysr_maxsize),
        binary_operators=parse_operator_list(args.pysr_binary_operators),
        unary_operators=parse_operator_list(args.pysr_unary_operators),
        model_selection=str(args.pysr_model_selection),
    )

    y_train_pred = predict_pysr_multivariate_model(pysr_model, X_train_arr)

    LOGGER.info("Selected PySR expression: %s", model_summary["expression"])
    LOGGER.info(
        "Train fit metrics: RMSE=%.4f, R2=%.4f",
        model_summary["metrics"]["rmse"],
        model_summary["metrics"]["r2"],
    )

    model_payload = {
        "regression_method": "pysr_multivariate",
        **model_summary,
        "graph_type": str(args.graph_type),
        "graph_params": {
            "erdos_p": float(args.erdos_p) if args.graph_type == "erdos_renyi" else None,
            "barabasi_m": int(args.barabasi_m) if args.graph_type == "barabasi_albert" else None,
        },
        "fit_data": {
            "rho": X_train_arr[:, 0].astype(float).tolist(),
            "N": X_train_arr[:, 1].astype(float).tolist(),
            "mean_tau": y_train_arr.astype(float).tolist(),
            "predicted_tau": y_train_pred.astype(float).tolist(),
        },
    }

    model_path = raw_dir / f"{args.graph_type}_tau_multin_pysr_model.json"
    save_json(model_path, model_payload)
    LOGGER.info("Saved PySR model to %s", model_path)

    # Evaluation on held-out N values.
    LOGGER.info("Evaluation N values: %s", test_n_values)
    LOGGER.info("Evaluation seeds per (N,rho): %d", args.test_seeds)

    eval_seed_records, eval_agg_records = run_grid(
        n_values=test_n_values,
        rho_values=rho_values,
        n_seeds=int(args.test_seeds),
        T_max=int(args.T_max),
        stable_window=int(args.stable_window),
        tol=float(args.tol),
        rng=rng,
        show_progress=not args.no_progress,
        phase="eval",
        graph_type=str(args.graph_type),
        erdos_p=float(args.erdos_p),
        barabasi_m=int(args.barabasi_m),
    )

    # Predictions for evaluation grid and per-N figures.
    eval_predictions = []
    for N in test_n_values:
        rows = [r for r in eval_agg_records if int(r["N"]) == int(N)]
        rows = sorted(rows, key=lambda d: float(d["rho"]))

        rho_plot = np.array([float(r["rho"]) for r in rows], dtype=float)
        mean_tau = np.array(
            [float(r["mean_stopping_time"]) if r["mean_stopping_time"] is not None else np.nan for r in rows],
            dtype=float,
        )
        std_tau = np.array(
            [float(r["std_stopping_time"]) if r["std_stopping_time"] is not None else 0.0 for r in rows],
            dtype=float,
        )

        X_eval = np.column_stack([rho_plot, np.full_like(rho_plot, float(N))])
        pred_tau = predict_pysr_multivariate_model(pysr_model, X_eval)

        for rho_v, pred_v in zip(rho_plot, pred_tau):
            eval_predictions.append(
                {
                    "N": int(N),
                    "rho": float(rho_v),
                    "predicted_tau": float(pred_v),
                }
            )

        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.errorbar(
            rho_plot,
            mean_tau,
            yerr=std_tau,
            fmt="o",
            capsize=3,
            label="Real trials (mean ± std)",
        )
        ax.plot(rho_plot, pred_tau, linewidth=2.0, label="PySR prediction")
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"Convergence time $\tau$")
        ax.set_title(f"Convergence Time vs Rho (N={N})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        fig_path = fig_dir / f"rho_vs_tau_real_vs_pysr_{args.graph_type}_N_{int(N)}.png"
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        LOGGER.info("Saved evaluation figure for N=%d to %s", N, fig_path)

    eval_payload = {
        "experiment": f"{args.graph_type}_tau_multin_evaluation",
        "config": {
            "graph_type": str(args.graph_type),
            "erdos_p": float(args.erdos_p) if args.graph_type == "erdos_renyi" else None,
            "barabasi_m": int(args.barabasi_m) if args.graph_type == "barabasi_albert" else None,
            "test_n_values": test_n_values,
            "rho_values": rho_values,
            "test_seeds": int(args.test_seeds),
            "T_max": int(args.T_max),
            "stable_window": int(args.stable_window),
            "tol": float(args.tol),
        },
        "seed_records": eval_seed_records,
        "aggregated_records": eval_agg_records,
        "predictions": eval_predictions,
    }

    eval_path = raw_dir / f"{args.graph_type}_tau_multin_eval_dataset.json"
    save_json(eval_path, eval_payload)
    LOGGER.info("Saved evaluation dataset to %s", eval_path)


if __name__ == "__main__":
    main()

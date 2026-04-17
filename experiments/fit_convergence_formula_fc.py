"""Estimate convergence-time law tau(rho) on fully connected graphs.

Workflow:
1) Run repeated simulations for each rho.
2) Estimate stopping times from m(t) trajectories.
3) Fit a symbolic-regression style formula tau(rho).
4) Save JSON outputs and figure comparing real data vs prediction.
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

from src.convergence_time import estimate_convergence_time, summarize_stopping_times
from src.graph_generation import generate_fully_connected
from src.simulation import run_simulation
from src.symbolic_regression import (
    fit_pysr_symbolic_regression,
    fit_sparse_symbolic_regression,
    predict_pysr_model,
    predict_symbolic_model,
)
from src.utils import ensure_dir, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--n", type=int, default=400, help="Number of nodes.")
    parser.add_argument("--T", type=int, default=80000, help="Simulation horizon in steps.")
    parser.add_argument("--n-runs", type=int, default=6, help="Independent runs per rho.")

    parser.add_argument("--rho-min", type=float, default=0.05, help="Minimum rho (inclusive).")
    parser.add_argument("--rho-max", type=float, default=0.50, help="Maximum rho (inclusive).")
    parser.add_argument("--rho-step", type=float, default=0.05, help="Rho grid step.")
    parser.add_argument(
        "--rho-values",
        type=str,
        default=None,
        help="Optional comma-separated list of rho values, overrides min/max/step.",
    )

    parser.add_argument(
        "--stable-window",
        type=int,
        default=500,
        help="Minimum terminal plateau length for convergence-time detection.",
    )
    parser.add_argument("--tol", type=float, default=1e-12, help="Tolerance for plateau detection.")

    parser.add_argument(
        "--max-terms",
        type=int,
        default=3,
        help="Maximum number of symbolic terms (including constant).",
    )
    parser.add_argument(
        "--criterion",
        choices=["bic", "aic"],
        default="bic",
        help="Model-selection criterion for symbolic regression.",
    )
    parser.add_argument(
        "--regression-method",
        choices=["basis", "pysr"],
        default="basis",
        help="Symbolic regression backend: basis (fast) or pysr (full symbolic search).",
    )
    parser.add_argument(
        "--pysr-niterations",
        type=int,
        default=400,
        help="PySR iterations (used only when --regression-method pysr).",
    )
    parser.add_argument(
        "--pysr-population-size",
        type=int,
        default=50,
        help="PySR population size (used only when --regression-method pysr).",
    )
    parser.add_argument(
        "--pysr-maxsize",
        type=int,
        default=20,
        help="PySR max expression size (used only when --regression-method pysr).",
    )
    parser.add_argument(
        "--pysr-binary-operators",
        type=str,
        default="+,-,*,/",
        help="Comma-separated PySR binary operators.",
    )
    parser.add_argument(
        "--pysr-unary-operators",
        type=str,
        default="log,exp",
        help="Comma-separated PySR unary operators.",
    )
    parser.add_argument(
        "--pysr-model-selection",
        choices=["best", "accuracy", "score"],
        default="best",
        help="PySR model selection strategy.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "fc_symbolic_regression",
        help="Output directory where raw/ and figures/ are written.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm bars.")

    return parser.parse_args()


def parse_rho_grid(args: argparse.Namespace) -> np.ndarray:
    if args.rho_values is not None:
        vals = [float(x.strip()) for x in args.rho_values.split(",") if x.strip()]
        rho = np.array(vals, dtype=float)
    else:
        if args.rho_step <= 0.0:
            raise ValueError("rho_step must be strictly positive.")
        n = int(np.floor((args.rho_max - args.rho_min) / args.rho_step + 0.5)) + 1
        rho = args.rho_min + args.rho_step * np.arange(n, dtype=float)

    if np.any(rho <= 0.0):
        raise ValueError("rho values must be strictly positive for this regression task.")
    if np.any(rho > 1.0):
        raise ValueError("rho values must be <= 1.")

    return np.unique(np.round(rho, 10))


def run_dataset(args: argparse.Namespace, rho_values: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    G = generate_fully_connected(args.n)

    records: list[dict] = []

    outer = tqdm(rho_values, desc="rho grid", disable=args.no_progress)
    for rho in outer:
        stop_times: list[int | None] = []
        final_m: list[float] = []

        inner = tqdm(
            range(args.n_runs),
            leave=False,
            desc=f"runs rho={rho:.3f}",
            disable=args.no_progress,
        )
        for _ in inner:
            run_seed = int(rng.integers(np.iinfo(np.int32).max))
            sim = run_simulation(
                G=G,
                rho=float(rho),
                T=int(args.T),
                burn_in=0,
                seed=run_seed,
                record=False,
                show_progress=False,
            )
            series = sim["magnetization"]
            tau = estimate_convergence_time(
                series,
                tol=float(args.tol),
                min_plateau=int(args.stable_window),
            )
            stop_times.append(tau)
            final_m.append(float(series[-1]))

        summary = summarize_stopping_times(stop_times)
        record = {
            "rho": float(rho),
            "stopping_times": [int(x) if x is not None else None for x in stop_times],
            "final_magnetization": final_m,
            **summary,
        }
        records.append(record)

    return records


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def parse_operator_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    setup_logging()

    rho_values = parse_rho_grid(args)
    raw_dir = ensure_dir(Path(args.results_dir) / "raw")
    fig_dir = ensure_dir(Path(args.results_dir) / "figures")

    LOGGER.info("Running fully-connected convergence dataset generation...")
    LOGGER.info("n=%d, T=%d, n_runs=%d", args.n, args.T, args.n_runs)
    LOGGER.info("rho grid: %s", rho_values.tolist())

    records = run_dataset(args, rho_values)

    dataset_payload = {
        "experiment": "fully_connected_convergence_time",
        "config": {
            "n": int(args.n),
            "T": int(args.T),
            "n_runs": int(args.n_runs),
            "stable_window": int(args.stable_window),
            "tol": float(args.tol),
            "seed": int(args.seed),
            "rho_values": rho_values.tolist(),
        },
        "records": records,
    }

    dataset_path = raw_dir / "fc_convergence_dataset.json"
    save_json(dataset_path, dataset_payload)
    LOGGER.info("Saved dataset JSON to %s", dataset_path)

    # Prepare regression data from converged points.
    rho_fit = []
    tau_fit = []
    tau_std = []
    for rec in records:
        if rec["mean_stopping_time"] is not None:
            rho_fit.append(float(rec["rho"]))
            tau_fit.append(float(rec["mean_stopping_time"]))
            tau_std.append(float(rec["std_stopping_time"]) if rec["std_stopping_time"] is not None else 0.0)

    if len(rho_fit) < 2:
        raise RuntimeError("Not enough converged rho points for symbolic regression.")

    rho_fit_arr = np.asarray(rho_fit, dtype=float)
    tau_fit_arr = np.asarray(tau_fit, dtype=float)
    tau_std_arr = np.asarray(tau_std, dtype=float)

    # Dense curve for visualization.
    rho_dense = np.linspace(float(np.min(rho_fit_arr)), float(np.max(rho_fit_arr)), 400)

    if args.regression_method == "basis":
        model = fit_sparse_symbolic_regression(
            rho=rho_fit_arr,
            target=tau_fit_arr,
            max_terms=int(args.max_terms),
            criterion=str(args.criterion),
        )
        pred_fit = predict_symbolic_model(model, rho_fit_arr)
        tau_dense = predict_symbolic_model(model, rho_dense)

        LOGGER.info("Selected symbolic model: %s", model["expression"])
        LOGGER.info(
            "Fit metrics: RMSE=%.4f, R2=%.4f, %s=%.4f",
            model["metrics"]["rmse"],
            model["metrics"]["r2"],
            args.criterion,
            model["selection_score"],
        )

        model_payload = {
            "regression_method": "basis",
            "model_type": "sparse_basis_symbolic_regression",
            "expression": model["expression"],
            "terms": model["terms"],
            "terms_human": model["terms_human"],
            "coefficients": model["coefficients"],
            "metrics": model["metrics"],
            "selection_score": model["selection_score"],
            "criterion": model["criterion"],
            "fit_data": {
                "rho": rho_fit_arr.tolist(),
                "mean_tau": tau_fit_arr.tolist(),
                "std_tau": tau_std_arr.tolist(),
                "predicted_tau": pred_fit.astype(float).tolist(),
            },
        }
    else:
        binary_ops = parse_operator_list(args.pysr_binary_operators)
        unary_ops = parse_operator_list(args.pysr_unary_operators)

        model, pysr_model = fit_pysr_symbolic_regression(
            rho=rho_fit_arr,
            target=tau_fit_arr,
            seed=int(args.seed),
            niterations=int(args.pysr_niterations),
            population_size=int(args.pysr_population_size),
            maxsize=int(args.pysr_maxsize),
            binary_operators=binary_ops,
            unary_operators=unary_ops,
            model_selection=str(args.pysr_model_selection),
        )
        pred_fit = predict_pysr_model(pysr_model, rho_fit_arr)
        tau_dense = predict_pysr_model(pysr_model, rho_dense)

        LOGGER.info("Selected PySR model: %s", model["expression"])
        LOGGER.info(
            "PySR fit metrics: RMSE=%.4f, R2=%.4f",
            model["metrics"]["rmse"],
            model["metrics"]["r2"],
        )

        model_payload = {
            "regression_method": "pysr",
            **model,
            "fit_data": {
                "rho": rho_fit_arr.tolist(),
                "mean_tau": tau_fit_arr.tolist(),
                "std_tau": tau_std_arr.tolist(),
                "predicted_tau": pred_fit.astype(float).tolist(),
            },
        }

    model_path = raw_dir / "fc_symbolic_model.json"
    save_json(model_path, model_payload)
    LOGGER.info("Saved symbolic model JSON to %s", model_path)

    # Figure: real data vs symbolic prediction.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    ax.errorbar(
        rho_fit_arr,
        tau_fit_arr,
        yerr=tau_std_arr,
        fmt="o",
        capsize=3,
        label="Measured mean stopping time",
    )
    ax.plot(rho_dense, tau_dense, linewidth=2.0, label="Symbolic prediction")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"Stopping time $\tau$")
    ax.set_title("Fully Connected Graph: Data vs Symbolic Fit")
    ax.grid(alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    ax2.scatter(tau_fit_arr, pred_fit)
    min_v = float(min(np.min(tau_fit_arr), np.min(pred_fit)))
    max_v = float(max(np.max(tau_fit_arr), np.max(pred_fit)))
    ax2.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="gray")
    ax2.set_xlabel("Measured mean stopping time")
    ax2.set_ylabel("Predicted stopping time")
    ax2.set_title("Prediction Quality")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = fig_dir / "fc_real_vs_symbolic_prediction.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    LOGGER.info("Saved comparison figure to %s", fig_path)


if __name__ == "__main__":
    main()

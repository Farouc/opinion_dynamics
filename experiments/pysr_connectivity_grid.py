"""Run connectivity-parameter grids for tau(N, rho) symbolic regression.

This script orchestrates repeated calls to `experiments/pysr_tau_multin.py`:
- Erdos-Renyi grid over p
- Barabasi-Albert grid over m

For each setting it saves full outputs in a dedicated subfolder and then builds a
summary JSON/CSV collecting expression and fit metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

LOGGER = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_float_list(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("Expected at least one float value.")
    return sorted(set(vals))


def parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("Expected at least one int value.")
    return sorted(set(vals))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slug_float(x: float, ndigits: int = 3) -> str:
    return f"{x:.{ndigits}f}".replace(".", "p")


def run_one_setting(
    graph_type: str,
    param_name: str,
    param_value: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Run one graph-parameter setting by calling pysr_tau_multin.py."""
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "pysr_tau_multin.py"),
        "--graph-type",
        graph_type,
        f"--{param_name}",
        str(param_value),
        "--n-values",
        args.n_values,
        "--rho-values",
        args.rho_values,
        "--n-seeds",
        str(args.n_seeds),
        "--test-n-values",
        args.test_n_values,
        "--test-seeds",
        str(args.test_seeds),
        "--T-max",
        str(args.T_max),
        "--stable-window",
        str(args.stable_window),
        "--tol",
        str(args.tol),
        "--seed",
        str(args.seed),
        "--pysr-niterations",
        str(args.pysr_niterations),
        "--pysr-population-size",
        str(args.pysr_population_size),
        "--pysr-maxsize",
        str(args.pysr_maxsize),
        "--pysr-binary-operators",
        args.pysr_binary_operators,
        "--pysr-unary-operators",
        args.pysr_unary_operators,
        "--pysr-model-selection",
        args.pysr_model_selection,
        "--results-dir",
        str(output_dir),
    ]
    if args.no_progress:
        cmd.append("--no-progress")

    env = os.environ.copy()
    env.setdefault("PYTHON_JULIAPKG_PROJECT", "/tmp/pyjuliapkg_project")
    env.setdefault("JULIA_DEPOT_PATH", "/tmp/julia_depot")

    LOGGER.info(
        "Running %s with %s=%s -> %s",
        graph_type,
        param_name,
        param_value,
        output_dir,
    )
    subprocess.run(cmd, env=env, check=True)


def load_model_summary(model_json: Path) -> dict:
    with model_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    return {
        "expression": data.get("expression"),
        "r2": metrics.get("r2"),
        "rmse": metrics.get("rmse"),
        "aic": metrics.get("aic"),
        "bic": metrics.get("bic"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--run-erdos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-barabasi", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--erdos-p-values", type=str, default="0.005,0.01,0.02")
    parser.add_argument("--barabasi-m-values", type=str, default="2,3,4,5")

    parser.add_argument("--n-values", type=str, default="100,200,300,400,500")
    parser.add_argument("--rho-values", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--test-n-values", type=str, default="150,250")
    parser.add_argument("--test-seeds", type=int, default=5)

    parser.add_argument("--T-max", type=int, default=100000)
    parser.add_argument("--stable-window", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pysr-niterations", type=int, default=400)
    parser.add_argument("--pysr-population-size", type=int, default=50)
    parser.add_argument("--pysr-maxsize", type=int, default=20)
    parser.add_argument("--pysr-binary-operators", type=str, default="+,-,*,/")
    parser.add_argument("--pysr-unary-operators", type=str, default="log,exp")
    parser.add_argument("--pysr-model-selection", choices=["best", "accuracy", "score"], default="best")

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "connectivity_grid",
    )
    parser.add_argument("--no-progress", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    if not args.run_erdos and not args.run_barabasi:
        raise ValueError("At least one of --run-erdos or --run-barabasi must be enabled.")

    ensure_dir(args.results_dir)
    summary_rows: list[dict] = []

    jobs: list[tuple[str, str, str, Path, str, str]] = []

    if args.run_erdos:
        p_values = parse_float_list(args.erdos_p_values)
        for p in p_values:
            tag = f"p_{slug_float(p, ndigits=4)}"
            out = args.results_dir / "erdos_renyi" / tag
            model_path = out / "raw" / "erdos_renyi_tau_multin_pysr_model.json"
            jobs.append(("erdos_renyi", "erdos-p", str(p), out, str(p), str(model_path)))

    if args.run_barabasi:
        m_values = parse_int_list(args.barabasi_m_values)
        for m in m_values:
            tag = f"m_{m}"
            out = args.results_dir / "barabasi_albert" / tag
            model_path = out / "raw" / "barabasi_albert_tau_multin_pysr_model.json"
            jobs.append(("barabasi_albert", "barabasi-m", str(m), out, str(m), str(model_path)))

    iterator = tqdm(jobs, desc="connectivity grid", disable=args.no_progress)
    for graph_type, param_flag, param_value, out_dir, param_label, model_path_str in iterator:
        run_one_setting(
            graph_type=graph_type,
            param_name=param_flag,
            param_value=param_value,
            output_dir=out_dir,
            args=args,
        )

        model_summary = load_model_summary(Path(model_path_str))
        summary_rows.append(
            {
                "graph_type": graph_type,
                "param_name": "p" if graph_type == "erdos_renyi" else "m",
                "param_value": float(param_label) if graph_type == "erdos_renyi" else int(param_label),
                "expression": model_summary["expression"],
                "r2": model_summary["r2"],
                "rmse": model_summary["rmse"],
                "aic": model_summary["aic"],
                "bic": model_summary["bic"],
                "results_dir": str(out_dir),
            }
        )

    summary_json = args.results_dir / "grid_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({"rows": summary_rows}, f, indent=2, default=str)

    summary_csv = args.results_dir / "grid_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_type",
                "param_name",
                "param_value",
                "expression",
                "r2",
                "rmse",
                "aic",
                "bic",
                "results_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    LOGGER.info("Saved grid summary JSON to %s", summary_json)
    LOGGER.info("Saved grid summary CSV to %s", summary_csv)


if __name__ == "__main__":
    main()

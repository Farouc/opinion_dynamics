"""Export a tabular two-camp tipping dataset for regression/symbolic discovery."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph_generation import (
    generate_barabasi_albert,
    generate_erdos_renyi,
    generate_fully_connected,
    generate_grid_lattice,
)
from src.symbolic_features_dataset import configuration_result_to_dataset_row, save_dataset_rows
from src.tipping_analysis import run_two_camp_configuration
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


def parse_str_list(raw: str) -> list[str]:
    vals = [str(x).strip() for x in str(raw).split(",") if str(x).strip()]
    if len(vals) == 0:
        raise ValueError("Expected at least one string value.")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--graph-types",
        type=str,
        default="fully_connected,erdos_renyi,barabasi_albert",
        help="Comma-separated graph families.",
    )
    parser.add_argument("--n-values", type=str, default="150,250")
    parser.add_argument("--L-values", type=str, default="20")
    parser.add_argument("--erdos-p-values", type=str, default="0.01,0.02")
    parser.add_argument("--barabasi-m-values", type=str, default="2,3")

    parser.add_argument("--n-pos-values", type=str, default="6,10,14")
    parser.add_argument("--n-neg-values", type=str, default="6,10,14,18")

    parser.add_argument(
        "--strategy-pos-list",
        type=str,
        default="random,highest_degree,highest_eigenvector,farthest_spread,wl_cover,hub_then_spread",
    )
    parser.add_argument(
        "--strategy-neg-list",
        type=str,
        default="random,highest_degree",
    )

    parser.add_argument("--T", type=int, default=15000)
    parser.add_argument("--burn-in", type=int, default=3000)
    parser.add_argument("--n-runs", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--wl-n-iter", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")

    return parser.parse_args()


def _build_graph_specs(args: argparse.Namespace) -> list[dict]:
    graph_types = parse_str_list(args.graph_types)
    n_values = parse_int_list(args.n_values)
    l_values = parse_int_list(args.L_values)
    erdos_p_values = parse_float_list(args.erdos_p_values)
    barabasi_m_values = parse_int_list(args.barabasi_m_values)

    specs: list[dict] = []

    for graph_type in graph_types:
        if graph_type == "fully_connected":
            for n in n_values:
                specs.append({"graph_type": graph_type, "n": int(n)})

        elif graph_type == "erdos_renyi":
            for n in n_values:
                for p in erdos_p_values:
                    specs.append({"graph_type": graph_type, "n": int(n), "p": float(p)})

        elif graph_type == "barabasi_albert":
            for n in n_values:
                for m in barabasi_m_values:
                    if int(m) >= int(n):
                        continue
                    specs.append({"graph_type": graph_type, "n": int(n), "m": int(m)})

        elif graph_type == "grid_lattice":
            for L in l_values:
                specs.append({"graph_type": graph_type, "L": int(L), "n": int(L * L)})

        else:
            raise ValueError(
                f"Unsupported graph type '{graph_type}'. "
                "Choose from fully_connected, erdos_renyi, barabasi_albert, grid_lattice."
            )

    return specs


def _make_graph(spec: dict, seed: int) -> tuple:
    graph_type = spec["graph_type"]
    if graph_type == "fully_connected":
        return generate_fully_connected(spec["n"]), {"n": int(spec["n"])}
    if graph_type == "erdos_renyi":
        return generate_erdos_renyi(spec["n"], spec["p"], seed=seed), {
            "n": int(spec["n"]),
            "p": float(spec["p"]),
        }
    if graph_type == "barabasi_albert":
        return generate_barabasi_albert(spec["n"], spec["m"], seed=seed), {
            "n": int(spec["n"]),
            "m": int(spec["m"]),
        }
    if graph_type == "grid_lattice":
        return generate_grid_lattice(spec["L"]), {"L": int(spec["L"]), "n": int(spec["L"] * spec["L"])}
    raise ValueError(f"Unknown graph type {graph_type}")


def main() -> None:
    args = parse_args()
    setup_logging()

    specs = _build_graph_specs(args)
    n_pos_values = parse_int_list(args.n_pos_values)
    n_neg_values = parse_int_list(args.n_neg_values)
    strategy_pos_list = parse_str_list(args.strategy_pos_list)
    strategy_neg_list = parse_str_list(args.strategy_neg_list)

    rng = np.random.default_rng(int(args.seed))

    rows: list[dict] = []

    total_jobs = 0
    for spec in specs:
        n_nodes = int(spec.get("n", spec.get("L", 0) ** 2))
        valid_pairs = [(a, b) for a in n_pos_values for b in n_neg_values if a + b <= n_nodes]
        total_jobs += len(valid_pairs) * len(strategy_pos_list) * len(strategy_neg_list)

    pbar = tqdm(total=total_jobs, desc="export symbolic dataset", disable=args.no_progress)

    for spec in specs:
        graph_seed = int(rng.integers(np.iinfo(np.int32).max))
        G, graph_params = _make_graph(spec, seed=graph_seed)
        graph_type = str(spec["graph_type"])

        n_nodes = G.number_of_nodes()
        valid_pairs = [(a, b) for a in n_pos_values for b in n_neg_values if a + b <= n_nodes]

        for n_pos, n_neg in valid_pairs:
            for strategy_pos in strategy_pos_list:
                for strategy_neg in strategy_neg_list:
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
                        threshold=float(args.threshold),
                        wl_n_iter=int(args.wl_n_iter),
                        show_progress=False,
                        store_run_records=False,
                    )

                    row = configuration_result_to_dataset_row(
                        res,
                        graph_type=graph_type,
                        graph_params={**graph_params, "graph_instance_seed": graph_seed},
                    )
                    rows.append(row)
                    pbar.update(1)

    pbar.close()

    raw_dir = ensure_dir(Path(args.results_dir) / "raw")
    json_path = raw_dir / "symbolic_dataset_two_camp.json"
    csv_path = raw_dir / "symbolic_dataset_two_camp.csv"
    save_dataset_rows(rows, json_path=json_path, csv_path=csv_path)

    meta = {
        "n_rows": len(rows),
        "graph_specs": specs,
        "n_pos_values": n_pos_values,
        "n_neg_values": n_neg_values,
        "strategy_pos_list": strategy_pos_list,
        "strategy_neg_list": strategy_neg_list,
        "T": int(args.T),
        "burn_in": int(args.burn_in),
        "n_runs": int(args.n_runs),
        "threshold": float(args.threshold),
        "wl_n_iter": int(args.wl_n_iter),
        "seed": int(args.seed),
    }

    meta_path = raw_dir / "symbolic_dataset_two_camp_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    LOGGER.info("Saved symbolic dataset JSON to %s", json_path)
    LOGGER.info("Saved symbolic dataset CSV to %s", csv_path)
    LOGGER.info("Saved symbolic dataset metadata to %s", meta_path)


if __name__ == "__main__":
    main()

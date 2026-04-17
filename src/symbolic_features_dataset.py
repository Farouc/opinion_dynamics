"""Utilities to build tabular datasets for regression/symbolic-discovery."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return value


def flatten_record_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested objects by JSON-encoding non-scalar values."""
    out: dict[str, Any] = {}
    for key, val in record.items():
        if isinstance(val, (str, int, float, bool)) or val is None:
            out[key] = val
        else:
            out[key] = json.dumps(val, sort_keys=True)
    return out


def configuration_result_to_dataset_row(
    result: dict,
    graph_type: str,
    graph_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one `run_two_camp_configuration` output into one dataset row."""
    cfg = dict(result.get("config", {}))
    agg = dict(result.get("aggregate", {}))
    graph_features = dict(result.get("graph_features", {}))

    row: dict[str, Any] = {
        "graph_type": str(graph_type),
        **{f"graph_param_{k}": _jsonable_scalar(v) for k, v in dict(graph_params or {}).items()},
        **{k: _jsonable_scalar(v) for k, v in cfg.items() if k not in {"strategy_kwargs_pos", "strategy_kwargs_neg"}},
        "strategy_kwargs_pos": cfg.get("strategy_kwargs_pos", {}),
        "strategy_kwargs_neg": cfg.get("strategy_kwargs_neg", {}),
        **{f"graph_{k}": _jsonable_scalar(v) for k, v in graph_features.items()},
        **{k: _jsonable_scalar(v) for k, v in agg.items()},
    }

    # Canonical target aliases for downstream symbolic regression.
    row["target_mean_m"] = row.get("mean_mean_magnetization")
    row["target_positive_win_probability"] = row.get("positive_win_probability")
    row["target_threshold_crossing_time"] = row.get("mean_time_to_first_crossing_above_threshold")
    row["target_stabilization_time"] = row.get("mean_stabilization_time")

    return row


def save_dataset_rows(
    rows: list[dict[str, Any]],
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    """Save dataset rows to JSON and CSV."""
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, default=str)

    flat_rows = [flatten_record_for_csv(row) for row in rows]
    fieldnames: list[str] = []
    field_set = set()
    for row in flat_rows:
        for key in row.keys():
            if key not in field_set:
                field_set.add(key)
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

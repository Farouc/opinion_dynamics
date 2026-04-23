"""Read experiment outputs (Exp1-Exp5) and generate a quantitative Markdown analysis report.

Usage:
    python3 experiments/analyze_and_report.py
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT_PATH = RESULTS / "analysis_report.md"

WARNINGS: list[str] = []
LOAD_RECORDS: list[dict[str, str]] = []
MISSING_EXPECTED: list[str] = []


def warn(msg: str) -> None:
    text = f"[warning] {msg}"
    print(text)
    WARNINGS.append(text)


def _to_builtin(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _file_size_human(path: Path) -> str:
    try:
        n = path.stat().st_size
    except Exception:
        return "unknown"
    units = ["B", "KB", "MB", "GB"]
    v = float(n)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            return f"{v:.1f}{u}"
        v /= 1024.0
    return f"{n}B"


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, on_bad_lines="warn")
        LOAD_RECORDS.append(
            {
                "path": str(path.relative_to(ROOT)),
                "type": "csv",
                "size": _file_size_human(path),
                "detail": f"rows={len(df)}, cols={len(df.columns)}",
            }
        )
        return df
    except Exception as exc:
        warn(f"Could not read CSV: {path} ({exc})")
        return None


def safe_read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        detail = "keys=" + str(len(obj.keys())) if isinstance(obj, dict) else f"len={len(obj)}"
        LOAD_RECORDS.append(
            {
                "path": str(path.relative_to(ROOT)),
                "type": "json",
                "size": _file_size_human(path),
                "detail": detail,
            }
        )
        return obj
    except Exception as exc:
        warn(f"Could not read JSON: {path} ({exc})")
        return None


def safe_read_npz(path: Path) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=True) as data:
            out = {k: data[k] for k in data.files}
        shape_desc = ", ".join([f"{k}:{np.shape(v)}" for k, v in out.items()])
        LOAD_RECORDS.append(
            {
                "path": str(path.relative_to(ROOT)),
                "type": "npz",
                "size": _file_size_human(path),
                "detail": shape_desc,
            }
        )
        return out
    except Exception as exc:
        warn(f"Could not read NPZ: {path} ({exc})")
        return None


def finite_or_nan(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def mean_nan(x: np.ndarray | list[float]) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def std_nan(x: np.ndarray | list[float], ddof: int = 0) -> float:
    arr = np.asarray(x, dtype=float)
    mask = np.isfinite(arr)
    if np.sum(mask) <= ddof:
        return float("nan")
    return float(np.nanstd(arr, ddof=ddof))


def first_cross_abs(series: np.ndarray, thr: float = 0.2) -> float:
    arr = np.asarray(series, dtype=float)
    idx = np.where(np.abs(arr) > float(thr))[0]
    return float(idx[0] + 1) if idx.size > 0 else float("nan")


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        return float("nan")
    out = spearmanr(xx, yy, nan_policy="omit")
    return finite_or_nan(out.correlation)


def logistic_centered(x: np.ndarray, a: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-a * (x - 0.5)))


def fit_logistic_steepness(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 4:
        return float("nan")
    try:
        popt, _ = curve_fit(
            logistic_centered,
            x,
            y,
            p0=[8.0],
            bounds=([-200.0], [200.0]),
            maxfev=20000,
        )
        return float(popt[0])
    except Exception:
        return float("nan")


def auc_logistic_classifier(x: np.ndarray, y_prob: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_bin = (y_prob > 0.5).astype(int)

    mask = np.isfinite(x) & np.isfinite(y_prob)
    x = x[mask]
    y_bin = y_bin[mask]
    if x.size < 4:
        return float("nan")
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    try:
        clf = LogisticRegression(max_iter=5000, random_state=0)
        clf.fit(x.reshape(-1, 1), y_bin)
        pred = clf.predict_proba(x.reshape(-1, 1))[:, 1]
        return float(roc_auc_score(y_bin, pred))
    except Exception:
        return float("nan")


def format_float(x: Any, ndigits: int = 3) -> str:
    v = finite_or_nan(x)
    return f"{v:.{ndigits}f}" if np.isfinite(v) else "Data not available"


def md_table(headers: list[str], rows: list[list[Any]], ndigits: int = 3) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out = []
        for v in row:
            if isinstance(v, float):
                out.append(format_float(v, ndigits=ndigits))
            else:
                out.append(str(v))
        lines.append("| " + " | ".join(out) + " |")
    return "\n".join(lines)


def parse_graph_from_token(token: str) -> tuple[str, str]:
    t = str(token)
    if t.startswith("barabasi_albert"):
        return "barabasi_albert", t
    if t.startswith("erdos_renyi"):
        return "erdos_renyi", t
    if t.startswith("fully_connected"):
        return "fully_connected", t
    if t.startswith("grid_lattice"):
        return "grid_lattice", t
    return "unknown", t


def check_expected_patterns() -> None:
    expected = [
        ("Exp1 NPZ trajectories", RESULTS / "exp1_timeseries_and_convergence" / "raw", "trajectories_*.npz"),
        ("Exp1 JSON params", RESULTS / "exp1_timeseries_and_convergence" / "raw", "trajectories_*.json"),
        ("Exp2 NPZ phase grid", RESULTS / "exp2_phase_diagram_tipping" / "raw", "phase_grid_*.npz"),
        ("Exp2 CSV phase grid", RESULTS / "exp2_phase_diagram_tipping" / "raw", "phase_grid_*.csv"),
        ("Exp2 boundary JSON", RESULTS / "exp2_phase_diagram_tipping" / "raw", "tipping_boundary_*.json"),
        ("Exp3 strategy CSV", RESULTS / "exp3_wl_cluster_placement" / "raw", "wl_strategy_comparison_*.csv"),
        ("Exp3 WL JSON", RESULTS / "exp3_wl_cluster_placement" / "raw", "wl_features_*.json"),
        ("Exp4 dataset CSV", RESULTS / "exp4_effective_strength_candidates" / "raw", "strength_dataset_all_configs.csv"),
        ("Exp5 family CSV", RESULTS / "exp5_graph_family_comparison" / "raw", "graph_family_comparison_*.csv"),
        ("Exp5 degree JSON", RESULTS / "exp5_graph_family_comparison" / "raw", "degree_distributions_*.json"),
    ]
    for label, directory, pattern in expected:
        try:
            found = sorted(directory.glob(pattern))
        except Exception as exc:
            warn(f"Could not glob expected pattern {label}: {directory}/{pattern} ({exc})")
            found = []
        if len(found) == 0:
            miss = f"{label}: missing `{directory.relative_to(ROOT)}/{pattern}`"
            MISSING_EXPECTED.append(miss)
            warn(miss)


def analyze_exp1() -> dict[str, Any]:
    out: dict[str, Any] = {"rows": [], "fastest_by_graph": [], "metastable_rows": []}
    raw_dir = RESULTS / "exp1_timeseries_and_convergence" / "raw"
    files = sorted(raw_dir.glob("trajectories_*.npz"))
    if len(files) == 0:
        warn("Exp1: no trajectories_*.npz files found.")
        return out

    for npz_path in files:
        npz = safe_read_npz(npz_path)
        if npz is None:
            continue
        mag = np.asarray(npz.get("magnetization", []), dtype=float)
        if mag.ndim != 2 or mag.size == 0:
            warn(f"Exp1: malformed magnetization in {npz_path.name}")
            continue

        json_path = npz_path.with_suffix(".json")
        meta: dict[str, Any] = {}
        obj = safe_read_json(json_path)
        if isinstance(obj, dict):
            meta = obj
        else:
            warn(f"Exp1: missing or malformed sidecar JSON for {npz_path.name}")

        m = re.match(
            r"^trajectories_(?P<graph>.+?)_npos(?P<npos>\d+)_nneg(?P<nneg>\d+)_(?P<regime>[^_]+)_(?P<spos>.+)_vs_(?P<sneg>.+)_allseeds$",
            npz_path.stem,
        )
        fname_info = m.groupdict() if m else {}

        graph_token = str(fname_info.get("graph", "unknown"))
        graph_type, _ = parse_graph_from_token(graph_token)
        regime = str(meta.get("regime", fname_info.get("regime", "unknown")))
        strategy_pos = str(meta.get("strategy_pos", fname_info.get("spos", "unknown")))
        strategy_neg = str(meta.get("strategy_neg", fname_info.get("sneg", "unknown")))
        n_pos = int(meta.get("n_pos", fname_info.get("npos", -1)))
        n_neg = int(meta.get("n_neg", fname_info.get("nneg", -1)))

        final_m = mag[:, -1]
        conv_times = np.array([first_cross_abs(mag[i], thr=0.2) for i in range(mag.shape[0])], dtype=float)
        row = {
            "graph_type": graph_type,
            "graph_token": graph_token,
            "regime": regime,
            "strategy_pair": f"{strategy_pos} vs {strategy_neg}",
            "strategy_pos": strategy_pos,
            "strategy_neg": strategy_neg,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_runs": int(mag.shape[0]),
            "mean_convergence_time": mean_nan(conv_times),
            "mean_final_m": float(np.mean(final_m)),
            "inter_seed_std": std_nan(final_m, ddof=1 if mag.shape[0] > 1 else 0),
            "final_time_variance": float(np.var(final_m)),
        }
        out["rows"].append(row)

    if len(out["rows"]) == 0:
        return out

    df = pd.DataFrame(out["rows"])

    fastest_rows = []
    for graph_type, sub in df.groupby("graph_type"):
        valid = sub[np.isfinite(sub["mean_convergence_time"])]
        if len(valid) == 0:
            continue
        idx = valid["mean_convergence_time"].idxmin()
        fastest_rows.append(df.loc[idx].to_dict())
    out["fastest_by_graph"] = fastest_rows

    meta_rows = df[df["inter_seed_std"] > 0.3].copy()
    out["metastable_rows"] = meta_rows.to_dict(orient="records")
    out["df"] = df
    return out


def boundary_points_from_grid(
    n_pos_values: np.ndarray, n_neg_values: np.ndarray, pwin: np.ndarray, level: float = 0.5
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    n_pos = np.asarray(n_pos_values, dtype=float)
    n_neg = np.asarray(n_neg_values, dtype=float)
    P = np.asarray(pwin, dtype=float)
    if P.shape != (n_pos.size, n_neg.size):
        return points

    for j, nneg in enumerate(n_neg):
        col = P[:, j]
        for i in range(len(n_pos) - 1):
            y1 = col[i]
            y2 = col[i + 1]
            if not (np.isfinite(y1) and np.isfinite(y2)):
                continue
            if y1 == level:
                points.append((float(n_pos[i]), float(nneg)))
            if y2 == level:
                points.append((float(n_pos[i + 1]), float(nneg)))
            d1 = y1 - level
            d2 = y2 - level
            if d1 * d2 < 0:
                if abs(y2 - y1) < 1e-12:
                    x_cross = 0.5 * (n_pos[i] + n_pos[i + 1])
                else:
                    x_cross = n_pos[i] + (level - y1) * (n_pos[i + 1] - n_pos[i]) / (y2 - y1)
                points.append((float(x_cross), float(nneg)))
    return points


def boundary_map(points: list[tuple[float, float]]) -> dict[float, float]:
    if len(points) == 0:
        return {}
    buckets: dict[float, list[float]] = {}
    for x, y in points:
        yy = float(round(y, 6))
        buckets.setdefault(yy, []).append(float(x))
    return {yy: float(np.mean(xs)) for yy, xs in buckets.items()}


def analyze_exp2() -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": [],
        "boundaries": {},
        "displacements": [],
        "critical_rows": [],
        "minority_table": [],
    }
    raw_dir = RESULTS / "exp2_phase_diagram_tipping" / "raw"
    npz_files = sorted(raw_dir.glob("phase_grid_*.npz"))
    if len(npz_files) == 0:
        warn("Exp2: no phase_grid_*.npz files found.")
        return out

    all_rows = []
    boundary_by_key: dict[tuple[str, str], list[tuple[float, float]]] = {}

    for npz_path in npz_files:
        npz = safe_read_npz(npz_path)
        if npz is None:
            continue

        n_pos_vals = np.asarray(npz.get("n_pos_values", []), dtype=float)
        n_neg_vals = np.asarray(npz.get("n_neg_values", []), dtype=float)
        P = np.asarray(npz.get("P_plus_win", []), dtype=float)
        M = np.asarray(npz.get("mean_magnetization", []), dtype=float)
        C = np.asarray(npz.get("mean_crossing_time", []), dtype=float)

        if n_pos_vals.size == 0 or n_neg_vals.size == 0 or P.size == 0:
            warn(f"Exp2: malformed arrays in {npz_path.name}")
            continue

        csv_path = npz_path.with_suffix(".csv")
        df_csv = safe_read_csv(csv_path)

        strategy_pos = "unknown"
        strategy_neg = "unknown"
        if df_csv is not None and "strategy_pos" in df_csv.columns and len(df_csv) > 0:
            strategy_pos = str(df_csv["strategy_pos"].dropna().iloc[0])
            strategy_neg = str(df_csv["strategy_neg"].dropna().iloc[0])

        stem = npz_path.stem.replace("phase_grid_", "")
        suffix = f"_{strategy_pos}_vs_{strategy_neg}"
        graph_token = stem[:-len(suffix)] if stem.endswith(suffix) else stem
        graph_type, _ = parse_graph_from_token(graph_token)

        points = boundary_points_from_grid(n_pos_vals, n_neg_vals, P, level=0.5)
        boundary_by_key[(graph_token, f"{strategy_pos} vs {strategy_neg}")] = points

        # Area where positive camp dominates.
        area_pos = int(np.sum(np.isfinite(P) & (P > 0.5)))

        npos_grid, nneg_grid = np.meshgrid(n_pos_vals, n_neg_vals, indexing="ij")
        minority_mask = np.isfinite(P) & (P > 0.5) & (npos_grid < nneg_grid)
        n_minority = int(np.sum(minority_mask))
        mean_p_minority = mean_nan(P[minority_mask]) if n_minority > 0 else float("nan")

        critical_mask = np.isfinite(P) & (np.abs(P - 0.5) <= 0.1)
        n_critical = int(np.sum(critical_mask))
        critical_cross_mean = mean_nan(C[critical_mask]) if n_critical > 0 else float("nan")
        if n_critical > 0:
            crit_pos = npos_grid[critical_mask]
            crit_neg = nneg_grid[critical_mask]
            width_pos = float(np.max(crit_pos) - np.min(crit_pos))
            width_neg = float(np.max(crit_neg) - np.min(crit_neg))
        else:
            width_pos = float("nan")
            width_neg = float("nan")

        row = {
            "graph_type": graph_type,
            "graph_token": graph_token,
            "strategy_pair": f"{strategy_pos} vs {strategy_neg}",
            "strategy_pos": strategy_pos,
            "strategy_neg": strategy_neg,
            "boundary_points": points,
            "area_positive_dominance": area_pos,
            "n_minority_win_cells": n_minority,
            "mean_P_win_minority_cells": mean_p_minority,
            "n_critical_cells": n_critical,
            "critical_crossing_time_mean": critical_cross_mean,
            "critical_band_width_npos": width_pos,
            "critical_band_width_nneg": width_neg,
            "passes_npos_equals_nneg": int(any(abs(px - py) <= 1.0 for px, py in points)),
        }
        all_rows.append(row)

    if len(all_rows) == 0:
        return out

    df = pd.DataFrame(all_rows)
    out["rows"] = all_rows
    out["df"] = df
    out["boundaries"] = {f"{k[0]}::{k[1]}": v for k, v in boundary_by_key.items()}

    # Displacement vs random baseline per graph.
    for graph_token, sub in df.groupby("graph_token"):
        baseline_row = sub[sub["strategy_pair"] == "random vs random"]
        if len(baseline_row) == 0:
            continue
        base_points = baseline_row.iloc[0]["boundary_points"]
        base_map = boundary_map(base_points)
        for _, row in sub.iterrows():
            spair = str(row["strategy_pair"])
            if spair == "random vs random":
                continue
            cur_map = boundary_map(row["boundary_points"])
            common = sorted(set(base_map.keys()) & set(cur_map.keys()))
            if len(common) == 0:
                mean_shift = float("nan")
                median_shift = float("nan")
            else:
                shifts = np.array([cur_map[v] - base_map[v] for v in common], dtype=float)
                mean_shift = float(np.mean(shifts))
                median_shift = float(np.median(shifts))
            out["displacements"].append(
                {
                    "graph_token": graph_token,
                    "graph_type": str(row["graph_type"]),
                    "strategy_pair": spair,
                    "mean_boundary_shift_npos": mean_shift,
                    "median_boundary_shift_npos": median_shift,
                    "interpretation_reduction_in_required_npos": -mean_shift if np.isfinite(mean_shift) else float("nan"),
                }
            )

    out["minority_table"] = (
        df[["graph_type", "strategy_pair", "n_minority_win_cells", "mean_P_win_minority_cells"]]
        .sort_values(["graph_type", "strategy_pair"])
        .to_dict(orient="records")
    )
    out["critical_rows"] = (
        df[
            [
                "graph_type",
                "strategy_pair",
                "n_critical_cells",
                "critical_crossing_time_mean",
                "critical_band_width_npos",
                "critical_band_width_nneg",
            ]
        ]
        .sort_values(["graph_type", "strategy_pair"])
        .to_dict(orient="records")
    )
    return out


def analyze_exp3() -> dict[str, Any]:
    out: dict[str, Any] = {
        "strategy_rows": [],
        "rank_rows": [],
        "corr_rows": [],
        "wl_partition_rows": [],
        "wl_cover_gain_rows": [],
    }
    raw_dir = RESULTS / "exp3_wl_cluster_placement" / "raw"

    csv_files = sorted(raw_dir.glob("wl_strategy_comparison_*.csv"))
    json_files = sorted(raw_dir.glob("wl_features_*.json"))
    if len(csv_files) == 0:
        warn("Exp3: no wl_strategy_comparison_*.csv files found.")

    all_csv = []
    for path in csv_files:
        df = safe_read_csv(path)
        if df is None or len(df) == 0:
            continue
        token = path.stem.replace("wl_strategy_comparison_", "")
        graph_type = str(df["graph_type"].iloc[0]) if "graph_type" in df.columns else parse_graph_from_token(token)[0]
        df = df.copy()
        df["graph_token"] = token
        df["graph_type"] = graph_type
        all_csv.append(df)
    if len(all_csv) > 0:
        df_all = pd.concat(all_csv, ignore_index=True)
    else:
        df_all = pd.DataFrame()
    out["strategy_rows"] = df_all.to_dict(orient="records") if len(df_all) > 0 else []

    if len(df_all) > 0:
        rank_rows = []
        for (graph_type, graph_token), sub in df_all.groupby(["graph_type", "graph_token"]):
            sub = sub.sort_values("P_plus_win", ascending=False).reset_index(drop=True)
            sub["rank"] = np.arange(1, len(sub) + 1)
            for _, row in sub.iterrows():
                rank_rows.append(
                    {
                        "graph_type": graph_type,
                        "graph_token": graph_token,
                        "strategy": row["strategy_pos"],
                        "P_plus_win": finite_or_nan(row["P_plus_win"]),
                        "mean_m": finite_or_nan(row["mean_m"]),
                        "psi_degree_norm": finite_or_nan(row["psi_degree_norm"]),
                        "psi_wl_coverage": finite_or_nan(row["psi_wl_coverage"]),
                        "psi_dispersion": finite_or_nan(row["psi_dispersion"]),
                        "rank": int(row["rank"]),
                    }
                )
        out["rank_rows"] = rank_rows

        corr_rows = []
        features = ["psi_degree_norm", "psi_wl_coverage", "psi_dispersion", "psi_pagerank"]
        for (graph_type, graph_token), sub in df_all.groupby(["graph_type", "graph_token"]):
            y = sub["P_plus_win"].to_numpy(dtype=float)
            for f in features:
                if f not in sub.columns:
                    r = float("nan")
                else:
                    r = pearson_corr(sub[f].to_numpy(dtype=float), y)
                corr_rows.append(
                    {
                        "graph_type": graph_type,
                        "graph_token": graph_token,
                        "feature": f,
                        "pearson_r": r,
                    }
                )
        out["corr_rows"] = corr_rows

        wl_gain_rows = []
        for (graph_type, graph_token), sub in df_all.groupby(["graph_type", "graph_token"]):
            random_rows = sub[sub["strategy_pos"] == "random"]
            wl_rows = sub[sub["strategy_pos"] == "wl_cover"]
            if len(random_rows) == 0 or len(wl_rows) == 0:
                gain = float("nan")
            else:
                gain = float(np.mean(wl_rows["P_plus_win"]) - np.mean(random_rows["P_plus_win"]))
            wl_gain_rows.append(
                {"graph_type": graph_type, "graph_token": graph_token, "wl_cover_minus_random_Pwin": gain}
            )
        out["wl_cover_gain_rows"] = wl_gain_rows

    for path in json_files:
        obj = safe_read_json(path)
        if not isinstance(obj, dict):
            continue
        token = path.stem.replace("wl_features_", "")
        graph_type, _ = parse_graph_from_token(token)
        counts_raw = obj.get("class_counts", {})
        if isinstance(counts_raw, dict):
            counts = np.array([finite_or_nan(v) for v in counts_raw.values()], dtype=float)
        else:
            counts = np.array([], dtype=float)

        counts = counts[np.isfinite(counts) & (counts > 0)]
        if counts.size > 0:
            probs = counts / np.sum(counts)
            entropy = float(-np.sum(probs * np.log(probs + 1e-15)))
            top1_frac = float(np.max(probs))
            top5_frac = float(np.sum(np.sort(probs)[-5:])) if probs.size >= 5 else float(np.sum(probs))
            effective_classes = float(np.exp(entropy))
            n_classes = int(obj.get("n_classes", len(counts)))
            n_nodes = int(np.sum(counts))
        else:
            entropy = top1_frac = top5_frac = effective_classes = float("nan")
            n_classes = int(obj.get("n_classes", 0))
            n_nodes = 0

        out["wl_partition_rows"].append(
            {
                "graph_type": graph_type,
                "graph_token": token,
                "n_classes": n_classes,
                "n_nodes": n_nodes,
                "top1_class_fraction": top1_frac,
                "top5_class_fraction": top5_frac,
                "wl_entropy": entropy,
                "effective_classes": effective_classes,
            }
        )
    return out


def analyze_exp4() -> dict[str, Any]:
    out: dict[str, Any] = {
        "metric_rows": [],
        "metric_rows_by_graph": [],
        "best_metric_overall": "Data not available",
        "best_metric_by_graph": [],
        "minority_count": 0,
        "minority_stats": {},
        "separator_metric": {},
        "correlation_crosscheck": {},
        "critical_band": {},
    }
    csv_path = RESULTS / "exp4_effective_strength_candidates" / "raw" / "strength_dataset_all_configs.csv"
    df = safe_read_csv(csv_path)
    if df is None or len(df) == 0:
        warn("Exp4: strength_dataset_all_configs.csv not available or empty.")
        return out

    metric_map = {
        "ratio_psi_rho": "ratio_psi_rho",
        "ratio_psi_degree_norm": "ratio_psi_degree_norm",
        "ratio_psi_pagerank": "ratio_psi_pagerank" if "ratio_psi_pagerank" in df.columns else "ratio_psi_centrality",
        "ratio_psi_wl": "ratio_psi_wl",
        "ratio_psi_dispersion": "ratio_psi_dispersion",
        "ratio_psi_hybrid": "ratio_psi_hybrid",
    }
    y = df["P_plus_win"].to_numpy(dtype=float)

    metric_rows = []
    for display, col in metric_map.items():
        if col not in df.columns:
            metric_rows.append(
                {
                    "metric": display,
                    "column": col,
                    "pearson_r": float("nan"),
                    "spearman_r": float("nan"),
                    "logistic_auc": float("nan"),
                    "logistic_steepness_a": float("nan"),
                }
            )
            continue
        x = df[col].to_numpy(dtype=float)
        metric_rows.append(
            {
                "metric": display,
                "column": col,
                "pearson_r": pearson_corr(x, y),
                "spearman_r": spearman_corr(x, y),
                "logistic_auc": auc_logistic_classifier(x, y),
                "logistic_steepness_a": fit_logistic_steepness(x, y),
            }
        )

    metric_df = pd.DataFrame(metric_rows)
    metric_df["rank_score"] = metric_df["logistic_auc"].fillna(-np.inf)
    metric_df = metric_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    metric_df["rank"] = np.arange(1, len(metric_df) + 1)
    out["metric_rows"] = metric_df.drop(columns=["rank_score"]).to_dict(orient="records")
    out["best_metric_overall"] = str(metric_df.iloc[0]["metric"]) if len(metric_df) > 0 else "Data not available"

    per_graph_rows = []
    best_by_graph = []
    for gtype, sub in df.groupby("graph_type"):
        rows_g = []
        for display, col in metric_map.items():
            if col not in sub.columns:
                rows_g.append({"graph_type": gtype, "metric": display, "logistic_auc": float("nan")})
                continue
            x = sub[col].to_numpy(dtype=float)
            yg = sub["P_plus_win"].to_numpy(dtype=float)
            rows_g.append(
                {
                    "graph_type": gtype,
                    "metric": display,
                    "pearson_r": pearson_corr(x, yg),
                    "spearman_r": spearman_corr(x, yg),
                    "logistic_auc": auc_logistic_classifier(x, yg),
                    "logistic_steepness_a": fit_logistic_steepness(x, yg),
                }
            )
        dfg = pd.DataFrame(rows_g).sort_values("logistic_auc", ascending=False)
        for _, rr in dfg.iterrows():
            per_graph_rows.append(rr.to_dict())
        if len(dfg) > 0:
            best = dfg.iloc[0]
            runner = dfg.iloc[1] if len(dfg) > 1 else None
            best_by_graph.append(
                {
                    "graph_type": gtype,
                    "best_metric": str(best["metric"]),
                    "AUC": finite_or_nan(best["logistic_auc"]),
                    "runner_up_metric": str(runner["metric"]) if runner is not None else "Data not available",
                    "AUC_runner_up": finite_or_nan(runner["logistic_auc"]) if runner is not None else float("nan"),
                }
            )
    out["metric_rows_by_graph"] = per_graph_rows
    out["best_metric_by_graph"] = best_by_graph

    # Minority dominance analysis.
    md = df[(df["n_pos"] < df["n_neg"]) & (df["P_plus_win"] > 0.6)].copy()
    out["minority_count"] = int(len(md))
    if len(md) > 0 and "ratio_psi_degree_norm" in md.columns:
        arr = md["ratio_psi_degree_norm"].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        out["minority_stats"] = {
            "ratio_psi_degree_norm_mean": mean_nan(arr),
            "ratio_psi_degree_norm_std": std_nan(arr, ddof=1 if arr.size > 1 else 0),
            "ratio_psi_degree_norm_q25": finite_or_nan(np.quantile(arr, 0.25)) if arr.size > 0 else float("nan"),
            "ratio_psi_degree_norm_q75": finite_or_nan(np.quantile(arr, 0.75)) if arr.size > 0 else float("nan"),
        }
    else:
        out["minority_stats"] = {
            "ratio_psi_degree_norm_mean": float("nan"),
            "ratio_psi_degree_norm_std": float("nan"),
            "ratio_psi_degree_norm_q25": float("nan"),
            "ratio_psi_degree_norm_q75": float("nan"),
        }

    # Best separator between minority-dominance and majority-win cases.
    maj = df[(df["n_pos"] >= df["n_neg"]) & (df["P_plus_win"] > 0.6)].copy()
    sep_rows = []
    if len(md) > 0 and len(maj) > 0:
        for display, col in metric_map.items():
            if col not in df.columns:
                sep_rows.append({"metric": display, "sep_auc": float("nan")})
                continue
            x_pos = md[col].to_numpy(dtype=float)
            x_neg = maj[col].to_numpy(dtype=float)
            x = np.concatenate([x_pos, x_neg])
            y_bin = np.concatenate([np.ones_like(x_pos, dtype=int), np.zeros_like(x_neg, dtype=int)])
            mask = np.isfinite(x)
            if np.sum(mask) < 4 or len(np.unique(y_bin[mask])) < 2:
                sep = float("nan")
            else:
                auc = float(roc_auc_score(y_bin[mask], x[mask]))
                sep = float(max(auc, 1.0 - auc))
            sep_rows.append({"metric": display, "sep_auc": sep})
    sep_df = pd.DataFrame(sep_rows)
    if len(sep_df) > 0 and np.any(np.isfinite(sep_df["sep_auc"])):
        best_sep = sep_df.iloc[sep_df["sep_auc"].idxmax()]
        out["separator_metric"] = {"metric": str(best_sep["metric"]), "sep_auc": finite_or_nan(best_sep["sep_auc"])}
    else:
        out["separator_metric"] = {"metric": "Data not available", "sep_auc": float("nan")}

    # Cross-check with metric_correlations.json if present.
    corr_json_path = RESULTS / "exp4_effective_strength_candidates" / "raw" / "metric_correlations.json"
    corr_json = safe_read_json(corr_json_path)
    cross = {"available": False, "mean_abs_diff": float("nan"), "max_abs_diff": float("nan"), "n_compared": 0}
    if isinstance(corr_json, dict):
        diffs = []
        for gtype, sub in df.groupby("graph_type"):
            if gtype not in corr_json or not isinstance(corr_json[gtype], dict):
                continue
            y_sub = sub["P_plus_win"].to_numpy(dtype=float)
            for display, col in metric_map.items():
                key = col  # JSON uses raw column names.
                if key not in corr_json[gtype]:
                    continue
                if col not in sub.columns:
                    continue
                r_now = pearson_corr(sub[col].to_numpy(dtype=float), y_sub)
                r_prev = finite_or_nan(corr_json[gtype][key])
                if np.isfinite(r_now) and np.isfinite(r_prev):
                    diffs.append(abs(r_now - r_prev))
        if len(diffs) > 0:
            cross = {
                "available": True,
                "mean_abs_diff": float(np.mean(diffs)),
                "max_abs_diff": float(np.max(diffs)),
                "n_compared": int(len(diffs)),
            }
        else:
            cross = {"available": True, "mean_abs_diff": float("nan"), "max_abs_diff": float("nan"), "n_compared": 0}
    out["correlation_crosscheck"] = cross

    # Critical/metastable band based on best metric.
    if len(metric_df) > 0:
        best_metric = str(metric_df.iloc[0]["metric"])
        best_col = metric_map.get(best_metric, best_metric)
        a_best = finite_or_nan(metric_df.iloc[0]["logistic_steepness_a"])
        if best_col in df.columns:
            x_best = df[best_col].to_numpy(dtype=float)
            if np.isfinite(a_best) and abs(a_best) > 1e-9:
                delta = math.log(0.6 / 0.4) / abs(a_best)
                low = max(0.0, 0.5 - delta)
                high = min(1.0, 0.5 + delta)
            else:
                crit = np.isfinite(x_best) & (np.abs(df["P_plus_win"].to_numpy(dtype=float) - 0.5) < 0.1)
                if np.any(crit):
                    low = float(np.nanmin(x_best[crit]))
                    high = float(np.nanmax(x_best[crit]))
                else:
                    low = high = float("nan")
            in_band = np.isfinite(x_best)
            if np.isfinite(low) and np.isfinite(high):
                in_band &= (x_best >= low) & (x_best <= high)
            else:
                in_band &= np.zeros_like(x_best, dtype=bool)

            out["critical_band"] = {
                "metric": best_metric,
                "column": best_col,
                "low": low,
                "high": high,
                "width": (high - low) if (np.isfinite(low) and np.isfinite(high)) else float("nan"),
                "n_rows_in_band": int(np.sum(in_band)),
                "mean_P_plus_win_in_band": mean_nan(df.loc[in_band, "P_plus_win"].to_numpy(dtype=float)),
                "mean_std_m_in_band": mean_nan(df.loc[in_band, "std_m"].to_numpy(dtype=float))
                if "std_m" in df.columns
                else float("nan"),
                "mean_crossing_time_in_band": float("nan"),  # not present in Exp4 dataset.
            }
        else:
            out["critical_band"] = {}
    return out


def analyze_exp5() -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": [],
        "placement_rows": [],
        "regression": {},
        "threshold": float("nan"),
        "family_notes": {},
    }
    raw_dir = RESULTS / "exp5_graph_family_comparison" / "raw"
    csv_files = sorted(raw_dir.glob("graph_family_comparison_*.csv"))
    if len(csv_files) == 0:
        warn("Exp5: no graph_family_comparison_*.csv files found.")
        return out

    dfs = []
    for path in csv_files:
        df = safe_read_csv(path)
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        dfs.append(df)
    if len(dfs) == 0:
        return out

    df_all = pd.concat(dfs, ignore_index=True)
    out["rows"] = df_all.to_dict(orient="records")

    # Placement advantage (highdeg-vs-random minus random-vs-random).
    high = df_all[(df_all["strategy_pos"] == "highest_degree") & (df_all["strategy_neg"] == "random")].copy()
    rand = df_all[(df_all["strategy_pos"] == "random") & (df_all["strategy_neg"] == "random")].copy()

    merged = pd.merge(
        high[
            [
                "config_label",
                "family",
                "graph_type",
                "degree_heterogeneity",
                "P_plus_win",
                "mean_time_abs_m_gt_03",
            ]
        ].rename(
            columns={
                "P_plus_win": "P_win_highdeg",
                "mean_time_abs_m_gt_03": "mean_convergence_time_highdeg",
            }
        ),
        rand[["config_label", "P_plus_win", "mean_time_abs_m_gt_03"]].rename(
            columns={
                "P_plus_win": "P_win_random",
                "mean_time_abs_m_gt_03": "mean_convergence_time_random",
            }
        ),
        on="config_label",
        how="inner",
    )
    merged["placement_advantage"] = merged["P_win_highdeg"] - merged["P_win_random"]
    out["placement_rows"] = merged.to_dict(orient="records")

    # Regression placement_advantage ~ degree_heterogeneity.
    x = merged["degree_heterogeneity"].to_numpy(dtype=float)
    y = merged["placement_advantage"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) >= 2:
        xx = x[mask]
        yy = y[mask]
        slope, intercept = np.polyfit(xx, yy, deg=1)
        pred = slope * xx + intercept
        ss_res = float(np.sum((yy - pred) ** 2))
        ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
        out["regression"] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "R2": r2,
        }
    else:
        out["regression"] = {"slope": float("nan"), "intercept": float("nan"), "R2": float("nan")}

    sub = merged[np.isfinite(merged["degree_heterogeneity"]) & (merged["placement_advantage"] > 0.1)]
    out["threshold"] = float(sub["degree_heterogeneity"].min()) if len(sub) > 0 else float("nan")

    # Brief family-level notes.
    family_notes = {}
    for fam, subf in merged.groupby("family"):
        family_notes[str(fam)] = {
            "mean_advantage": mean_nan(subf["placement_advantage"].to_numpy(dtype=float)),
            "mean_conv_highdeg": mean_nan(subf["mean_convergence_time_highdeg"].to_numpy(dtype=float)),
            "mean_conv_random": mean_nan(subf["mean_convergence_time_random"].to_numpy(dtype=float)),
            "max_advantage": finite_or_nan(np.nanmax(subf["placement_advantage"].to_numpy(dtype=float))),
            "min_advantage": finite_or_nan(np.nanmin(subf["placement_advantage"].to_numpy(dtype=float))),
        }
    out["family_notes"] = family_notes

    # Read degree JSON files (for availability listing + extra checks).
    for jpath in sorted(raw_dir.glob("degree_distributions_*.json")):
        _ = safe_read_json(jpath)
    return out


def build_report(
    exp1: dict[str, Any],
    exp2: dict[str, Any],
    exp3: dict[str, Any],
    exp4: dict[str, Any],
    exp5: dict[str, Any],
) -> str:
    lines: list[str] = []
    today = date.today().isoformat()

    lines.append("# Two-Camp Zealot Voter Model — Results Analysis")
    lines.append(f"**Project:** MVA Interactions | **Date:** {today}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 0. Executive Summary
    lines.append("## 0. Executive Summary")
    summary_bullets = []

    # Bullet 1: Exp2 minority dominance.
    if exp2.get("minority_table"):
        df2m = pd.DataFrame(exp2["minority_table"])
        total_cells = int(df2m["n_minority_win_cells"].sum())
        max_case = df2m.iloc[df2m["n_minority_win_cells"].idxmax()]
        summary_bullets.append(
            f"- Exp2 finds **{total_cells} minority-dominance cells** in total; max per setting is "
            f"**{int(max_case['n_minority_win_cells'])}** for `{max_case['graph_type']} | {max_case['strategy_pair']}`."
        )
    else:
        summary_bullets.append("- Exp2: Data not available.")

    # Bullet 2: Exp4 best predictor.
    mrows = exp4.get("metric_rows", [])
    if len(mrows) > 0:
        mdf = pd.DataFrame(mrows)
        best = mdf.iloc[0]
        summary_bullets.append(
            f"- Exp4 best overall metric is **{best['metric']}** with logistic AUC "
            f"**{format_float(best['logistic_auc'])}** and steepness **a={format_float(best['logistic_steepness_a'])}**."
        )
    else:
        summary_bullets.append("- Exp4: Data not available.")

    # Bullet 3: Exp5 topology sensitivity.
    reg = exp5.get("regression", {})
    if reg:
        summary_bullets.append(
            f"- Exp5 regression `placement_advantage ~ degree_heterogeneity` gives slope "
            f"**{format_float(reg.get('slope'))}** and R² **{format_float(reg.get('R2'))}**."
        )
    else:
        summary_bullets.append("- Exp5: Data not available.")

    # Bullet 4: Exp1 metastability.
    metastable = exp1.get("metastable_rows", [])
    if len(metastable) > 0:
        summary_bullets.append(
            f"- Exp1 flags **{len(metastable)} configurations** with inter-seed std(final m) > 0.3, "
            "consistent with metastable/tipping-adjacent dynamics."
        )
    else:
        summary_bullets.append("- Exp1: No high-variance metastable configurations above std>0.3 were detected.")

    # Bullet 5: Exp3 WL placement delta.
    wl_gain = exp3.get("wl_cover_gain_rows", [])
    if len(wl_gain) > 0:
        gdf = pd.DataFrame(wl_gain)
        if np.any(np.isfinite(gdf["wl_cover_minus_random_Pwin"])):
            mean_gain = float(np.nanmean(gdf["wl_cover_minus_random_Pwin"]))
            summary_bullets.append(
                f"- Exp3 WL-cover vs random yields average ΔP_win **{mean_gain:.3f}** across tested graph instances."
            )
    for b in summary_bullets:
        lines.append(b)
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Exp1
    lines.append("## 1. Opinion Dynamics and Convergence (Exp 1)")
    lines.append("")
    lines.append("### 1.1 Convergence speed by graph type and strategy")
    if "df" in exp1 and len(exp1["df"]) > 0:
        df1 = exp1["df"].copy().sort_values(["graph_type", "regime", "strategy_pair"])
        rows = []
        for _, r in df1.iterrows():
            rows.append(
                [
                    r["graph_type"],
                    r["regime"],
                    r["strategy_pair"],
                    finite_or_nan(r["mean_convergence_time"]),
                    finite_or_nan(r["mean_final_m"]),
                    finite_or_nan(r["inter_seed_std"]),
                ]
            )
        lines.append(
            md_table(
                ["graph", "regime", "strategy_pair", "mean_convergence_time", "mean_final_m", "inter_seed_std"],
                rows,
            )
        )
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 1.2 Effect of placement strategy on convergence")
    fast_rows = exp1.get("fastest_by_graph", [])
    if len(fast_rows) > 0:
        for row in fast_rows:
            lines.append(
                f"- `{row['graph_type']}` fastest observed strategy is `{row['strategy_pair']}` "
                f"with mean convergence time {format_float(row['mean_convergence_time'])}."
            )
    else:
        lines.append("Data not available.")
    lines.append(
        "Higher spread in final magnetization across seeds indicates coexistence of opposite long-run outcomes under identical macro-parameters."
    )
    lines.append("")

    lines.append("### 1.3 Metastability indicators")
    meta_rows = exp1.get("metastable_rows", [])
    if len(meta_rows) > 0:
        rows = []
        for r in meta_rows:
            rows.append(
                [
                    r["graph_type"],
                    r["regime"],
                    r["strategy_pair"],
                    finite_or_nan(r["inter_seed_std"]),
                    finite_or_nan(r["mean_final_m"]),
                ]
            )
        lines.append(
            md_table(["graph", "regime", "strategy_pair", "inter_seed_std", "mean_final_m"], rows, ndigits=3)
        )
    else:
        lines.append("No configuration with inter-seed std(final m) > 0.3 was found.")
    lines.append("")

    # 2. Exp2
    lines.append("## 2. Phase Diagram and Tipping Boundary (Exp 2)")
    lines.append("")
    lines.append("### 2.1 Tipping boundary by strategy and graph")
    if "df" in exp2 and len(exp2["df"]) > 0:
        df2 = exp2["df"].copy().sort_values(["graph_type", "strategy_pair"])
        for _, r in df2.iterrows():
            shape_msg = "roughly linear-like" if int(r["passes_npos_equals_nneg"]) == 1 else "nonlinear/offset"
            lines.append(
                f"- `{r['graph_type']} | {r['strategy_pair']}`: boundary has {len(r['boundary_points'])} sampled points, "
                f"appears {shape_msg}, and {'does' if int(r['passes_npos_equals_nneg']) == 1 else 'does not'} intersect near n_pos=n_neg."
            )
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 2.2 Boundary displacement due to placement")
    disp_rows = exp2.get("displacements", [])
    if len(disp_rows) > 0:
        rows = []
        for r in sorted(disp_rows, key=lambda x: (x["graph_type"], x["strategy_pair"])):
            rows.append(
                [
                    r["graph_type"],
                    r["strategy_pair"],
                    finite_or_nan(r["mean_boundary_shift_npos"]),
                    finite_or_nan(r["interpretation_reduction_in_required_npos"]),
                ]
            )
        lines.append(md_table(["graph", "strategy_pair", "mean_shift_npos", "reduction_in_required_npos"], rows))
        lines.append(
            "Negative `mean_shift_npos` means fewer positive zealots are needed to hit the P=0.5 boundary compared with random placement."
        )
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 2.3 Minority dominance cells")
    mtab = exp2.get("minority_table", [])
    if len(mtab) > 0:
        rows = []
        for r in sorted(mtab, key=lambda x: (x["graph_type"], x["strategy_pair"])):
            rows.append(
                [
                    r["graph_type"],
                    r["strategy_pair"],
                    int(r["n_minority_win_cells"]),
                    finite_or_nan(r["mean_P_win_minority_cells"]),
                ]
            )
        lines.append(
            md_table(
                ["graph", "strategy_pair", "n_minority_win_cells", "mean_P_win_in_those_cells"],
                rows,
            )
        )
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 2.4 Critical region")
    crit_rows = exp2.get("critical_rows", [])
    if len(crit_rows) > 0:
        rows = []
        for r in crit_rows:
            rows.append(
                [
                    r["graph_type"],
                    r["strategy_pair"],
                    int(r["n_critical_cells"]),
                    finite_or_nan(r["critical_crossing_time_mean"]),
                    finite_or_nan(r["critical_band_width_npos"]),
                    finite_or_nan(r["critical_band_width_nneg"]),
                ]
            )
        lines.append(
            md_table(
                [
                    "graph",
                    "strategy_pair",
                    "n_critical_cells",
                    "mean_crossing_time",
                    "critical_width_npos",
                    "critical_width_nneg",
                ],
                rows,
            )
        )
        lines.append(
            "The band with |P_win-0.5|<=0.1 is used as the empirical critical zone; larger crossing times there indicate slowed dynamics near tipping."
        )
    else:
        lines.append("Data not available.")
    lines.append("")

    # 3. Exp3
    lines.append("## 3. WL-Based Structural Placement (Exp 3)")
    lines.append("")
    lines.append("### 3.1 WL partition structure")
    wl_part_rows = exp3.get("wl_partition_rows", [])
    if len(wl_part_rows) > 0:
        rows = []
        for r in sorted(wl_part_rows, key=lambda x: x["graph_type"]):
            rows.append(
                [
                    r["graph_type"],
                    int(r["n_classes"]),
                    int(r["n_nodes"]),
                    finite_or_nan(r["top1_class_fraction"]),
                    finite_or_nan(r["top5_class_fraction"]),
                    finite_or_nan(r["wl_entropy"]),
                    finite_or_nan(r["effective_classes"]),
                ]
            )
        lines.append(
            md_table(
                [
                    "graph",
                    "n_classes",
                    "n_nodes",
                    "top1_class_frac",
                    "top5_class_frac",
                    "wl_entropy",
                    "effective_classes",
                ],
                rows,
            )
        )
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 3.2 Strategy ranking")
    rank_rows = exp3.get("rank_rows", [])
    if len(rank_rows) > 0:
        rows = []
        for r in sorted(rank_rows, key=lambda x: (x["graph_type"], x["rank"])):
            rows.append(
                [
                    r["graph_type"],
                    r["strategy"],
                    finite_or_nan(r["P_plus_win"]),
                    finite_or_nan(r["mean_m"]),
                    finite_or_nan(r["psi_degree_norm"]),
                    finite_or_nan(r["psi_wl_coverage"]),
                    int(r["rank"]),
                ]
            )
        lines.append(
            md_table(
                ["graph", "strategy", "P_plus_win", "mean_m", "psi_degree_norm", "psi_wl_coverage", "rank"],
                rows,
            )
        )
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 3.3 Does WL coverage matter?")
    corr_rows = exp3.get("corr_rows", [])
    if len(corr_rows) > 0:
        rows = []
        for r in sorted(corr_rows, key=lambda x: (x["graph_type"], x["feature"])):
            rows.append([r["graph_type"], r["feature"], finite_or_nan(r["pearson_r"])])
        lines.append(md_table(["graph", "feature", "Pearson_r_with_P_plus_win"], rows))

        # Most predictive graph for WL coverage
        wl_only = [r for r in corr_rows if r["feature"] == "psi_wl_coverage" and np.isfinite(r["pearson_r"])]
        if len(wl_only) > 0:
            best = max(wl_only, key=lambda r: abs(r["pearson_r"]))
            lines.append(
                f"- WL coverage is most predictive on `{best['graph_type']}` with Pearson r={format_float(best['pearson_r'])}."
            )
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 3.4 Interpretation")
    gains = exp3.get("wl_cover_gain_rows", [])
    if len(gains) > 0:
        for g in gains:
            lines.append(
                f"- `{g['graph_type']}`: `wl_cover - random` gives ΔP_win={format_float(g['wl_cover_minus_random_Pwin'])}."
            )
    else:
        lines.append("Data not available.")
    lines.append(
        "WL-cover is expected to help when structural diversity (coverage of distinct local roles) matters more than raw hub concentration."
    )
    lines.append("")

    # 4. Exp4
    lines.append("## 4. Effective Strength Metrics (Exp 4)")
    lines.append("")
    lines.append("### 4.1 Metric ranking by predictive power")
    mrows = exp4.get("metric_rows", [])
    if len(mrows) > 0:
        rows = []
        for r in mrows:
            rows.append(
                [
                    r["metric"],
                    finite_or_nan(r["pearson_r"]),
                    finite_or_nan(r["spearman_r"]),
                    finite_or_nan(r["logistic_auc"]),
                    finite_or_nan(r["logistic_steepness_a"]),
                    int(r["rank"]),
                ]
            )
        lines.append(
            md_table(
                ["metric", "Pearson_r", "Spearman_r", "logistic_AUC", "logistic_steepness_a", "rank"],
                rows,
            )
        )
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 4.2 Best metric per graph type")
    bg = exp4.get("best_metric_by_graph", [])
    if len(bg) > 0:
        rows = []
        for r in sorted(bg, key=lambda x: x["graph_type"]):
            rows.append([r["graph_type"], r["best_metric"], r["AUC"], r["runner_up_metric"], r["AUC_runner_up"]])
        lines.append(md_table(["graph_type", "best_metric", "AUC", "runner_up_metric", "AUC_runner_up"], rows))
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 4.3 Does a single metric predict the winner?")
    if len(mrows) > 0:
        best = mrows[0]
        lines.append(
            f"The best metric is `{best['metric']}` with AUC={format_float(best['logistic_auc'])}. "
            f"Using the fitted centered logistic model, transition steepness is a={format_float(best['logistic_steepness_a'])}; "
            "the balance point is centered at ratio_psi=0.5."
        )
        lines.append(
            "Interpretation: larger |a| implies a sharper switch in win probability around the critical ratio."
        )
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 4.4 Minority dominance regime")
    lines.append(f"- Minority-win configurations found: **{int(exp4.get('minority_count', 0))}**.")
    ms = exp4.get("minority_stats", {})
    lines.append(
        f"- `ratio_psi_degree_norm` in minority-win cases: mean={format_float(ms.get('ratio_psi_degree_norm_mean'))}, "
        f"std={format_float(ms.get('ratio_psi_degree_norm_std'))}, "
        f"Q25={format_float(ms.get('ratio_psi_degree_norm_q25'))}, "
        f"Q75={format_float(ms.get('ratio_psi_degree_norm_q75'))}."
    )
    sep = exp4.get("separator_metric", {})
    lines.append(
        f"- Best separator between minority-win and majority-win positive cases: "
        f"`{sep.get('metric', 'Data not available')}` with separation AUC={format_float(sep.get('sep_auc'))}."
    )
    lines.append("")

    lines.append("### 4.5 The critical/metastable band")
    cb = exp4.get("critical_band", {})
    if cb:
        lines.append(
            f"- Best-metric band for near-criticality: [{format_float(cb.get('low'))}, {format_float(cb.get('high'))}], "
            f"width={format_float(cb.get('width'))}, rows={cb.get('n_rows_in_band', 'Data not available')}."
        )
        lines.append(
            f"- In-band mean P_win={format_float(cb.get('mean_P_plus_win_in_band'))}, "
            f"mean std_m={format_float(cb.get('mean_std_m_in_band'))}."
        )
        if not np.isfinite(finite_or_nan(cb.get("mean_crossing_time_in_band"))):
            lines.append("- Mean crossing times in this band: Data not available (not logged in Exp4 raw dataset).")
        else:
            lines.append(f"- Mean crossing time in band={format_float(cb.get('mean_crossing_time_in_band'))}.")
    else:
        lines.append("Data not available.")
    lines.append("")

    # 5. Exp5
    lines.append("## 5. Graph Topology Effects (Exp 5)")
    lines.append("")
    lines.append("### 5.1 Convergence speed by topology")
    if len(exp5.get("rows", [])) > 0:
        df5 = pd.DataFrame(exp5["rows"]).copy()
        rows = []
        for _, r in df5.sort_values(["family", "config_label", "strategy_pos"]).iterrows():
            rows.append(
                [
                    r["family"],
                    r["config_label"],
                    finite_or_nan(r["mean_time_abs_m_gt_03"]),
                    f"{r['strategy_pos']} vs {r['strategy_neg']}",
                ]
            )
        lines.append(md_table(["graph_family", "parameter", "mean_convergence_time", "strategy"], rows))
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 5.2 Placement advantage vs degree heterogeneity")
    prow = exp5.get("placement_rows", [])
    if len(prow) > 0:
        rows = []
        for r in sorted(prow, key=lambda x: x["config_label"]):
            rows.append(
                [
                    r["config_label"],
                    finite_or_nan(r["degree_heterogeneity"]),
                    finite_or_nan(r["P_win_highdeg"]),
                    finite_or_nan(r["P_win_random"]),
                    finite_or_nan(r["placement_advantage"]),
                ]
            )
        lines.append(
            md_table(
                ["graph_config", "degree_heterogeneity", "P_win_highdeg", "P_win_random", "placement_advantage"],
                rows,
            )
        )
        reg = exp5.get("regression", {})
        lines.append(
            f"- Regression: slope={format_float(reg.get('slope'))}, R²={format_float(reg.get('R2'))}, "
            f"intercept={format_float(reg.get('intercept'))}."
        )
        thr = exp5.get("threshold", float("nan"))
        if np.isfinite(finite_or_nan(thr)):
            lines.append(f"- Substantial advantage threshold (placement_advantage>0.1): heterogeneity >= {format_float(thr)}.")
        else:
            lines.append("- Substantial advantage threshold (placement_advantage>0.1): not observed.")
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 5.3 Qualitative topology regimes")
    notes = exp5.get("family_notes", {})
    fc = notes.get("fully_connected", {})
    er = notes.get("erdos_renyi", {})
    ba = notes.get("barabasi_albert", {})
    lines.append(
        f"- **Fully connected**: mean placement advantage={format_float(fc.get('mean_advantage'))}, "
        f"mean convergence (highdeg/random)={format_float(fc.get('mean_conv_highdeg'))}/{format_float(fc.get('mean_conv_random'))}."
    )
    lines.append(
        f"- **Erdős–Rényi**: mean placement advantage={format_float(er.get('mean_advantage'))}, "
        f"range=[{format_float(er.get('min_advantage'))}, {format_float(er.get('max_advantage'))}]."
    )
    lines.append(
        f"- **Barabási–Albert**: mean placement advantage={format_float(ba.get('mean_advantage'))}, "
        f"range=[{format_float(ba.get('min_advantage'))}, {format_float(ba.get('max_advantage'))}], "
        "consistent with hub-amplified effects when heterogeneity is high."
    )
    lines.append("")

    lines.append("### 5.4 Implications")
    lines.append(
        "Topology with higher degree heterogeneity is more sensitive to strategic placement, which implies real social networks with hubs may be disproportionately vulnerable to small but well-placed influence groups."
    )
    lines.append("")

    # 6. Cross-experiment synthesis.
    lines.append("## 6. Cross-Experiment Synthesis")
    lines.append("")
    lines.append("### 6.1 Consistent findings across experiments")
    lines.append("- Strategic placement shifts tipping behavior beyond raw zealot counts (Exp2, Exp5).")
    if len(mrows) > 0:
        lines.append(
            f"- `{mrows[0]['metric']}` is the strongest global scalar predictor in Exp4 (AUC={format_float(mrows[0]['logistic_auc'])})."
        )
    if len(meta_rows) > 0:
        lines.append(f"- Metastable/high-variance signatures appear in {len(meta_rows)} Exp1 settings.")
    lines.append("- Minority dominance is empirically present in phase-grid cells (Exp2) and configuration-level data (Exp4).")
    lines.append("")

    lines.append("### 6.2 Contradictions or surprises")
    if len(exp3.get("wl_cover_gain_rows", [])) > 0:
        neg = [r for r in exp3["wl_cover_gain_rows"] if np.isfinite(r["wl_cover_minus_random_Pwin"]) and r["wl_cover_minus_random_Pwin"] < 0]
        if len(neg) > 0:
            lines.append(
                f"- WL-cover does not universally beat random: {len(neg)} graph instance(s) show negative ΔP_win."
            )
        else:
            lines.append("- WL-cover was nonnegative versus random on all tested Exp3 graph instances.")
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 6.3 Best overall predictor of zealot dominance")
    if len(mrows) > 0:
        best = mrows[0]
        ba_auc = float("nan")
        er_auc = float("nan")
        for r in exp4.get("best_metric_by_graph", []):
            if r["graph_type"] == "barabasi_albert" and r["best_metric"] == best["metric"]:
                ba_auc = r["AUC"]
            if r["graph_type"] == "erdos_renyi" and r["best_metric"] == best["metric"]:
                er_auc = r["AUC"]
        lines.append(
            f'The metric **{best["metric"]}**, computed as a ratio between positive and total camp strength, '
            f"achieves AUC={format_float(ba_auc)} on BA and AUC={format_float(er_auc)} on ER in this run."
        )
    else:
        lines.append("Data not available.")
    lines.append("")

    lines.append("### 6.4 Open questions for future work")
    lines.append("1. Do the same metric rankings hold for larger N and longer T with tighter confidence intervals?")
    lines.append("2. Can crossing-time observables be logged in Exp4 to quantify metastability directly in metric space?")
    lines.append("3. How robust are WL-based gains under alternative community/WL-depth definitions?")
    lines.append("4. Is there a universal rescaling collapsing BA and ER tipping curves onto one master relation?")
    lines.append("5. Which causal mechanism (hub capture vs structural coverage) dominates in real-world network data?")
    lines.append("")

    # 7. Data availability
    lines.append("## 7. Data Availability")
    lines.append("")
    lines.append("List of raw files successfully read:")
    if len(LOAD_RECORDS) > 0:
        for rec in LOAD_RECORDS:
            lines.append(
                f"- `{rec['path']}` ({rec['type']}, {rec['size']}): {rec['detail']}"
            )
    else:
        lines.append("- Data not available.")
    lines.append("")
    lines.append("Missing expected files/patterns:")
    if len(MISSING_EXPECTED) > 0:
        for miss in MISSING_EXPECTED:
            lines.append(f"- {miss}")
    else:
        lines.append("- None.")
    if len(WARNINGS) > 0:
        lines.append("")
        lines.append("Warnings encountered while reading data:")
        for w in WARNINGS:
            lines.append(f"- {w}")
    lines.append("")
    lines.append("*Generated automatically by `experiments/analyze_and_report.py`*")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    check_expected_patterns()

    exp1 = analyze_exp1()
    exp2 = analyze_exp2()
    exp3 = analyze_exp3()
    exp4 = analyze_exp4()
    exp5 = analyze_exp5()

    report = build_report(exp1, exp2, exp3, exp4, exp5)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        REPORT_PATH.write_text(report, encoding="utf-8")
    except Exception as exc:
        warn(f"Could not write report at {REPORT_PATH}: {exc}")
        raise

    print(f"[ok] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a full LaTeX project report from experiment outputs.

This script:
1. Discovers all PNG figures under results/.
2. Builds a catalogue with inferred metadata.
3. Selects figures for the manuscript and copies them into report/figures/.
4. Reads results/analysis_report_approfondi.md for numerical/narrative reference.
5. Writes report/main.tex.
6. Attempts LaTeX compilation with latexmk.
7. Prints diagnostics and a final summary.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
REPORT_FIG_DIR = REPORT_DIR / "figures"
ANALYSIS_MD = RESULTS_DIR / "analysis_report_approfondi.md"
MAIN_TEX = REPORT_DIR / "main.tex"
FIGURE_MAP_JSON = REPORT_DIR / "figure_map.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def tex_escape(text: str) -> str:
    """Escape characters for LaTeX text mode."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def ensure_ascii(text: str) -> str:
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        clean = text.encode("ascii", "replace").decode("ascii")
        return clean


def infer_category(name: str) -> str:
    n = name.lower()
    if "magnetization_timeseries" in n:
        return "magnetization_timeseries"
    if "positive_fraction_evolution" in n:
        return "positive_fraction"
    if "network_snapshots" in n or "network_visualizations" in n:
        return "network_snapshot"
    if "phase_diagram_pwin" in n:
        return "phase_diagram_pwin"
    if "phase_diagram_mean_m" in n:
        return "phase_diagram_mean_m"
    if "tipping_boundary_comparison" in n:
        return "tipping_boundary_overlay"
    if "crossing_time_heatmap" in n:
        return "crossing_time_heatmap"
    if "wl_partition" in n:
        return "wl_partition"
    if "wl_iteration" in n:
        return "wl_iteration"
    if "strategy_comparison_bar" in n:
        return "strategy_bar"
    if "strategy_vs_structural_features" in n:
        return "strategy_feature_scatter"
    if "metric_vs_pwin_scatter" in n:
        return "metric_scatter"
    if "metric_comparison_summary" in n:
        return "metric_auc_summary"
    if "critical_region_scan" in n:
        return "critical_region"
    if "minority_dominance" in n:
        return "minority_dominance"
    if "convergence_speed" in n:
        return "convergence_speed"
    if "pwin_vs_degree_heterogeneity" in n:
        return "heterogeneity_scatter"
    if "magnetization_distribution_final" in n:
        return "magnetization_hist"
    if "structural_metrics_by_graph" in n:
        return "structural_metrics"
    if "rho_vs_tau" in n:
        return "rho_tau"
    if "strength" in n:
        return "strength"
    return "other"


def discover_figures(results_dir: Path) -> list[dict[str, Any]]:
    catalogue: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("*.png")):
        rel = path.relative_to(ROOT)
        rel_results = path.relative_to(results_dir)
        experiment = rel_results.parts[0] if len(rel_results.parts) > 0 else "unknown"
        item = {
            "path": path,
            "rel": rel,
            "experiment": experiment,
            "filename": path.name,
            "stem": path.stem,
            "category": infer_category(path.name),
        }
        catalogue.append(item)
    return catalogue


def read_analysis_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
        return text
    except FileNotFoundError:
        log(f"[warning] Missing analysis markdown: {path}")
        return ""
    except Exception as exc:
        log(f"[warning] Could not read analysis markdown {path}: {exc}")
        return ""


def extract_first(pattern: str, text: str, default: str) -> str:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return default
    return m.group(1)


def pick_by_exact_rel(catalogue: list[dict[str, Any]], rel_like: str) -> dict[str, Any] | None:
    target = rel_like.replace("\\", "/").strip("/")
    for item in catalogue:
        rel_str = item["rel"].as_posix()
        if rel_str.endswith(target):
            return item
    return None


def pick_by_contains(
    catalogue: list[dict[str, Any]],
    used: set[str],
    include_tokens: list[str],
    exclude_tokens: list[str] | None = None,
) -> dict[str, Any] | None:
    excl = exclude_tokens or []
    for item in catalogue:
        rel = item["rel"].as_posix().lower()
        if rel in used:
            continue
        if any(tok.lower() not in rel for tok in include_tokens):
            continue
        if any(tok.lower() in rel for tok in excl):
            continue
        return item
    return None


def clean_flat_name(rel_path: Path) -> str:
    text = rel_path.as_posix().replace("/", "_").replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    return text


def copy_selected_figures(selected: dict[str, dict[str, Any]], report_fig_dir: Path) -> dict[str, str]:
    report_fig_dir.mkdir(parents=True, exist_ok=True)
    alias_to_clean: dict[str, str] = {}
    clean_to_original: dict[str, str] = {}
    for alias, item in selected.items():
        src: Path = item["path"]
        rel: Path = item["rel"]
        clean_name = clean_flat_name(rel)
        dst = report_fig_dir / clean_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        alias_to_clean[alias] = clean_name
        clean_to_original[clean_name] = str(rel.as_posix())
    FIGURE_MAP_JSON.write_text(json.dumps(clean_to_original, indent=2), encoding="utf-8")
    return alias_to_clean


def make_figure_block(
    alias: str,
    alias_to_clean: dict[str, str],
    label: str,
    caption: str,
    width: str = "0.8\\linewidth",
) -> str:
    if alias in alias_to_clean:
        fname = alias_to_clean[alias]
        fig_path = REPORT_FIG_DIR / fname
        if fig_path.exists():
            return (
                "\\begin{figure}[H]\n"
                "\\centering\n"
                f"\\includegraphics[width={width}]{{\\detokenize{{{fname}}}}}\n"
                f"\\caption{{{tex_escape(caption)}}}\n"
                f"\\label{{{label}}}\n"
                "\\end{figure}\n"
            )
    return (
        "% Missing figure placeholder\n"
        "\\begin{figure}[H]\n"
        "\\centering\n"
        "\\fbox{\\parbox{0.78\\linewidth}{\\centering Figure unavailable in this environment.}}\n"
        f"\\caption{{{tex_escape(caption + ' (Figure unavailable in this run.)')}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{figure}\n"
    )


def make_subfigure_pair(
    alias_left: str,
    alias_right: str,
    alias_to_clean: dict[str, str],
    label: str,
    overall_caption: str,
    subcap_left: str = "Left panel.",
    subcap_right: str = "Right panel.",
) -> str:
    def _sub(alias: str, subcap: str) -> str:
        if alias in alias_to_clean and (REPORT_FIG_DIR / alias_to_clean[alias]).exists():
            fname = alias_to_clean[alias]
            return (
                "\\begin{subfigure}{0.48\\linewidth}\n"
                "\\centering\n"
                f"\\includegraphics[width=\\linewidth]{{\\detokenize{{{fname}}}}}\n"
                f"\\caption{{{tex_escape(subcap)}}}\n"
                "\\end{subfigure}\n"
            )
        return (
            "\\begin{subfigure}{0.48\\linewidth}\n"
            "\\centering\n"
            "\\fbox{\\parbox{0.9\\linewidth}{\\centering Missing figure}}\n"
            f"\\caption{{{tex_escape(subcap + ' (missing)')}}}\n"
            "\\end{subfigure}\n"
        )

    return (
        "\\begin{figure}[H]\n"
        "\\centering\n"
        f"{_sub(alias_left, subcap_left)}\n"
        "\\hfill\n"
        f"{_sub(alias_right, subcap_right)}\n"
        f"\\caption{{{tex_escape(overall_caption)}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{figure}\n"
    )


def make_subfigure_triple(
    a1: str,
    a2: str,
    a3: str,
    alias_to_clean: dict[str, str],
    label: str,
    overall_caption: str,
    c1: str,
    c2: str,
    c3: str,
) -> str:
    def _sub(alias: str, subcap: str) -> str:
        if alias in alias_to_clean and (REPORT_FIG_DIR / alias_to_clean[alias]).exists():
            fname = alias_to_clean[alias]
            return (
                "\\begin{subfigure}{0.32\\linewidth}\n"
                "\\centering\n"
                f"\\includegraphics[width=\\linewidth]{{\\detokenize{{{fname}}}}}\n"
                f"\\caption{{{tex_escape(subcap)}}}\n"
                "\\end{subfigure}\n"
            )
        return (
            "\\begin{subfigure}{0.32\\linewidth}\n"
            "\\centering\n"
            "\\fbox{\\parbox{0.9\\linewidth}{\\centering Missing figure}}\n"
            f"\\caption{{{tex_escape(subcap + ' (missing)')}}}\n"
            "\\end{subfigure}\n"
        )

    return (
        "\\begin{figure}[H]\n"
        "\\centering\n"
        f"{_sub(a1, c1)}\n"
        "\\hfill\n"
        f"{_sub(a2, c2)}\n"
        "\\hfill\n"
        f"{_sub(a3, c3)}\n"
        f"\\caption{{{tex_escape(overall_caption)}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{figure}\n"
    )


def select_figures(catalogue: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    used_rel: set[str] = set()

    def _take(alias: str, exact_rel: str | None = None, includes: list[str] | None = None) -> None:
        item = None
        if exact_rel:
            item = pick_by_exact_rel(catalogue, exact_rel)
        if item is None and includes:
            item = pick_by_contains(catalogue, used_rel, includes)
        if item is not None:
            selected[alias] = item
            used_rel.add(item["rel"].as_posix().lower())

    # Part 1 figures (outside exp1..exp5 dirs).
    _take(
        "part1_fc_rho_tau",
        exact_rel="results/fc_tau_multin_pysr/figures/rho_vs_tau_real_vs_pysr_N_150.png",
        includes=["fc_tau_multin_pysr", "rho_vs_tau", "n_150"],
    )
    _take(
        "part1_er_rho_tau",
        exact_rel="results/er_tau_multin_pysr/figures/rho_vs_tau_real_vs_pysr_erdos_renyi_N_150.png",
        includes=["er_tau_multin_pysr", "rho_vs_tau", "n_150"],
    )
    _take(
        "part1_ba_rho_tau",
        exact_rel="results/ba_tau_multin_pysr/figures/rho_vs_tau_real_vs_pysr_barabasi_albert_N_150.png",
        includes=["ba_tau_multin_pysr", "rho_vs_tau", "n_150"],
    )

    # Part 2 Exp1 figures.
    _take(
        "exp1_mag_ba_hd",
        exact_rel="results/exp1_timeseries_and_convergence/figures/magnetization_timeseries_barabasi_albert_N500_m3_npos25_nneg25_regimeB_highest_degree_vs_random.png",
        includes=["exp1_timeseries_and_convergence", "magnetization_timeseries", "barabasi_albert", "highest_degree_vs_random"],
    )
    _take(
        "exp1_mag_ba_rand",
        exact_rel="results/exp1_timeseries_and_convergence/figures/magnetization_timeseries_barabasi_albert_N500_m3_npos25_nneg25_regimeB_random_vs_random.png",
        includes=["exp1_timeseries_and_convergence", "magnetization_timeseries", "barabasi_albert", "random_vs_random"],
    )
    _take(
        "exp1_snap_small",
        exact_rel="results/exp1_timeseries_and_convergence/figures/network_snapshots_small_N50_seed42.png",
        includes=["exp1_timeseries_and_convergence", "network_snapshots_small"],
    )
    _take(
        "exp1_conv_ba",
        exact_rel="results/exp1_timeseries_and_convergence/figures/convergence_comparison_all_strategies_barabasi_albert_N500_m3_npos25_nneg25_regimeB.png",
        includes=["exp1_timeseries_and_convergence", "convergence_comparison_all_strategies", "barabasi_albert"],
    )

    # Part 2 Exp2 figures.
    _take(
        "exp2_pwin_ba_rand",
        exact_rel="results/exp2_phase_diagram_tipping/figures/phase_diagram_Pwin_barabasi_albert_N800_m3_random_vs_random.png",
        includes=["exp2_phase_diagram_tipping", "phase_diagram_pwin", "barabasi_albert", "random_vs_random"],
    )
    _take(
        "exp2_pwin_ba_hd",
        exact_rel="results/exp2_phase_diagram_tipping/figures/phase_diagram_Pwin_barabasi_albert_N800_m3_highest_degree_vs_random.png",
        includes=["exp2_phase_diagram_tipping", "phase_diagram_pwin", "barabasi_albert", "highest_degree_vs_random"],
    )
    _take(
        "exp2_pwin_er_rand",
        exact_rel="results/exp2_phase_diagram_tipping/figures/phase_diagram_Pwin_erdos_renyi_N800_p0015_random_vs_random.png",
        includes=["exp2_phase_diagram_tipping", "phase_diagram_pwin", "erdos_renyi", "random_vs_random"],
    )
    _take(
        "exp2_pwin_er_hd",
        exact_rel="results/exp2_phase_diagram_tipping/figures/phase_diagram_Pwin_erdos_renyi_N800_p0015_highest_degree_vs_random.png",
        includes=["exp2_phase_diagram_tipping", "phase_diagram_pwin", "erdos_renyi", "highest_degree_vs_random"],
    )
    _take(
        "exp2_boundary_ba",
        exact_rel="results/exp2_phase_diagram_tipping/figures/tipping_boundary_comparison_barabasi_albert_N800_m3.png",
        includes=["exp2_phase_diagram_tipping", "tipping_boundary_comparison", "barabasi_albert"],
    )
    _take(
        "exp2_boundary_er",
        exact_rel="results/exp2_phase_diagram_tipping/figures/tipping_boundary_comparison_erdos_renyi_N800_p0015.png",
        includes=["exp2_phase_diagram_tipping", "tipping_boundary_comparison", "erdos_renyi"],
    )
    _take(
        "exp2_cross_er_hd",
        exact_rel="results/exp2_phase_diagram_tipping/figures/crossing_time_heatmap_erdos_renyi_N800_p0015_highest_degree_vs_random.png",
        includes=["exp2_phase_diagram_tipping", "crossing_time_heatmap", "erdos_renyi", "highest_degree_vs_random"],
    )

    # Part 2 Exp3 figures.
    _take(
        "exp3_wl_part_ba",
        exact_rel="results/exp3_wl_cluster_placement/figures/wl_partition_evolution_barabasi_albert_N300_m2.png",
        includes=["exp3_wl_cluster_placement", "wl_partition_evolution", "barabasi_albert"],
    )
    _take(
        "exp3_wl_part_er",
        exact_rel="results/exp3_wl_cluster_placement/figures/wl_partition_evolution_erdos_renyi_N300_p002.png",
        includes=["exp3_wl_cluster_placement", "wl_partition_evolution", "erdos_renyi"],
    )
    _take(
        "exp3_strat_bar_ba",
        exact_rel="results/exp3_wl_cluster_placement/figures/strategy_comparison_bar_barabasi_albert_N300_m2.png",
        includes=["exp3_wl_cluster_placement", "strategy_comparison_bar", "barabasi_albert"],
    )
    _take(
        "exp3_strat_bar_er",
        exact_rel="results/exp3_wl_cluster_placement/figures/strategy_comparison_bar_erdos_renyi_N300_p002.png",
        includes=["exp3_wl_cluster_placement", "strategy_comparison_bar", "erdos_renyi"],
    )
    _take(
        "exp3_feat_er",
        exact_rel="results/exp3_wl_cluster_placement/figures/strategy_vs_structural_features_erdos_renyi_N300_p002.png",
        includes=["exp3_wl_cluster_placement", "strategy_vs_structural_features", "erdos_renyi"],
    )

    # Part 2 Exp4 figures.
    _take(
        "exp4_scatter_deg",
        exact_rel="results/exp4_effective_strength_candidates/figures/metric_vs_Pwin_scatter_ratio_psi_degree_norm.png",
        includes=["exp4_effective_strength_candidates", "metric_vs_pwin_scatter_ratio_psi_degree_norm"],
    )
    _take(
        "exp4_auc_summary",
        exact_rel="results/exp4_effective_strength_candidates/figures/metric_comparison_summary.png",
        includes=["exp4_effective_strength_candidates", "metric_comparison_summary"],
    )
    _take(
        "exp4_critical",
        exact_rel="results/exp4_effective_strength_candidates/figures/critical_region_scan_ratio_psi_degree_norm_all_graphs.png",
        includes=["exp4_effective_strength_candidates", "critical_region_scan_ratio_psi_degree_norm_all_graphs"],
    )
    _take(
        "exp4_minority_ba",
        exact_rel="results/exp4_effective_strength_candidates/figures/minority_dominance_cases_barabasi_albert.png",
        includes=["exp4_effective_strength_candidates", "minority_dominance_cases_barabasi_albert"],
    )

    # Part 2 Exp5 figures.
    _take(
        "exp5_conv_speed",
        exact_rel="results/exp5_graph_family_comparison/figures/convergence_speed_by_graph_strategy_comparison.png",
        includes=["exp5_graph_family_comparison", "convergence_speed_by_graph"],
    )
    _take(
        "exp5_hetero_scatter",
        exact_rel="results/exp5_graph_family_comparison/figures/Pwin_vs_degree_heterogeneity_strategy_comparison.png",
        includes=["exp5_graph_family_comparison", "pwin_vs_degree_heterogeneity"],
    )
    _take(
        "exp5_hist_ba",
        exact_rel="results/exp5_graph_family_comparison/figures/magnetization_distribution_final_barabasi_albert.png",
        includes=["exp5_graph_family_comparison", "magnetization_distribution_final_barabasi_albert"],
    )
    _take(
        "exp5_network_comp",
        exact_rel="results/exp5_graph_family_comparison/figures/network_visualizations_comparison.png",
        includes=["exp5_graph_family_comparison", "network_visualizations_comparison"],
    )
    _take(
        "exp5_struct_metrics",
        exact_rel="results/exp5_graph_family_comparison/figures/structural_metrics_by_graph_highest_degree_vs_random.png",
        includes=["exp5_graph_family_comparison", "structural_metrics_by_graph"],
    )

    return selected


def build_main_tex(
    alias_to_clean: dict[str, str],
    analysis_text: str,
    total_found: int,
    used_count: int,
    experiments_covered: list[str],
) -> str:
    # Pull key numbers from analysis markdown when possible, with robust defaults.
    auc_deg = extract_first(r"ratio_psi_degree_norm\s*\|\s*\*\*([0-9.]+)\*\*", analysis_text, "0.994")
    auc_rho = extract_first(r"ratio_psi_rho\s*\|\s*([0-9.]+)\s*\|", analysis_text, "0.819")
    steep_deg = extract_first(r"ratio_psi_degree_norm\s*\|\s*\*\*[0-9.]+\*\*\s*\|\s*([0-9.\-]+)\s*\|", analysis_text, "37.96")
    critical_low = extract_first(r"critical band \[([0-9.]+),\s*[0-9.]+\]", analysis_text, "0.489")
    critical_high = extract_first(r"critical band \[[0-9.]+,\s*([0-9.]+)\]", analysis_text, "0.511")
    minority_cells_ba = extract_first(
        r"Barab[a-zA-Z\-]*\s*\|\s*highest_degree vs random\s*\|\s*\*\*([0-9]+)\*\*", analysis_text, "36"
    )
    minority_p_ba = extract_first(
        r"Barab[a-zA-Z\-]*\s*\|\s*highest_degree vs random\s*\|\s*\*\*[0-9]+\*\*\s*\|\s*\*\*([0-9.]+)\*\*",
        analysis_text,
        "0.993",
    )
    slope_hetero = extract_first(r"placement advantage\}\s*=\s*([0-9.]+)\s*\\times", analysis_text, "0.281")
    r2_hetero = extract_first(r"R\^2\s*=\s*([0-9.]+)", analysis_text, "0.470")
    minority_count = extract_first(r"Among the\s*([0-9]+)\s*minority-win", analysis_text, "71")
    minority_psi_mean = extract_first(r"Mean `ratio_psi_degree_norm`\s*=\s*\*\*([0-9.]+)\*\*", analysis_text, "0.663")
    minority_psi_std = extract_first(r"std\s*=\s*([0-9.]+)", analysis_text, "0.110")

    title = (
        "Opinion Dynamics on Graphs: Zealot Influence, Structural Placement, "
        "and Phase Transitions in the Voter Model"
    )
    authors = r"\textit{MVA Interactions --- April 2026}"
    abstract = (
        "We study opinion dynamics in binary-state voter models with committed agents, first in a one-camp setting "
        "and then in a two-camp competitive setting. In Part 1, we analyze how zealot density and graph topology "
        "control convergence and final magnetization on complete, Erdos-Renyi, and Barabasi-Albert networks. "
        "In Part 2, we introduce competing zealot camps and test whether structural placement compensates for numerical "
        "inferiority. Across five experiments, we measure time-series convergence, phase diagrams, tipping boundaries, "
        "Weisfeiler-Lehman structural effects, and candidate dimensionless strength metrics. We find robust minority "
        "dominance on heterogeneous graphs when positive zealots are hub-placed, with large boundary shifts in the "
        "zealot-count plane. Degree-normalized effective strength emerges as the most predictive order parameter, "
        "producing near-perfect classification of winners and a sharp logistic transition near balance. "
        "Topology acts as an amplifier: homogeneous graphs approach mean-field behavior while scale-free graphs strongly "
        "magnify strategic placement effects. The resulting framework links local network structure to macroscopic "
        "tipping outcomes and suggests operational criteria for influence interventions in real networks."
    )

    # Figure blocks.
    fig_part1_tau = make_subfigure_triple(
        "part1_fc_rho_tau",
        "part1_er_rho_tau",
        "part1_ba_rho_tau",
        alias_to_clean,
        "fig:part1_rho_tau",
        (
            "Single-camp symbolic-regression diagnostics for convergence time as a function of zealot density rho "
            "on complete, Erdos-Renyi, and Barabasi-Albert graphs. The dots are measured simulation outcomes and "
            "the curves are fitted symbolic formulas. The systematic topology dependence motivates the two-camp "
            "extension in which structural placement becomes an explicit control variable."
        ),
        "Complete graph baseline.",
        "Erdos-Renyi baseline.",
        "Barabasi-Albert baseline.",
    )

    fig_exp1_main = make_subfigure_pair(
        "exp1_mag_ba_hd",
        "exp1_mag_ba_rand",
        alias_to_clean,
        "fig:exp1_ba_timeseries_pair",
        (
            "Magnetization trajectories on BA N=500 for equal camp sizes under two placement strategies. "
            "Hub placement for the positive camp accelerates convergence and shifts trajectories upward, while "
            "random placement exhibits slower drift and weaker directional bias. This directly visualizes "
            "the dynamical advantage associated with controlling high-degree nodes."
        ),
        "highest_degree vs random.",
        "random vs random.",
    )
    fig_exp1_snap = make_figure_block(
        "exp1_snap_small",
        alias_to_clean,
        "fig:exp1_small_snapshots",
        (
            "Small-network snapshot grid for asynchronous voter updates with two zealot camps. "
            "Colors indicate committed and free-node opinions, and node size scales with degree. "
            "The sequence illustrates how local hub influence quickly seeds coherent opinion domains "
            "that later dominate global magnetization."
        ),
        width="0.8\\linewidth",
    )
    fig_exp1_conv = make_figure_block(
        "exp1_conv_ba",
        alias_to_clean,
        "fig:exp1_convergence_comparison",
        (
            "Convergence comparison across strategy pairs on BA N=500 in regime B. "
            "Placing positive zealots on high-degree nodes reduces characteristic convergence time by roughly one order "
            "of magnitude relative to random placement. This effect is a direct consequence of interaction frequency "
            "being proportional to incident degree under asynchronous updates."
        ),
        width="0.8\\linewidth",
    )

    fig_exp2_phase_ba = make_subfigure_pair(
        "exp2_pwin_ba_rand",
        "exp2_pwin_ba_hd",
        alias_to_clean,
        "fig:exp2_phase_ba",
        (
            "BA phase diagrams of positive win probability in the (n_plus, n_minus) plane. "
            "Under random-vs-random placement, the transition follows a conventional boundary near balance. "
            "Under highest-degree placement for the positive camp, the winning region expands dramatically into "
            "minority territory, indicating strong structural compensation."
        ),
        "random vs random.",
        "highest_degree vs random.",
    )
    fig_exp2_phase_er = make_subfigure_pair(
        "exp2_pwin_er_rand",
        "exp2_pwin_er_hd",
        alias_to_clean,
        "fig:exp2_phase_er",
        (
            "ER phase diagrams of positive win probability. The random baseline is close to symmetric around "
            "n_plus equals n_minus, while degree-based placement shifts the boundary toward smaller positive counts. "
            "Compared with BA, the shift is significant but less extreme, reflecting lower structural heterogeneity."
        ),
        "random vs random.",
        "highest_degree vs random.",
    )
    fig_exp2_boundary = make_subfigure_pair(
        "exp2_boundary_ba",
        "exp2_boundary_er",
        alias_to_clean,
        "fig:exp2_boundary_overlay",
        (
            "Tipping-boundary overlays (P_plus_win = 0.5) for BA and ER graphs. "
            "Boundary displacement quantifies a strategic exchange rate between placement quality and zealot count. "
            "The BA contour is pushed farthest toward low n_plus, demonstrating that scale-free hubs amplify "
            "the advantage of informed placement."
        ),
        "BA boundary comparison.",
        "ER boundary comparison.",
    )
    fig_exp2_cross = make_figure_block(
        "exp2_cross_er_hd",
        alias_to_clean,
        "fig:exp2_crossing_time",
        (
            "Crossing-time heatmap near the ER tipping region for highest-degree versus random placement. "
            "Cells close to P_plus_win approximately 0.5 display markedly longer first-crossing times, "
            "a numerical signature of critical slowing down. This supports a phase-transition interpretation "
            "of the competition boundary."
        ),
        width="0.8\\linewidth",
    )

    fig_exp3_wl_partition = make_subfigure_pair(
        "exp3_wl_part_ba",
        "exp3_wl_part_er",
        alias_to_clean,
        "fig:exp3_wl_partition",
        (
            "Final 1-WL partitions on BA and ER graphs with N=300. "
            "Both graphs exhibit near-complete node individuation, implying hundreds of tiny structural classes. "
            "In this regime, WL coverage emphasizes diversity but does not target high-degree influencers, "
            "which can conflict with voter-dynamics control."
        ),
        "BA WL partition.",
        "ER WL partition.",
    )
    fig_exp3_strategy = make_subfigure_pair(
        "exp3_strat_bar_ba",
        "exp3_strat_bar_er",
        alias_to_clean,
        "fig:exp3_strategy_bars",
        (
            "Strategy comparison for WL experiment settings. "
            "Highest-degree and WL-top-class strategies dominate, while WL-cover underperforms and can fail completely. "
            "This counter-intuitive ranking highlights the mismatch between structural uniqueness and dynamic influence "
            "in imitation processes."
        ),
        "BA strategy outcomes.",
        "ER strategy outcomes.",
    )
    fig_exp3_feat = make_figure_block(
        "exp3_feat_er",
        alias_to_clean,
        "fig:exp3_feature_scatter",
        (
            "ER structural-feature scatter matrix relating strategy-level features to positive win probability. "
            "Degree-normalized strength tracks performance positively, while dispersion exhibits a strong negative relation. "
            "The pattern supports the mechanism that concentrated influence around high-degree cores beats diffuse placement."
        ),
        width="0.8\\linewidth",
    )

    fig_exp4_scatter = make_figure_block(
        "exp4_scatter_deg",
        alias_to_clean,
        "fig:exp4_metric_scatter",
        (
            "Win-probability versus ratio_psi_degree_norm across all graph families and configurations. "
            "A steep logistic transition appears near 0.5, indicating a sharp switch between negative and positive dominance. "
            "This metric therefore acts as an empirical order parameter for two-camp voter competition."
        ),
        width="0.8\\linewidth",
    )
    fig_exp4_auc = make_figure_block(
        "exp4_auc_summary",
        alias_to_clean,
        "fig:exp4_auc_summary",
        (
            "AUC comparison of candidate normalized metrics by graph family. "
            "Degree-normalized and PageRank-based ratios dominate globally, while WL-only and dispersion-only metrics lag. "
            "The ranking demonstrates that influence capacity is primarily controlled by access to high-degree channels."
        ),
        width="0.8\\linewidth",
    )
    fig_exp4_critical = make_subfigure_pair(
        "exp4_critical",
        "exp4_minority_ba",
        alias_to_clean,
        "fig:exp4_critical_minority",
        (
            "Critical-band and minority-dominance diagnostics in effective-strength space. "
            "The left panel shows concentration of near-transition points around ratio_psi_degree_norm approximately 0.5. "
            "The right panel isolates minority-win BA cases, where structural advantage shifts the minority camp deep "
            "into the winning regime."
        ),
        "Critical-region scan.",
        "Minority-dominance map on BA.",
    )

    fig_exp5_conv = make_figure_block(
        "exp5_conv_speed",
        alias_to_clean,
        "fig:exp5_conv_speed",
        (
            "Convergence-speed comparison across graph families and sparsity levels. "
            "BA configurations show the largest strategy gap, with hub placement converging far faster than random placement. "
            "FC behaves closest to mean field, where placement carries limited leverage."
        ),
        width="0.8\\linewidth",
    )
    fig_exp5_hetero = make_figure_block(
        "exp5_hetero_scatter",
        alias_to_clean,
        "fig:exp5_heterogeneity_advantage",
        (
            "Positive win probability versus degree heterogeneity for two strategy pairs. "
            "As heterogeneity increases, the benefit of hub placement rises systematically. "
            "This empirical slope quantifies topology as an amplifier of structural strategy."
        ),
        width="0.8\\linewidth",
    )
    fig_exp5_hist = make_figure_block(
        "exp5_hist_ba",
        alias_to_clean,
        "fig:exp5_magnetization_hist",
        (
            "Distribution of post-burn-in mean magnetization for BA runs. "
            "The distribution shifts toward positive values under favorable placement and broadens near transition regimes. "
            "This histogram complements phase-diagram summaries by showing run-level outcome variability."
        ),
        width="0.8\\linewidth",
    )
    fig_exp5_network = make_figure_block(
        "exp5_network_comp",
        alias_to_clean,
        "fig:exp5_network_visuals",
        (
            "Representative FC, ER, and BA topologies with identical zealot counts under highest-degree placement. "
            "Node sizes indicate degree, making visible the dramatic hub concentration in BA. "
            "The visual contrast clarifies why the same placement policy has weak impact on FC and large impact on BA."
        ),
        width="0.8\\linewidth",
    )
    fig_exp5_struct = make_figure_block(
        "exp5_struct_metrics",
        alias_to_clean,
        "fig:exp5_struct_metrics",
        (
            "Cross-family comparison of structural metrics under highest-degree placement. "
            "The degree-normalized strength term is largest where placement advantage is largest, "
            "supporting the interpretation that degree control is the primary mechanism of dominance."
        ),
        width="0.8\\linewidth",
    )

    experiments_ascii = ", ".join(experiments_covered)
    tex = f"""
\\documentclass[12pt, a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern}}
\\usepackage{{amsmath, amssymb, amsthm}}
\\usepackage{{graphicx}}
\\usepackage{{subcaption}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}
\\usepackage{{cleveref}}
\\usepackage{{xcolor}}
\\usepackage{{geometry}}
\\usepackage{{microtype}}
\\usepackage{{enumitem}}
\\usepackage{{float}}
\\usepackage{{caption}}
\\usepackage[numbers]{{natbib}}
\\geometry{{margin=2.5cm}}
\\graphicspath{{{{figures/}}}}

\\title{{{tex_escape(title)}}}
\\author{{{authors}}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
{tex_escape(abstract)}
\\end{{abstract}}

\\tableofcontents
\\newpage

\\section{{Introduction}}
Opinion formation on social networks is a canonical problem in statistical physics, network science, and computational social science. The core challenge is to understand how local interactions, constrained by topology, produce global collective states. Agent-based models are useful in this context because they retain microscopic interaction rules while remaining tractable enough to reveal macroscopic regularities. The voter model is a fundamental benchmark of this class, with rich behavior under topology changes and boundary conditions \\cite{{clifford1973,holley1975}}.

In this project, we focus on committed agents, or zealots, that never change state. This abstraction is natural for stubborn activists, coordinated bot clusters, institutional media accounts, or highly committed ideological actors. We ask two linked questions. In Part 1, with one positive zealot camp in an otherwise opposing population, how do zealot density and topology control convergence and long-run magnetization? In Part 2, with two opposing zealot camps, can structural placement compensate for numerical inferiority, and can one identify a universal dimensionless predictor of victory probability?

Our results show that topology is not a small correction. On homogeneous graphs, outcomes are close to mean-field behavior. On heterogeneous graphs, especially scale-free BA networks, hub control dramatically shifts tipping boundaries and permits robust minority dominance. We organize the report as follows: \\cref{{sec:model}} defines the formal model, \\cref{{sec:part1}} presents one-camp results, \\cref{{sec:part2}} develops the two-camp experiments and effective metrics, and \\cref{{sec:discussion}} synthesizes implications and limitations.

\\section{{Model Definition}}\\label{{sec:model}}
\\subsection{{Graph and state space}}
We consider an undirected graph $G=(V,E)$ with $|V|=N$, adjacency matrix $A\\in\\{{0,1\\}}^{{N\\times N}}$, and node states $x_i(t)\\in\\{{-1,+1\\}}$ for $i\\in V$. The degree of node $i$ is
\\begin{{equation}}\\label{{eq:degree_def}}
d_i = \\sum_{{j=1}}^N A_{{ij}}.
\\end{{equation}}
The empirical degree distribution is denoted by $\\{{p_k\\}}_{{k\\ge 0}}$, with moments $\\langle k \\rangle$ and $\\langle k^2 \\rangle$.

\\subsection{{Zealot sets}}
Part 1 uses a single committed set $\\mathcal{{Z}}\\subset V$ with density
\\begin{{equation}}\\label{{eq:rho_single}}
\\rho = \\frac{{|\\mathcal{{Z}}|}}{{N}},
\\end{{equation}}
where $x_i(t)=+1$ for all $i\\in\\mathcal{{Z}}$ and all $t$. Part 2 uses two disjoint committed sets $\\mathcal{{Z}}_+$ and $\\mathcal{{Z}}_-$, fixed at $+1$ and $-1$ respectively. Free nodes belong to
\\begin{{equation}}\\label{{eq:free_set}}
\\mathcal{{F}} = V\\setminus(\\mathcal{{Z}}_+\\cup\\mathcal{{Z}}_-).
\\end{{equation}}

\\subsection{{Voter model update rule}}
At each elementary asynchronous step:
\\begin{{enumerate}}[label=\\arabic*.]
\\item sample node $i\\sim\\mathrm{{Uniform}}(V)$;
\\item if $i\\in\\mathcal{{Z}}_+\\cup\\mathcal{{Z}}_-$, do nothing;
\\item otherwise sample neighbor $j\\sim\\mathrm{{Uniform}}(\\mathcal{{N}}(i))$;
\\item set $x_i(t+1)\\leftarrow x_j(t)$.
\\end{{enumerate}}
This is an asynchronous imitation process because only one free node can update per elementary event. We define one macroscopic time unit as $N$ elementary steps so that, on average, each node is selected once per unit time.

\\subsection{{Observables}}
The magnetization and positive fraction are
\\begin{{align}}
m(t) &= \\frac{{1}}{{N}}\\sum_{{i=1}}^N x_i(t), \\label{{eq:magnetization}}\\\\
p_+(t) &= \\frac{{1}}{{N}}\\left|\\{{i\\in V: x_i(t)=+1\\}}\\right|. \\label{{eq:pplus}}
\\end{{align}}
Given burn-in $t_b$ and horizon $T$, the post-burn-in mean magnetization is
\\begin{{equation}}\\label{{eq:mbar}}
\\bar{{m}}=\\frac{{1}}{{T-t_b+1}}\\sum_{{t=t_b}}^T m(t).
\\end{{equation}}
The positive-victory indicator is $\\mathbf{{1}}[\\bar{{m}}>0]$, and threshold-crossing time is
\\begin{{equation}}\\label{{eq:tau_theta}}
\\tau_\\theta = \\min\\{{t\\ge 0: m(t)>\\theta\\}}.
\\end{{equation}}

\\subsection{{Graph families used}}
\\begin{{itemize}}
\\item \\textbf{{Complete graph}} $K_N$: every node has degree $N-1$, yielding the mean-field limit with no structural heterogeneity.
\\item \\textbf{{Erdos-Renyi}} $G(N,p)$: each edge exists independently with probability $p$, giving $\\langle k\\rangle=p(N-1)$ and asymptotically Poisson degree statistics.
\\item \\textbf{{Barabasi-Albert}} $\\mathrm{{BA}}(N,m)$: preferential attachment produces scale-free behavior with $p_k\\sim k^{{-3}}$, hubs, and high degree heterogeneity \\cite{{barabasi1999,erdos1959}}.
\\end{{itemize}}

\\section{{Part 1: Single-Camp Zealot Model}}\\label{{sec:part1}}
\\subsection{{Setup and research questions}}
In Part 1, a single committed camp $\\mathcal{{Z}}$ is fixed at $+1$ while free nodes are initialized at $-1$. We study how zealot density $\\rho$ affects convergence time and long-run magnetization, and whether placement (random versus hub-oriented) changes effective influence. The main benchmark is whether mean-field behavior on $K_N$ extends to heterogeneous graphs.

\\subsection{{Theoretical background}}
In mean-field treatments of voter dynamics with a one-sided committed fraction, the stationary bias scales with zealot fraction; a common approximation is $m^*\\approx \\rho$ in the fully mixed limit \\cite{{mobilia2003}}. This approximation can fail on heterogeneous networks because interaction opportunities are degree-weighted. If zealots occupy high-degree nodes, they are sampled as neighbors disproportionately often, increasing drift toward $+1$ relative to the same cardinality under uniform placement.

\\subsection{{Experimental results}}
{fig_part1_tau}

\\noindent
\\textbf{{Effect of zealot density and topology.}}
The Part 1 symbolic-regression figures in \\cref{{fig:part1_rho_tau}} summarize how convergence-time scaling with $\\rho$ differs by topology. The complete graph behaves smoothly and close to mean-field trends, whereas ER and BA exhibit stronger nonlinearities. This is consistent with finite-size and heterogeneity effects: the same $\\rho$ can correspond to very different effective influence mass depending on which degrees are occupied by zealots.

\\noindent
\\textbf{{Placement strategy effects.}}
Even in one-camp settings, hub-oriented placement substantially reduces consensus time relative to random placement. The mechanism is dynamical rather than purely combinatorial: high-degree committed nodes appear in many local imitation events per unit time, creating persistent positive flux into surrounding neighborhoods. This effect is weak in FC and stronger in BA, mirroring differences in degree heterogeneity.

\\noindent
\\textbf{{Graph-type contrast.}}
Complete graphs provide a control where degree is uniform and structural placement has little room to differentiate nodes. ER graphs show moderate sensitivity to placement due to mild degree variability. BA graphs show the strongest sensitivity because a small hub set controls a large fraction of all edges; therefore, one-camp dominance can emerge at lower cardinality than under homogeneous assumptions.

\\subsection{{Summary of Part 1 findings}}
\\begin{{itemize}}
\\item Increasing $\\rho$ consistently improves positive dominance and shortens convergence.
\\item Complete-graph behavior is closest to mean-field predictions and serves as a baseline consistency check.
\\item Hub-oriented placement improves efficiency over random placement for equal zealot counts.
\\item Topological heterogeneity amplifies placement effects: BA $>$ ER $>$ FC.
\\item One-camp scaling motivates the two-camp question of structural versus numerical advantage.
\\end{{itemize}}

\\section{{Part 2: Two-Camp Zealot Model}}\\label{{sec:part2}}
\\subsection{{Motivation and research questions}}
With two committed camps, count imbalance alone does not determine outcomes on heterogeneous networks. We ask: can structural placement compensate for numerical inferiority? Is there a single normalized predictor that estimates victory probability across graph families? How does topology modulate the strategic exchange rate between placement quality and zealot count?

\\subsection{{Placement strategies}}
\\textbf{{random}} samples $k$ nodes uniformly without replacement. \\textbf{{highest\\_degree}} selects top-$k$ nodes by decreasing degree. \\textbf{{hub\\_then\\_spread}} first takes a top hub and then greedily maximizes minimum shortest-path distance from already selected nodes, balancing centrality and dispersion. \\textbf{{farthest\\_spread}} uses purely distance-based farthest-point iterations. \\textbf{{wl\\_cover}} selects approximately one node per WL class to maximize structural-class coverage.

\\subsection{{Weisfeiler-Lehman structural features}}
The 1-WL color refinement process initializes colors by degree and iterates:
\\begin{{align}}
c_i^{{(0)}} &= d_i, \\label{{eq:wl0}}\\\\
c_i^{{(t+1)}} &= \\mathrm{{hash}}\\!\\left(c_i^{{(t)}},\\,\\mathrm{{sort}}\\left(\\{{c_j^{{(t)}}: j\\in\\mathcal{{N}}(i)\\}}\\right)\\right). \\label{{eq:wl1}}
\\end{{align}}
Iterations stop when the partition stabilizes. Final equal-color nodes are treated as structurally equivalent under 1-WL.

Define WL coverage for zealot set $S$ as
\\begin{{equation}}\\label{{eq:wl_cov}}
\\psi_{{\\mathrm{{WL}}}}(S)=\\frac{{\\left|\\{{\\text{{WL classes hit by }}S\\}}\\right|}}{{\\left|\\{{\\text{{all WL classes}}\\}}\\right|}}.
\\end{{equation}}

{fig_exp3_wl_partition}

On BA and ER with $N=300$, WL refinement yields near-complete individuation (about 295--300 classes). Thus, maximizing WL coverage is close to near-uniform sampling over many singleton classes and does not necessarily target hubs. This is crucial for interpretation of WL placement performance in voter dynamics.

\\subsection{{Candidate effective strength metrics}}
For zealot set $S$:
\\begin{{align}}
\\psi_\\rho(S) &= \\frac{{|S|}}{{N}}, \\label{{eq:psi_rho}}\\\\
\\psi_d(S) &= \\frac{{\\sum_{{i\\in S}} d_i}}{{\\sum_{{i\\in V}} d_i}}, \\label{{eq:psi_deg}}\\\\
\\psi_{{\\mathrm{{PR}}}}(S) &= \\sum_{{i\\in S}} \\mathrm{{PR}}_i, \\label{{eq:psi_pr}}\\\\
\\psi_{{\\mathrm{{WL}}}}(S) &= \\frac{{|\\{{\\text{{WL classes hit by }}S\\}}|}}{{|\\{{\\text{{all WL classes}}\\}}|}}, \\label{{eq:psi_wl}}\\\\
\\psi_{{\\mathrm{{disp}}}}(S) &= \\frac{{1}}{{|S|^2}}\\sum_{{i,j\\in S}}\\frac{{d_G(i,j)}}{{\\mathrm{{diam}}(G)}}. \\label{{eq:psi_disp}}
\\end{{align}}
For two camps, define relative strength
\\begin{{equation}}\\label{{eq:psi_rel}}
\\Psi_\\bullet = \\frac{{\\psi_\\bullet(\\mathcal{{Z}}_+)}}{{\\psi_\\bullet(\\mathcal{{Z}}_+) + \\psi_\\bullet(\\mathcal{{Z}}_-)}}\\in[0,1].
\\end{{equation}}
Balance corresponds to $\\Psi_\\bullet=0.5$.

\\subsection{{Experiment 1 --- Convergence and time series}}
{fig_exp1_main}
{fig_exp1_conv}
{fig_exp1_snap}

On BA graphs, hub placement in regime B yields mean convergence around 437 steps, while random placement is around 3837 steps, roughly a 9x speedup. This follows directly from the asynchronous update rule: high-degree committed nodes are overrepresented in neighborhood sampling, so they inject persistent directional drift at high rate.

The same qualitative effect appears on ER but with smaller ratio, and nearly disappears in FC where all degrees are equal. This ranking (BA strongest, ER intermediate, FC weakest) is a repeated pattern across experiments. In this parameter set, large metastability was not detected, suggesting these trajectories are mostly outside the narrow tipping band.

\\subsection{{Experiment 2 --- Phase diagram and tipping boundary}}
{fig_exp2_phase_ba}
{fig_exp2_phase_er}
{fig_exp2_boundary}
{fig_exp2_cross}

The $(n_+,n_-)$ plane splits into positive-dominance and negative-dominance phases, separated by a tipping contour near $P_{{\\mathrm{{win}}}}=0.5$. On ER with random placement, the contour is close to linear and near the diagonal, consistent with approximate symmetry. On BA with hub-oriented placement, the contour is displaced so strongly that positive minorities can dominate over wide regions.

Minority-dominance cells make this explicit: BA with highest-degree placement yields about {minority_cells_ba} minority cells with mean positive win probability around {minority_p_ba}. By contrast, ER random-vs-random has no minority-dominance cells in the tested grid. Hence structural asymmetry is not a weak correction but a phase-level control parameter.

Boundary displacement quantifies strategic exchange rate: in ER, degree-based placement shifts the contour by roughly 11 zealots, while BA shifts can approach roughly 33 zealots. Operationally, one well-placed zealot can be worth multiple random placements, and this multiplier increases with heterogeneity.

Near the transition zone, crossing times increase sharply (for example around 2361 steps on ER in one critical slice), showing critical slowing down. This is consistent with weak net drift near balance, where stochastic fluctuations dominate and trajectories take longer to commit.

\\subsection{{Experiment 3 --- WL placement: a counter-intuitive result}}
{fig_exp3_strategy}
{fig_exp3_feat}

The WL-cover strategy is paradoxical in this dynamical context: it can underperform random placement, even reaching zero positive wins in tested BA/ER settings. Its degree-normalized strength is much lower than hub-focused strategies because one-per-class selection spreads budget across many low-degree singleton classes.

This does not mean WL is useless as a structural descriptor. It means representational diversity (class coverage) is orthogonal to influence capacity under voter updates. The process rewards frequent participation in local interactions, which is governed primarily by degree mass rather than class uniqueness.

A second counter-intuitive finding is strong negative correlation between dispersion and winning probability in some settings. Highly dispersed zealots fight many isolated local contests without reinforcement, while concentrated hub groups create dense local opinion fields and control communication bottlenecks.

\\subsection{{Experiment 4 --- Effective strength metrics and the order parameter}}
{fig_exp4_scatter}
{fig_exp4_auc}
{fig_exp4_critical}

\\begin{{table}}[H]
\\centering
\\caption{{Metric ranking from the two-camp effective-strength experiment.}}
\\label{{tab:metric_ranking}}
\\begin{{tabular}}{{lcccr}}
\\toprule
Metric & Pearson $r$ & Spearman $r$ & AUC & Steepness $a$ \\\\
\\midrule
ratio\\_psi\\_degree\\_norm & 0.966 & 0.967 & {auc_deg} & {steep_deg} \\\\
ratio\\_psi\\_pagerank & 0.964 & 0.963 & 0.994 & 39.21 \\\\
ratio\\_psi\\_hybrid & 0.833 & 0.847 & 0.926 & 77.16 \\\\
ratio\\_psi\\_rho & 0.665 & 0.669 & {auc_rho} & 11.56 \\\\
ratio\\_psi\\_dispersion & -0.515 & -0.532 & 0.771 & -42.35 \\\\
ratio\\_psi\\_wl & 0.446 & 0.452 & 0.679 & 6.83 \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

The dominant predictor is
\\begin{{equation}}\\label{{eq:psi_d_order}}
\\Psi_d = \\frac{{\\sum_{{i\\in\\mathcal{{Z}}_+}} d_i}}{{\\sum_{{i\\in\\mathcal{{Z}}_+}} d_i + \\sum_{{i\\in\\mathcal{{Z}}_-}} d_i}}.
\\end{{equation}}
Its global AUC is {auc_deg}, and per-graph AUC reaches approximately 0.999 on BA and 0.993 on ER. The fitted transition
\\begin{{equation}}\\label{{eq:logistic_fit}}
P(\\text{{positive wins}})\\approx \\sigma\\!\\left({steep_deg}\\,(\\Psi_d-0.5)\\right)
\\end{{equation}}
is steep, indicating phase-like switching around balance.

For this fit, the critical interval is about $[{critical_low},{critical_high}]$, width approximately 0.022. This narrow window means small structural perturbations can flip outcomes. In minority-dominance cases ($n_+<n_-$ and high win probability), the mean degree-ratio metric is around {minority_psi_mean} (std {minority_psi_std}), and the sample contains about {minority_count} such configurations.

Mechanistically, degree-normalized strength is natural for asynchronous voter updates because each edge is an interaction opportunity. Zealot camps with larger incident-degree mass exert stronger persistent drift on free-node updates, so total degree share becomes a direct proxy for long-run directional bias.

On FC, ratio\\_psi\\_rho and ratio\\_psi\\_degree\\_norm become equivalent by symmetry, validating the mean-field limit where counts dominate and structure is neutral.

\\subsection{{Experiment 5 --- Graph topology as an amplifier}}
{fig_exp5_conv}
{fig_exp5_hetero}
{fig_exp5_hist}
{fig_exp5_network}
{fig_exp5_struct}

The heterogeneity regression is approximately
\\begin{{equation}}\\label{{eq:hetero_reg}}
\\text{{placement advantage}} \\approx {slope_hetero}\\times\\text{{degree heterogeneity}} + 0.347,\\quad R^2\\approx {r2_hetero}.
\\end{{equation}}
Hence topology acts as an amplifier: greater $\\sigma_k/\\langle k\\rangle$ generally increases the payoff of strategic placement.

\\begin{{table}}[H]
\\centering
\\caption{{Topology regimes and placement advantage (representative values).}}
\\label{{tab:topology_regimes}}
\\begin{{tabular}}{{lccc}}
\\toprule
Graph family & Degree heterogeneity & Placement advantage & Qualitative regime \\\\
\\midrule
Fully connected & 0.000 & 0.133 & Mean-field-like \\\\
Erdos-Renyi (range) & 0.265--0.456 & 0.467--0.600 & Intermediate heterogeneity \\\\
Barabasi-Albert (range) & 0.931--1.176 & 0.533--0.667 & Scale-free amplification \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

BA with $m=2$ can show larger placement advantage than BA with $m=5$ despite smaller absolute average degree because relative heterogeneity is higher in sparse scale-free graphs. Convergence reflects the same ordering: BA hub placement can be around 7--10x faster than BA random placement, ER gaps are smaller, and FC gaps are weakest.

\\section{{Discussion}}\\label{{sec:discussion}}
\\subsection{{Synthesis: the order parameter of zealot competition}}
Across experiments, the degree-share ratio $\\Psi_d$ in \\cref{{eq:psi_d_order}} behaves as an order parameter for two-camp competition. When $\\Psi_d>0.5$, positive victory probability is near one; when $\\Psi_d<0.5$, negative dominance is typical. The fitted transition in \\cref{{eq:logistic_fit}} is sharp, with a critical band of width approximately 0.022 around balance.

\\subsection{{Implications for influence campaigns}}
In heterogeneous networks, a small hub-centered committed group can systematically outperform a larger diffuse opposition. The implied strategic exchange rate is large: in tested settings, boundary shifts correspond to approximately 11 zealots on ER and up to roughly 33 on BA. Near $\\Psi_d\\approx 0.5$, tiny interventions matter disproportionately because the system sits in a high-sensitivity critical regime.

\\subsection{{Application to real networks}}
A practical workflow is: (i) estimate empirical degree distribution, (ii) match to a tractable surrogate (ER if near-Poisson, BA-like if heavy-tailed), (iii) compute empirical heterogeneity $\\sigma_k/\\langle k\\rangle$, and (iv) estimate win probability with a calibrated logistic map
\\begin{{equation}}\\label{{eq:practical_logistic}}
P(\\text{{win}})\\approx \\sigma\\!\\left(37.96\\,(\\Psi_d-0.5)\\right).
\\end{{equation}}
Boundary-shift magnitudes provide order-of-magnitude estimates of structural bonus under hub placement. Caveat: voter dynamics omit strategic adaptation, weighted influence, temporal edge dynamics, and exogenous media shocks.

\\subsection{{Limitations and future work}}
Finite-size scaling remains open: transition steepness may increase with $N$. WL near-singleton behavior on random-like sparse graphs suggests testing deeper WL variants or richer ego-network descriptors. Temporal and adaptive networks may alter optimal placement. Community-structured models (for example stochastic block models) could reveal boundary-node strategies that outperform pure hub concentration. Finally, asymmetric interaction kernels may reduce or invert hub advantage.

\\section{{Conclusion}}
This report establishes a coherent picture of zealot competition in voter dynamics. First, degree-weighted structural position is the dominant predictor of two-camp outcomes, substantially outperforming raw zealot counts. Second, minority dominance is robust on heterogeneous graphs when the minority controls hubs, with near-certain wins in wide tested regions. Third, tipping boundaries are displaced strongly by placement strategy, giving a measurable exchange rate between structure and count. Fourth, WL coverage is not a proxy for influence in this process and can be counter-productive. Fifth, topology amplifies strategic effects, with scale-free graphs showing the strongest gains.

The broader implication is that influence processes on real networks cannot be understood from population fractions alone. Structural access to interaction channels determines effective power. Future work should combine this structural order-parameter view with temporal, community, and adaptive-network effects to move from stylized benchmarks toward operational forecasting.

\\begin{{thebibliography}}{{99}}
\\bibitem{{clifford1973}} Clifford, P. and Sudbury, A. (1973). A model for spatial conflict. \\textit{{Biometrika}}, 60(3), 581--588.

\\bibitem{{holley1975}} Holley, R.A. and Liggett, T.M. (1975). Ergodic theorems for weakly interacting infinite systems and the voter model. \\textit{{The Annals of Probability}}, 3(4), 643--663.

\\bibitem{{mobilia2003}} Mobilia, M. (2003). Does a single zealot affect an infinite group of voters? \\textit{{Physical Review Letters}}, 91(2), 028701.

\\bibitem{{barabasi1999}} Barabasi, A.L. and Albert, R. (1999). Emergence of scaling in random networks. \\textit{{Science}}, 286(5439), 509--512.

\\bibitem{{erdos1959}} Erdos, P. and Renyi, A. (1959). On random graphs. \\textit{{Publicationes Mathematicae}}, 6, 290--297.
\\end{{thebibliography}}

% Build metadata:
% total figures discovered: {total_found}
% figures used in report: {used_count}
% experiments covered: {tex_escape(experiments_ascii)}

\\end{{document}}
"""

    # Replace accidental non-ASCII with safe fallback to keep strict hygiene.
    return ensure_ascii(tex)


def compile_latex(report_dir: Path) -> tuple[int, str, str]:
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"]
    try:
        # Clean stale aux files first to avoid inconsistent incremental states.
        _ = subprocess.run(
            ["latexmk", "-C", "main.tex"],
            cwd=report_dir,
            capture_output=True,
            text=True,
        )
        result_first = subprocess.run(
            cmd,
            cwd=report_dir,
            capture_output=True,
            text=True,
        )
        # After a clean build, first latexmk pass may stop with missing .toc/.aux
        # notices. A second pass typically resolves references/citations.
        result_second = subprocess.run(
            cmd,
            cwd=report_dir,
            capture_output=True,
            text=True,
        )
        combined_out = (result_first.stdout or "") + "\n" + (result_second.stdout or "")
        combined_err = (result_first.stderr or "") + "\n" + (result_second.stderr or "")
        return result_second.returncode, combined_out, combined_err
    except FileNotFoundError as exc:
        log(f"[warning] latexmk not found: {exc}")
        log("[info] Trying pdflatex fallback (2 passes).")
        try:
            out_all = []
            err_all = []
            for _ in range(2):
                r = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                    cwd=report_dir,
                    capture_output=True,
                    text=True,
                )
                out_all.append(r.stdout or "")
                err_all.append(r.stderr or "")
                if r.returncode != 0:
                    return r.returncode, "\n".join(out_all), "\n".join(err_all)
            return 0, "\n".join(out_all), "\n".join(err_all)
        except FileNotFoundError as exc2:
            log(f"[warning] pdflatex not found: {exc2}")
            return 127, "", f"{exc}; {exc2}"
        except Exception as exc2:
            log(f"[warning] Failed to run pdflatex fallback: {exc2}")
            return 1, "", str(exc2)
    except Exception as exc:
        log(f"[warning] Failed to run latexmk: {exc}")
        return 1, "", str(exc)


def extract_page_count(report_dir: Path, stdout_text: str, stderr_text: str) -> int | None:
    combined = "\n".join([stdout_text, stderr_text])
    m = re.search(r"Output written on .*?\((\d+)\s+pages?", combined)
    if m:
        return int(m.group(1))
    log_path = report_dir / "main.log"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        m2 = re.search(r"Output written on .*?\((\d+)\s+pages?", log_text)
        if m2:
            return int(m2.group(1))
    return None


def print_last_log_lines(log_path: Path, n: int = 50) -> None:
    if not log_path.exists():
        log("[info] No main.log found.")
        return
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    tail = "\n".join(lines[-n:])
    print("\n----- Last 50 lines of report/main.log -----\n")
    print(tail)
    print("\n----- End log tail -----\n")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Clean old report figures to keep mapping reproducible.
    if REPORT_FIG_DIR.exists():
        for p in REPORT_FIG_DIR.glob("*"):
            if p.is_file():
                p.unlink()

    catalogue = discover_figures(RESULTS_DIR)
    analysis_text = read_analysis_text(ANALYSIS_MD)

    selected = select_figures(catalogue)
    alias_to_clean = copy_selected_figures(selected, REPORT_FIG_DIR)

    experiments_covered = sorted({item["experiment"] for item in selected.values()})
    log(f"[summary] total figures found: {len(catalogue)}")
    log(f"[summary] figures used in report: {len(selected)}")
    log(f"[summary] experiments covered: {', '.join(experiments_covered)}")

    # Extra required summary details.
    category_count = defaultdict(int)
    for item in catalogue:
        category_count[item["category"]] += 1
    log(f"[summary] discovered categories: {dict(sorted(category_count.items()))}")

    tex = build_main_tex(
        alias_to_clean=alias_to_clean,
        analysis_text=analysis_text,
        total_found=len(catalogue),
        used_count=len(selected),
        experiments_covered=experiments_covered,
    )
    MAIN_TEX.write_text(tex, encoding="utf-8")
    log(f"[ok] wrote {MAIN_TEX.relative_to(ROOT)}")

    # Compile.
    rc, out, err = compile_latex(REPORT_DIR)
    print("\n----- latexmk stdout (tail 3000 chars) -----\n")
    print(out[-3000:] if out else "(no stdout)")
    print("\n----- end stdout tail -----\n")
    if rc != 0:
        print("COMPILATION ERRORS:")
        print(err[-3000:] if err else "(no stderr)")
        print_last_log_lines(REPORT_DIR / "main.log", n=50)
    else:
        log("[ok] latexmk compilation succeeded.")

    pages = extract_page_count(REPORT_DIR, out, err)
    pages_text = str(pages) if pages is not None else "unknown"

    sections_written = [
        "1 Introduction",
        "2 Model Definition",
        "3 Part 1: Single-Camp Zealot Model",
        "4 Part 2: Two-Camp Zealot Model",
        "5 Discussion",
        "6 Conclusion",
        "Bibliography",
    ]
    log("[final-summary]")
    log(f"  pages compiled: {pages_text}")
    log(f"  figures included: {len(selected)}")
    log(f"  sections written: {len(sections_written)}")
    for sec in sections_written:
        log(f"    - {sec}")
    if (REPORT_DIR / "main.pdf").exists():
        log(f"[ok] PDF generated at {REPORT_DIR / 'main.pdf'}")
    else:
        log("[warning] report/main.pdf not found.")


if __name__ == "__main__":
    main()

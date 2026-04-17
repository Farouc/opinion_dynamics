"""Candidate effective-strength measures for two zealot camps."""

from __future__ import annotations

from typing import Iterable

import networkx as nx

from src.graph_features import compute_graph_context, compute_two_camp_comparison_features
from src.wl_features import compute_two_camp_wl_features


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1e-15:
        return float(default)
    return float(num / den)


def _get_feature_value(d: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in d and d[key] is not None:
            return float(d[key])
    return float(default)


def _camp_scores(
    prefix: str,
    comp: dict,
    wl: dict,
    graph_features: dict,
    weights: dict,
) -> dict[str, float]:
    n_nodes = float(graph_features.get("n_nodes", 0.0))
    n_edges = float(graph_features.get("n_edges", 0.0))
    degree_total = max(2.0 * n_edges, 1e-15)

    size = _get_feature_value(comp, f"{prefix}_n_zealots")
    rho = _get_feature_value(comp, f"{prefix}_zealot_fraction", default=_safe_div(size, n_nodes, 0.0))
    deg_sum = _get_feature_value(comp, f"{prefix}_degree_sum")
    pr_sum = _get_feature_value(comp, f"{prefix}_pagerank_sum")

    wl_cov = _get_feature_value(
        wl,
        f"{prefix}_wl_covered_fraction",
        f"{prefix}_wl_coverage_fraction",
        default=_get_feature_value(comp, f"{prefix}_wl_coverage_fraction", default=0.0),
    )

    disp = _get_feature_value(comp, f"{prefix}_dispersion_score")
    disp_norm = float(disp / (1.0 + abs(disp)))

    deg_norm = float(deg_sum / degree_total)

    alpha = float(weights.get("alpha", 0.35))
    beta = float(weights.get("beta", 0.35))
    gamma = float(weights.get("gamma", 0.15))
    delta = float(weights.get("delta", 0.15))

    hybrid = float(alpha * rho + beta * deg_norm + gamma * wl_cov + delta * disp_norm)

    return {
        "psi_size": float(size),
        "psi_rho": float(rho),
        "psi_degree": float(deg_sum),
        "psi_degree_norm": float(deg_norm),
        "psi_centrality": float(pr_sum),
        "psi_wl": float(wl_cov),
        "psi_dispersion": float(disp_norm),
        "psi_hybrid": float(hybrid),
    }


def compute_effective_strength_candidates(
    G: nx.Graph,
    pos_nodes: Iterable[int],
    neg_nodes: Iterable[int],
    feature_bundle: dict | None = None,
    hybrid_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute multiple candidate effective-strength scalars and deltas."""
    if feature_bundle is None:
        context = compute_graph_context(G)
        comp = compute_two_camp_comparison_features(G, pos_nodes, neg_nodes, context=context)
        wl = compute_two_camp_wl_features(
            G,
            pos_nodes=pos_nodes,
            neg_nodes=neg_nodes,
            wl_labels=context["wl_labels"],
        )
        graph_f = context["graph_features"]
    else:
        comp = feature_bundle.get("comparison_features") or feature_bundle.get("camp_comparison_features")
        if comp is None:
            raise ValueError("feature_bundle must include 'comparison_features'.")
        wl = feature_bundle.get("wl_features", {})
        graph_f = feature_bundle.get("graph_features", {})

    weights = dict(hybrid_weights or {})
    pos_scores = _camp_scores("pos", comp=comp, wl=wl, graph_features=graph_f, weights=weights)
    neg_scores = _camp_scores("neg", comp=comp, wl=wl, graph_features=graph_f, weights=weights)

    out: dict[str, float] = {
        **{f"pos_{k}": float(v) for k, v in pos_scores.items()},
        **{f"neg_{k}": float(v) for k, v in neg_scores.items()},
    }

    for key in [
        "psi_size",
        "psi_rho",
        "psi_degree",
        "psi_degree_norm",
        "psi_centrality",
        "psi_wl",
        "psi_dispersion",
        "psi_hybrid",
    ]:
        p = float(pos_scores[key])
        n = float(neg_scores[key])
        out[f"delta_{key}"] = float(p - n)
        out[f"ratio_{key}_pos_over_neg"] = _safe_div(p, n, default=float("inf"))

    return out

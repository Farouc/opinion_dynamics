"""Graph-level and camp-level structural feature extraction."""

from __future__ import annotations

import math
import warnings
from collections import Counter
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.stats import skew

from src.wl_features import wl_color_refinement


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1e-15:
        return float(default)
    return float(num / den)


def _to_float(value: float | int) -> float:
    if value is None:
        return float("nan")
    return float(value)


def compute_graph_features(G: nx.Graph) -> dict[str, float]:
    """Compute global graph structural features."""
    n = G.number_of_nodes()
    e = G.number_of_edges()

    deg = np.asarray([float(d) for _, d in G.degree()], dtype=float)
    mean_deg = float(np.mean(deg)) if deg.size > 0 else 0.0
    var_deg = float(np.var(deg)) if deg.size > 0 else 0.0
    if deg.size >= 3 and np.std(deg) > 1e-12:
        skew_deg = float(skew(deg, bias=False))
    else:
        skew_deg = 0.0

    clustering = float(nx.average_clustering(G)) if n > 0 else 0.0
    density = float(nx.density(G)) if n > 1 else 0.0

    components = list(nx.connected_components(G)) if n > 0 else []
    lcc_size = int(max((len(c) for c in components), default=0))
    lcc_fraction = _safe_div(lcc_size, n, default=0.0)

    is_connected = bool(nx.is_connected(G)) if n > 0 else False
    if is_connected and n > 1:
        avg_path = float(nx.average_shortest_path_length(G))
    elif lcc_size > 1:
        lcc_nodes = max(components, key=len)
        avg_path = float(nx.average_shortest_path_length(G.subgraph(lcc_nodes).copy()))
    else:
        avg_path = float("nan")

    try:
        if n > 1 and np.std(deg) > 1e-12:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                assort = float(nx.degree_assortativity_coefficient(G))
        else:
            assort = float("nan")
    except Exception:
        assort = float("nan")

    lambda2 = _estimate_laplacian_lambda2(G)

    return {
        "n_nodes": float(n),
        "n_edges": float(e),
        "density": float(density),
        "average_degree": float(mean_deg),
        "degree_variance": float(var_deg),
        "degree_skewness": float(skew_deg),
        "degree_moment_1": float(mean_deg),
        "degree_moment_2": float(np.mean(deg**2)) if deg.size > 0 else 0.0,
        "clustering_coefficient": float(clustering),
        "average_shortest_path_length": float(avg_path),
        "largest_component_size": float(lcc_size),
        "largest_component_fraction": float(lcc_fraction),
        "is_connected": float(1.0 if is_connected else 0.0),
        "degree_assortativity": float(assort),
        "laplacian_lambda2": float(lambda2),
    }


def _estimate_laplacian_lambda2(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n < 2:
        return 0.0

    try:
        L = nx.laplacian_matrix(G).astype(float)
        if n <= 1200:
            evals = eigsh(L, k=2, which="SM", return_eigenvectors=False)
            evals = np.sort(np.real(evals))
            return float(evals[1])
    except Exception:
        pass

    if n <= 300:
        try:
            dense = np.asarray(nx.laplacian_matrix(G).todense(), dtype=float)
            evals = np.linalg.eigvalsh(dense)
            evals = np.sort(np.real(evals))
            if evals.size >= 2:
                return float(evals[1])
        except Exception:
            return float("nan")

    return float("nan")


def compute_graph_context(
    G: nx.Graph,
    wl_n_iter: int = 3,
    include_betweenness: bool = False,
    include_eigenvector: bool = False,
    include_communities: bool = True,
) -> dict:
    """Reusable graph context for repeated camp feature computations."""
    degree = {int(n): float(d) for n, d in G.degree()}
    pagerank = {int(n): float(v) for n, v in nx.pagerank(G).items()}

    context: dict = {
        "graph_features": compute_graph_features(G),
        "degree": degree,
        "pagerank": pagerank,
        "wl_labels": wl_color_refinement(G, n_iter=wl_n_iter)["labels"],
    }

    if include_betweenness:
        context["betweenness"] = {int(n): float(v) for n, v in nx.betweenness_centrality(G).items()}
    else:
        context["betweenness"] = None

    if include_eigenvector:
        try:
            ev = nx.eigenvector_centrality_numpy(G)
        except Exception:
            ev = nx.eigenvector_centrality(G, max_iter=500, tol=1e-8)
        context["eigenvector"] = {int(n): float(v) for n, v in ev.items()}
    else:
        context["eigenvector"] = None

    if include_communities:
        communities = list(nx.algorithms.community.greedy_modularity_communities(G))
        labels: dict[int, int] = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                labels[int(node)] = int(cid)
        next_id = len(communities)
        for node in G.nodes():
            if int(node) not in labels:
                labels[int(node)] = int(next_id)
                next_id += 1
        context["community_labels"] = labels
    else:
        context["community_labels"] = None

    return context


def _mean_pairwise_distance(G: nx.Graph, nodes: list[int]) -> float:
    if len(nodes) < 2:
        return 0.0

    total = 0.0
    count = 0
    for i, u in enumerate(nodes[:-1]):
        lengths = nx.single_source_shortest_path_length(G, u)
        for v in nodes[i + 1 :]:
            d = lengths.get(v)
            if d is None:
                continue
            total += float(d)
            count += 1

    if count == 0:
        return float("inf")
    return float(total / count)


def _coverage_from_labels(nodes: list[int], labels: dict[int, int] | None) -> tuple[float, float]:
    if labels is None:
        return 0.0, 0.0
    if not labels:
        return 0.0, 0.0

    total_classes = len(set(int(v) for v in labels.values()))
    if total_classes == 0:
        return 0.0, 0.0

    covered = len({int(labels[int(node)]) for node in nodes}) if nodes else 0
    return float(covered), float(covered / total_classes)


def compute_camp_features(
    G: nx.Graph,
    camp_nodes: Iterable[int],
    context: dict | None = None,
) -> dict[str, float]:
    """Compute structural features for one zealot camp."""
    if context is None:
        context = compute_graph_context(G)

    nodes = sorted({int(n) for n in camp_nodes})
    n = G.number_of_nodes()

    degree = context["degree"]
    pagerank = context["pagerank"]
    graph_avg_path = _to_float(context["graph_features"].get("average_shortest_path_length", float("nan")))

    deg_vals = np.asarray([degree[node] for node in nodes], dtype=float) if nodes else np.array([], dtype=float)
    pr_vals = np.asarray([pagerank[node] for node in nodes], dtype=float) if nodes else np.array([], dtype=float)

    avg_pair_dist = _mean_pairwise_distance(G, nodes)
    if math.isfinite(graph_avg_path) and graph_avg_path > 0 and math.isfinite(avg_pair_dist):
        dispersion = float(avg_pair_dist / graph_avg_path)
    else:
        dispersion = 0.0

    comm_cov_count, comm_cov_frac = _coverage_from_labels(nodes, context.get("community_labels"))
    wl_cov_count, wl_cov_frac = _coverage_from_labels(nodes, context.get("wl_labels"))

    return {
        "n_zealots": float(len(nodes)),
        "zealot_fraction": _safe_div(len(nodes), n, default=0.0),
        "degree_sum": float(np.sum(deg_vals)) if deg_vals.size > 0 else 0.0,
        "degree_mean": float(np.mean(deg_vals)) if deg_vals.size > 0 else 0.0,
        "degree_max": float(np.max(deg_vals)) if deg_vals.size > 0 else 0.0,
        "pagerank_sum": float(np.sum(pr_vals)) if pr_vals.size > 0 else 0.0,
        "pagerank_mean": float(np.mean(pr_vals)) if pr_vals.size > 0 else 0.0,
        "avg_pairwise_distance": float(avg_pair_dist),
        "dispersion_score": float(dispersion),
        "community_coverage_count": float(comm_cov_count),
        "community_coverage_fraction": float(comm_cov_frac),
        "wl_coverage_count": float(wl_cov_count),
        "wl_coverage_fraction": float(wl_cov_frac),
    }


def _mean_dist_free_to_camp(
    G: nx.Graph,
    free_nodes: list[int],
    camp_nodes: list[int],
) -> tuple[float, float]:
    if len(free_nodes) == 0:
        return 0.0, 1.0
    if len(camp_nodes) == 0:
        return float("inf"), 0.0

    try:
        from networkx.algorithms.shortest_paths.unweighted import (
            multi_source_shortest_path_length,
        )

        lengths = multi_source_shortest_path_length(G, camp_nodes)
    except Exception:
        # Compatibility fallback for NetworkX versions without the helper import.
        lengths = nx.multi_source_dijkstra_path_length(G, camp_nodes, weight=None)
    vals = np.asarray([float(lengths.get(node, math.inf)) for node in free_nodes], dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return float("inf"), 0.0

    return float(np.mean(vals[finite])), float(np.mean(finite))


def compute_two_camp_comparison_features(
    G: nx.Graph,
    pos_nodes: Iterable[int],
    neg_nodes: Iterable[int],
    context: dict | None = None,
) -> dict[str, float]:
    """Compute camp-wise features and relative structural comparisons."""
    if context is None:
        context = compute_graph_context(G)

    pos_list = sorted({int(n) for n in pos_nodes})
    neg_list = sorted({int(n) for n in neg_nodes})

    pos = compute_camp_features(G, pos_list, context=context)
    neg = compute_camp_features(G, neg_list, context=context)

    all_nodes = set(int(n) for n in G.nodes())
    free_nodes = sorted(all_nodes.difference(set(pos_list)).difference(set(neg_list)))

    dist_free_pos, free_pos_reach = _mean_dist_free_to_camp(G, free_nodes, pos_list)
    dist_free_neg, free_neg_reach = _mean_dist_free_to_camp(G, free_nodes, neg_list)

    out: dict[str, float] = {
        **{f"pos_{k}": float(v) for k, v in pos.items()},
        **{f"neg_{k}": float(v) for k, v in neg.items()},
        "n_free_nodes": float(len(free_nodes)),
        "mean_dist_free_to_pos": float(dist_free_pos),
        "mean_dist_free_to_neg": float(dist_free_neg),
        "free_to_pos_reachable_fraction": float(free_pos_reach),
        "free_to_neg_reachable_fraction": float(free_neg_reach),
        "distance_advantage_pos": float(dist_free_neg - dist_free_pos)
        if math.isfinite(dist_free_neg) and math.isfinite(dist_free_pos)
        else 0.0,
    }

    diff_keys = [
        "n_zealots",
        "zealot_fraction",
        "degree_sum",
        "degree_mean",
        "degree_max",
        "pagerank_sum",
        "pagerank_mean",
        "avg_pairwise_distance",
        "dispersion_score",
        "community_coverage_count",
        "community_coverage_fraction",
        "wl_coverage_count",
        "wl_coverage_fraction",
    ]
    for key in diff_keys:
        out[f"diff_{key}"] = float(pos[key] - neg[key])

    out["ratio_degree_sum_pos_over_neg"] = _safe_div(
        pos["degree_sum"], neg["degree_sum"], default=float("inf")
    )
    out["ratio_pagerank_sum_pos_over_neg"] = _safe_div(
        pos["pagerank_sum"], neg["pagerank_sum"], default=float("inf")
    )

    return out

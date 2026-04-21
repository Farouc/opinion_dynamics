"""Node placement strategies for zealot assignment.

The API is centered around `select_nodes_by_strategy`, making it easy to add
new strategies through a single dispatch point.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Iterable

import networkx as nx
import numpy as np

LOGGER = logging.getLogger(__name__)


def _node_key(node) -> tuple[int, str]:
    """Stable ordering key for mixed node-id types."""
    try:
        return (0, f"{int(node):012d}")
    except Exception:
        return (1, str(node))


def _validate_k(k: int, n_available: int) -> None:
    if k < 0:
        raise ValueError("k must be non-negative.")
    if k > n_available:
        raise ValueError(
            f"Cannot select k={k} nodes from only {n_available} available nodes."
        )


def _available_nodes(G: nx.Graph, exclude_nodes: Iterable[int] | None = None) -> list[int]:
    excluded = set(exclude_nodes or [])
    return [node for node in G.nodes() if node not in excluded]


def _sort_nodes_by_score(
    nodes: list[int],
    score_map: dict[int, float],
    descending: bool = True,
) -> list[int]:
    sign = -1.0 if descending else 1.0
    return sorted(
        nodes,
        key=lambda node: (
            sign * float(score_map.get(node, float("-inf" if descending else "inf"))),
            _node_key(node),
        ),
    )


def _select_random(rng: np.random.Generator, nodes: list[int], k: int) -> list[int]:
    if k == 0:
        return []
    idx = rng.choice(len(nodes), size=k, replace=False)
    return [nodes[int(i)] for i in idx]


def _community_labels(G: nx.Graph) -> dict[int, int]:
    cache_key = "_community_labels_greedy"
    cached = G.graph.get(cache_key)
    if cached is not None:
        return cached

    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    labels: dict[int, int] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            labels[int(node)] = int(cid)

    # Fallback in rare cases where some nodes may not appear.
    next_cid = len(communities)
    for node in G.nodes():
        if node not in labels:
            labels[int(node)] = int(next_cid)
            next_cid += 1

    G.graph[cache_key] = labels
    return labels


def _wl_labels(G: nx.Graph, n_iter: int = 3) -> dict[int, int]:
    cache_key = f"_wl_labels_{int(n_iter)}"
    cached = G.graph.get(cache_key)
    if cached is not None:
        return cached

    from src.wl_features import wl_color_refinement

    wl_out = wl_color_refinement(G, n_iter=n_iter)
    labels = {int(k): int(v) for k, v in wl_out["labels"].items()}
    G.graph[cache_key] = labels
    return labels


def _degree_dict(G: nx.Graph) -> dict[int, float]:
    cache_key = "_degree_dict"
    cached = G.graph.get(cache_key)
    if cached is not None:
        return cached

    degree = {int(node): float(val) for node, val in G.degree()}
    G.graph[cache_key] = degree
    return degree


def _pagerank_dict(G: nx.Graph) -> dict[int, float]:
    cache_key = "_pagerank_dict"
    cached = G.graph.get(cache_key)
    if cached is not None:
        return cached

    pr = nx.pagerank(G)
    out = {int(node): float(score) for node, score in pr.items()}
    G.graph[cache_key] = out
    return out


def _betweenness_dict(G: nx.Graph) -> dict[int, float]:
    cache_key = "_betweenness_dict"
    cached = G.graph.get(cache_key)
    if cached is not None:
        return cached

    b = nx.betweenness_centrality(G)
    out = {int(node): float(score) for node, score in b.items()}
    G.graph[cache_key] = out
    return out


def _eigenvector_dict(G: nx.Graph) -> dict[int, float]:
    cache_key = "_eigenvector_dict"
    cached = G.graph.get(cache_key)
    if cached is not None:
        return cached

    try:
        e = nx.eigenvector_centrality_numpy(G)
    except Exception:
        try:
            # Larger iteration budget for sparse/ill-conditioned cases.
            e = nx.eigenvector_centrality(
                G,
                max_iter=5000,
                tol=1e-8,
                nstart={node: 1.0 for node in G.nodes()},
            )
        except Exception:
            # Robust fallback for disconnected graphs or persistent convergence issues:
            # compute per connected component, then weight by component size so tiny
            # components do not dominate global ranking.
            n_total = max(1, G.number_of_nodes())
            e = {}
            for comp_nodes in nx.connected_components(G):
                comp = G.subgraph(comp_nodes).copy()
                comp_size = comp.number_of_nodes()
                comp_weight = float(comp_size / n_total)

                try:
                    local = nx.eigenvector_centrality_numpy(comp)
                except Exception:
                    try:
                        local = nx.eigenvector_centrality(
                            comp,
                            max_iter=5000,
                            tol=1e-8,
                            nstart={node: 1.0 for node in comp.nodes()},
                        )
                    except Exception:
                        # Last-resort stable fallback: degree-proportional score.
                        deg = {int(node): float(comp.degree(node)) for node in comp.nodes()}
                        deg_sum = float(sum(deg.values()))
                        if deg_sum <= 0.0:
                            local = {int(node): float(1.0 / comp_size) for node in comp.nodes()}
                        else:
                            local = {int(node): float(val / deg_sum) for node, val in deg.items()}

                for node, score in local.items():
                    e[int(node)] = float(score) * comp_weight

            LOGGER.warning(
                "Eigenvector centrality fallback path used (component-wise/degree fallback)."
            )

    out = {int(node): float(score) for node, score in e.items()}
    G.graph[cache_key] = out
    return out


def _farthest_spread(
    G: nx.Graph,
    candidates: list[int],
    k: int,
    start_node: int | None = None,
) -> list[int]:
    """Greedy max-min distance node selection over `candidates`."""
    if k == 0:
        return []

    candidate_set = set(candidates)
    degree = _degree_dict(G)

    if start_node is None or start_node not in candidate_set:
        start_node = _sort_nodes_by_score(candidates, degree, descending=True)[0]

    selected = [int(start_node)]
    remaining = [node for node in candidates if node != start_node]

    dist_first = nx.single_source_shortest_path_length(G, start_node)
    min_dist = {node: float(dist_first.get(node, math.inf)) for node in remaining}

    while len(selected) < k and remaining:
        next_node = max(
            remaining,
            key=lambda node: (min_dist.get(node, -math.inf), degree.get(node, 0.0), _node_key(node)),
        )
        selected.append(int(next_node))
        remaining.remove(next_node)

        dist_new = nx.single_source_shortest_path_length(G, next_node)
        for node in remaining:
            candidate_dist = float(dist_new.get(node, math.inf))
            prev = min_dist.get(node, math.inf)
            if candidate_dist < prev:
                min_dist[node] = candidate_dist

    return selected[:k]


def _community_cover(
    G: nx.Graph,
    candidates: list[int],
    k: int,
    rng: np.random.Generator,
) -> list[int]:
    if k == 0:
        return []

    labels = _community_labels(G)
    degree = _degree_dict(G)

    pools: dict[int, list[int]] = defaultdict(list)
    for node in candidates:
        pools[int(labels[int(node)])].append(int(node))

    for cid in pools:
        pools[cid] = _sort_nodes_by_score(pools[cid], degree, descending=True)

    community_ids = sorted(pools.keys())
    selected: list[int] = []
    while len(selected) < k and community_ids:
        new_ids: list[int] = []
        for cid in community_ids:
            if len(selected) >= k:
                break
            if pools[cid]:
                selected.append(int(pools[cid].pop(0)))
            if pools[cid]:
                new_ids.append(cid)
        community_ids = new_ids

    if len(selected) < k:
        remaining = [node for node in candidates if node not in set(selected)]
        if remaining:
            selected.extend(_select_random(rng, remaining, min(k - len(selected), len(remaining))))

    return selected[:k]


def _wl_cover(
    G: nx.Graph,
    candidates: list[int],
    k: int,
    rng: np.random.Generator,
    n_iter: int = 3,
) -> list[int]:
    if k == 0:
        return []

    labels = _wl_labels(G, n_iter=n_iter)
    degree = _degree_dict(G)

    pools: dict[int, list[int]] = defaultdict(list)
    for node in candidates:
        pools[int(labels[int(node)])].append(int(node))

    for cls in pools:
        pools[cls] = _sort_nodes_by_score(pools[cls], degree, descending=True)

    class_order = sorted(pools.keys(), key=lambda c: (-len(pools[c]), c))
    selected: list[int] = []
    while len(selected) < k and class_order:
        next_classes: list[int] = []
        for cls in class_order:
            if len(selected) >= k:
                break
            if pools[cls]:
                selected.append(int(pools[cls].pop(0)))
            if pools[cls]:
                next_classes.append(cls)
        class_order = next_classes

    if len(selected) < k:
        remaining = [node for node in candidates if node not in set(selected)]
        if remaining:
            selected.extend(_select_random(rng, remaining, min(k - len(selected), len(remaining))))

    return selected[:k]


def _wl_top_class(
    G: nx.Graph,
    candidates: list[int],
    k: int,
    rng: np.random.Generator,
    n_iter: int = 3,
) -> list[int]:
    if k == 0:
        return []

    labels = _wl_labels(G, n_iter=n_iter)
    degree = _degree_dict(G)

    pools: dict[int, list[int]] = defaultdict(list)
    for node in candidates:
        pools[int(labels[int(node)])].append(int(node))

    if not pools:
        return []

    class_scores: dict[int, float] = {}
    for cls, nodes in pools.items():
        avg_degree = float(np.mean([degree[n] for n in nodes])) if nodes else 0.0
        class_scores[cls] = float(len(nodes) * avg_degree)
        pools[cls] = _sort_nodes_by_score(nodes, degree, descending=True)

    class_order = sorted(pools.keys(), key=lambda c: (-class_scores[c], c))
    selected: list[int] = []
    for cls in class_order:
        for node in pools[cls]:
            if len(selected) >= k:
                break
            selected.append(int(node))
        if len(selected) >= k:
            break

    if len(selected) < k:
        remaining = [node for node in candidates if node not in set(selected)]
        if remaining:
            selected.extend(_select_random(rng, remaining, min(k - len(selected), len(remaining))))

    return selected[:k]


def _hub_then_spread(
    G: nx.Graph,
    candidates: list[int],
    k: int,
    hub_fraction: float = 0.5,
) -> list[int]:
    if k == 0:
        return []

    degree = _degree_dict(G)
    ranked = _sort_nodes_by_score(candidates, degree, descending=True)
    k_hub = int(round(hub_fraction * k))
    if k > 1:
        k_hub = max(1, min(k - 1, k_hub))
    else:
        k_hub = 1

    hubs = ranked[:k_hub]
    remaining = [node for node in candidates if node not in set(hubs)]
    spread = _farthest_spread(G, remaining, k=min(k - len(hubs), len(remaining)))
    selected = hubs + spread

    if len(selected) < k:
        leftovers = [node for node in candidates if node not in set(selected)]
        selected.extend(leftovers[: (k - len(selected))])

    return selected[:k]


def available_placement_strategies() -> list[str]:
    return [
        "random",
        "highest_degree",
        "lowest_degree",
        "highest_betweenness",
        "highest_eigenvector",
        "highest_pagerank",
        "community_cover",
        "wl_cover",
        "wl_top_class",
        "farthest_spread",
        "hub_then_spread",
        "explicit",
    ]


def select_nodes_by_strategy(
    G: nx.Graph,
    k: int,
    strategy: str,
    seed: int | None = None,
    exclude_nodes: Iterable[int] | None = None,
    **kwargs,
) -> list[int]:
    """Select `k` nodes from `G` using the requested placement strategy.

    Parameters
    ----------
    G : nx.Graph
    k : int
        Number of nodes to select.
    strategy : str
        Placement strategy name.
    seed : int | None
        RNG seed used by stochastic strategies.
    exclude_nodes : Iterable[int] | None
        Nodes excluded from candidate pool.

    Returns
    -------
    list[int]
        Selected node ids.
    """
    strategy_key = str(strategy).strip().lower()
    candidates = _available_nodes(G, exclude_nodes=exclude_nodes)
    _validate_k(int(k), len(candidates))

    rng = np.random.default_rng(seed)

    if k == 0:
        return []

    if strategy_key == "explicit":
        explicit_nodes = list(kwargs.get("nodes", []))
        if len(explicit_nodes) != int(k):
            raise ValueError(
                "explicit strategy requires `nodes` with exactly k entries. "
                f"Received len(nodes)={len(explicit_nodes)} and k={k}."
            )
        explicit_set = set(explicit_nodes)
        if len(explicit_set) != len(explicit_nodes):
            raise ValueError("explicit node list contains duplicates.")
        if not explicit_set.issubset(set(candidates)):
            raise ValueError("explicit node list contains excluded or invalid nodes.")
        return [int(node) for node in explicit_nodes]

    if strategy_key == "random":
        return [int(node) for node in _select_random(rng, candidates, k)]

    if strategy_key == "highest_degree":
        degree = _degree_dict(G)
        return [int(node) for node in _sort_nodes_by_score(candidates, degree, descending=True)[:k]]

    if strategy_key == "lowest_degree":
        degree = _degree_dict(G)
        return [int(node) for node in _sort_nodes_by_score(candidates, degree, descending=False)[:k]]

    if strategy_key == "highest_betweenness":
        b = _betweenness_dict(G)
        return [int(node) for node in _sort_nodes_by_score(candidates, b, descending=True)[:k]]

    if strategy_key == "highest_eigenvector":
        e = _eigenvector_dict(G)
        return [int(node) for node in _sort_nodes_by_score(candidates, e, descending=True)[:k]]

    if strategy_key == "highest_pagerank":
        pr = _pagerank_dict(G)
        return [int(node) for node in _sort_nodes_by_score(candidates, pr, descending=True)[:k]]

    if strategy_key == "community_cover":
        return _community_cover(G, candidates=candidates, k=int(k), rng=rng)

    if strategy_key == "wl_cover":
        return _wl_cover(
            G,
            candidates=candidates,
            k=int(k),
            rng=rng,
            n_iter=int(kwargs.get("n_iter", 3)),
        )

    if strategy_key == "wl_top_class":
        return _wl_top_class(
            G,
            candidates=candidates,
            k=int(k),
            rng=rng,
            n_iter=int(kwargs.get("n_iter", 3)),
        )

    if strategy_key == "farthest_spread":
        start_node = kwargs.get("start_node")
        if kwargs.get("random_start", False):
            start_node = int(_select_random(rng, candidates, 1)[0])
        return _farthest_spread(G, candidates=candidates, k=int(k), start_node=start_node)

    if strategy_key == "hub_then_spread":
        return _hub_then_spread(
            G,
            candidates=candidates,
            k=int(k),
            hub_fraction=float(kwargs.get("hub_fraction", 0.5)),
        )

    known = ", ".join(available_placement_strategies())
    raise ValueError(f"Unknown strategy '{strategy}'. Available: {known}")

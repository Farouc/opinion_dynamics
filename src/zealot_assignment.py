"""Assignment helpers for two competing zealot camps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np

from src.placement_strategies import select_nodes_by_strategy


@dataclass(frozen=True)
class TwoCampAssignment:
    """Container describing a two-camp zealot assignment."""

    pos_nodes: list[int]
    neg_nodes: list[int]
    pos_zealot_mask: np.ndarray
    neg_zealot_mask: np.ndarray
    states: np.ndarray


def _ensure_integer_graph(G: nx.Graph) -> None:
    n = G.number_of_nodes()
    if set(G.nodes()) != set(range(n)):
        raise ValueError(
            "Graph nodes must be labeled 0..N-1 for array-based simulation. "
            "Convert with nx.convert_node_labels_to_integers(...)."
        )


def _normalize_node_ids(nodes: Iterable[int], n: int) -> list[int]:
    out = [int(node) for node in nodes]
    if len(set(out)) != len(out):
        raise ValueError("Zealot node list contains duplicates.")
    for node in out:
        if node < 0 or node >= n:
            raise ValueError(f"Invalid node id {node}; expected range [0, {n - 1}].")
    return out


def _resolve_explicit_nodes(
    direct_nodes: Iterable[int] | None,
    strategy: str,
    strategy_kwargs: dict | None,
) -> list[int] | None:
    if direct_nodes is not None:
        return list(direct_nodes)

    strategy_key = str(strategy).strip().lower()
    if strategy_key == "explicit":
        if strategy_kwargs is None or "nodes" not in strategy_kwargs:
            raise ValueError("explicit strategy requires strategy_kwargs={'nodes': [...]}.")
        return list(strategy_kwargs["nodes"])

    return None


def assign_two_zealot_sets(
    G: nx.Graph,
    n_pos: int,
    n_neg: int,
    strategy_pos: str = "random",
    strategy_neg: str = "random",
    seed: int | None = None,
    strategy_kwargs_pos: dict | None = None,
    strategy_kwargs_neg: dict | None = None,
    pos_nodes: Iterable[int] | None = None,
    neg_nodes: Iterable[int] | None = None,
    free_init: str = "random",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign two disjoint zealot camps and initialize node states.

    Parameters
    ----------
    G : nx.Graph
    n_pos, n_neg : int
        Number of positive and negative zealots.
    strategy_pos, strategy_neg : str
        Placement strategy names for each camp.
    seed : int | None
        Seed for reproducible placement and initialization.
    strategy_kwargs_pos, strategy_kwargs_neg : dict | None
        Optional strategy-specific keyword dictionaries.
    pos_nodes, neg_nodes : iterable[int] | None
        Optional explicit zealot sets. If provided, they override strategies.
    free_init : {'random', 'all_plus', 'all_minus'}
        Initialization for non-zealot nodes.

    Returns
    -------
    pos_zealot_mask : np.ndarray, shape (N,)
    neg_zealot_mask : np.ndarray, shape (N,)
    states : np.ndarray, shape (N,)
        Initial states in {-1,+1} with zealot constraints enforced.
    """
    _ensure_integer_graph(G)

    if n_pos < 0 or n_neg < 0:
        raise ValueError("n_pos and n_neg must be non-negative.")

    n = G.number_of_nodes()
    if n_pos + n_neg > n:
        raise ValueError("n_pos + n_neg cannot exceed number of nodes.")

    strategy_kwargs_pos = dict(strategy_kwargs_pos or {})
    strategy_kwargs_neg = dict(strategy_kwargs_neg or {})

    rng = np.random.default_rng(seed)

    explicit_pos = _resolve_explicit_nodes(
        direct_nodes=pos_nodes,
        strategy=strategy_pos,
        strategy_kwargs=strategy_kwargs_pos,
    )
    explicit_neg = _resolve_explicit_nodes(
        direct_nodes=neg_nodes,
        strategy=strategy_neg,
        strategy_kwargs=strategy_kwargs_neg,
    )

    if explicit_pos is not None:
        pos_list = _normalize_node_ids(explicit_pos, n=n)
        if len(pos_list) != int(n_pos):
            raise ValueError(
                f"len(pos_nodes)={len(pos_list)} does not match n_pos={n_pos}."
            )
    else:
        pos_seed = int(rng.integers(np.iinfo(np.int32).max))
        pos_list = select_nodes_by_strategy(
            G,
            k=int(n_pos),
            strategy=strategy_pos,
            seed=pos_seed,
            **strategy_kwargs_pos,
        )

    pos_set = set(pos_list)

    if explicit_neg is not None:
        neg_list = _normalize_node_ids(explicit_neg, n=n)
        if len(neg_list) != int(n_neg):
            raise ValueError(
                f"len(neg_nodes)={len(neg_list)} does not match n_neg={n_neg}."
            )
    else:
        neg_seed = int(rng.integers(np.iinfo(np.int32).max))
        neg_list = select_nodes_by_strategy(
            G,
            k=int(n_neg),
            strategy=strategy_neg,
            seed=neg_seed,
            exclude_nodes=pos_set,
            **strategy_kwargs_neg,
        )

    neg_set = set(neg_list)
    overlap = pos_set.intersection(neg_set)
    if overlap:
        raise ValueError(f"Positive and negative zealot sets overlap: {sorted(overlap)}")

    pos_mask = np.zeros(n, dtype=bool)
    neg_mask = np.zeros(n, dtype=bool)
    if pos_list:
        pos_mask[np.asarray(pos_list, dtype=int)] = True
    if neg_list:
        neg_mask[np.asarray(neg_list, dtype=int)] = True

    free_init_key = str(free_init).strip().lower()
    if free_init_key == "random":
        states = rng.choice(np.array([-1, +1], dtype=np.int8), size=n)
    elif free_init_key == "all_plus":
        states = np.ones(n, dtype=np.int8)
    elif free_init_key == "all_minus":
        states = -np.ones(n, dtype=np.int8)
    else:
        raise ValueError("free_init must be one of {'random', 'all_plus', 'all_minus'}.")

    states[pos_mask] = np.int8(+1)
    states[neg_mask] = np.int8(-1)

    return pos_mask, neg_mask, states


def build_two_camp_assignment(
    G: nx.Graph,
    n_pos: int,
    n_neg: int,
    strategy_pos: str = "random",
    strategy_neg: str = "random",
    seed: int | None = None,
    strategy_kwargs_pos: dict | None = None,
    strategy_kwargs_neg: dict | None = None,
    pos_nodes: Iterable[int] | None = None,
    neg_nodes: Iterable[int] | None = None,
    free_init: str = "random",
) -> TwoCampAssignment:
    """Create a dataclass bundle for two-camp assignment artifacts."""
    pos_mask, neg_mask, states = assign_two_zealot_sets(
        G=G,
        n_pos=n_pos,
        n_neg=n_neg,
        strategy_pos=strategy_pos,
        strategy_neg=strategy_neg,
        seed=seed,
        strategy_kwargs_pos=strategy_kwargs_pos,
        strategy_kwargs_neg=strategy_kwargs_neg,
        pos_nodes=pos_nodes,
        neg_nodes=neg_nodes,
        free_init=free_init,
    )

    pos_nodes_out = np.where(pos_mask)[0].astype(int).tolist()
    neg_nodes_out = np.where(neg_mask)[0].astype(int).tolist()

    return TwoCampAssignment(
        pos_nodes=pos_nodes_out,
        neg_nodes=neg_nodes_out,
        pos_zealot_mask=pos_mask,
        neg_zealot_mask=neg_mask,
        states=states,
    )

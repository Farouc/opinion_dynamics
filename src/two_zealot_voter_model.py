"""Two-camp zealot voter dynamics.

States are binary {-1,+1}; positive zealots are fixed to +1 and negative zealots
are fixed to -1.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from tqdm import tqdm

from src.observables import magnetization
from src.utils import get_neighbor_lookup
from src.zealot_assignment import assign_two_zealot_sets

LOGGER = logging.getLogger(__name__)


def _ensure_integer_graph(G: nx.Graph) -> nx.Graph:
    if set(G.nodes()) == set(range(G.number_of_nodes())):
        return G
    LOGGER.warning("Converting graph labels to consecutive integers for array indexing.")
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def step_two_zealot_voter(
    G: nx.Graph,
    states: np.ndarray,
    pos_zealot_mask: np.ndarray,
    neg_zealot_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """One asynchronous update for the two-camp zealot voter model."""
    n = states.shape[0]
    i = int(rng.integers(n))

    if bool(pos_zealot_mask[i] or neg_zealot_mask[i]):
        return

    neighbor_lookup = get_neighbor_lookup(G)
    neighbors_i = neighbor_lookup[i]
    if neighbors_i.size == 0:
        return

    j = int(neighbors_i[rng.integers(neighbors_i.size)])
    states[i] = states[j]


def step_two_zealot_voter_with_delta(
    G: nx.Graph,
    states: np.ndarray,
    pos_zealot_mask: np.ndarray,
    neg_zealot_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, int | None]:
    """One update with O(1) observable increments.

    Returns
    -------
    delta_m : float
        Magnetization increment.
    delta_pos_frac : float
        Increment for fraction of +1 nodes.
    changed_node : int | None
        Updated node id if state changed, else None.
    """
    n = states.shape[0]
    i = int(rng.integers(n))

    if bool(pos_zealot_mask[i] or neg_zealot_mask[i]):
        return 0.0, 0.0, None

    neighbor_lookup = get_neighbor_lookup(G)
    neighbors_i = neighbor_lookup[i]
    if neighbors_i.size == 0:
        return 0.0, 0.0, None

    old_val = int(states[i])
    j = int(neighbors_i[rng.integers(neighbors_i.size)])
    new_val = int(states[j])

    if new_val == old_val:
        return 0.0, 0.0, None

    states[i] = np.int8(new_val)
    delta_m = float((new_val - old_val) / n)
    delta_pos = float(((1 if new_val == 1 else 0) - (1 if old_val == 1 else 0)) / n)
    return delta_m, delta_pos, i


def run_two_zealot_simulation(
    G: nx.Graph,
    n_pos: int,
    n_neg: int,
    T: int,
    burn_in: int,
    seed: int | None = None,
    strategy_pos: str = "random",
    strategy_neg: str = "random",
    strategy_kwargs_pos: dict | None = None,
    strategy_kwargs_neg: dict | None = None,
    pos_nodes: list[int] | None = None,
    neg_nodes: list[int] | None = None,
    free_init: str = "random",
    record: bool = False,
    show_progress: bool = True,
    print_magnetization: bool = False,
    magnetization_interval: int = 1000,
    compute_flip_activity: bool = False,
) -> dict:
    """Run two-camp zealot voter simulation.

    Returns a dictionary containing full trajectory observables and assignment
    metadata for downstream tipping/feature analysis.
    """
    if T <= 0:
        raise ValueError("T must be strictly positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if burn_in >= T:
        raise ValueError("burn_in must be strictly smaller than T.")
    if magnetization_interval <= 0:
        raise ValueError("magnetization_interval must be strictly positive.")

    G_int = _ensure_integer_graph(G)
    n = G_int.number_of_nodes()

    rng = np.random.default_rng(seed)
    assign_seed = int(rng.integers(np.iinfo(np.int32).max))
    pos_mask, neg_mask, states = assign_two_zealot_sets(
        G=G_int,
        n_pos=int(n_pos),
        n_neg=int(n_neg),
        strategy_pos=strategy_pos,
        strategy_neg=strategy_neg,
        seed=assign_seed,
        strategy_kwargs_pos=strategy_kwargs_pos,
        strategy_kwargs_neg=strategy_kwargs_neg,
        pos_nodes=pos_nodes,
        neg_nodes=neg_nodes,
        free_init=free_init,
    )

    zealot_mask = np.logical_or(pos_mask, neg_mask)
    free_mask = np.logical_not(zealot_mask)

    initial_states = states.copy()
    m_series = np.empty(T, dtype=float)
    pos_series = np.empty(T, dtype=float)
    neg_series = np.empty(T, dtype=float)
    trajectory = np.empty((T, n), dtype=np.int8) if record else None

    flip_counts = np.zeros(n, dtype=np.int32) if compute_flip_activity else None

    m_t = float(magnetization(states))
    pos_t = float(np.mean(states == 1))

    iterator = tqdm(
        range(T),
        desc=f"two-camp sim n+={n_pos} n-={n_neg}",
        leave=False,
        disable=not show_progress,
    )

    cumulative_m = 0.0
    for t in iterator:
        delta_m, delta_pos, changed_node = step_two_zealot_voter_with_delta(
            G_int,
            states,
            pos_zealot_mask=pos_mask,
            neg_zealot_mask=neg_mask,
            rng=rng,
        )
        m_t += delta_m
        pos_t += delta_pos
        neg_t = float(1.0 - pos_t)

        m_series[t] = m_t
        pos_series[t] = pos_t
        neg_series[t] = neg_t

        cumulative_m += m_t
        if print_magnetization and (
            t == 0 or (t + 1) % magnetization_interval == 0 or t == T - 1
        ):
            running_mean = cumulative_m / float(t + 1)
            tqdm.write(
                f"[magnetization] step={t + 1}/{T} "
                f"m(t)={m_t:.6f} running_mean={running_mean:.6f}"
            )

        if compute_flip_activity and changed_node is not None and free_mask[changed_node]:
            flip_counts[changed_node] += 1

        if record:
            trajectory[t] = states

    flip_activity = None
    if compute_flip_activity and flip_counts is not None:
        free_flip_counts = flip_counts[free_mask]
        flip_activity = {
            "total_free_flips": int(np.sum(free_flip_counts)),
            "mean_flips_per_free_node": float(np.mean(free_flip_counts))
            if free_flip_counts.size > 0
            else 0.0,
            "active_free_fraction": float(np.mean(free_flip_counts > 0))
            if free_flip_counts.size > 0
            else 0.0,
        }

    return {
        "magnetization": m_series,
        "positive_fraction": pos_series,
        "negative_fraction": neg_series,
        "trajectory": trajectory,
        "pos_zealot_mask": pos_mask,
        "neg_zealot_mask": neg_mask,
        "zealot_mask": zealot_mask,
        "free_mask": free_mask,
        "pos_nodes": np.where(pos_mask)[0].astype(int).tolist(),
        "neg_nodes": np.where(neg_mask)[0].astype(int).tolist(),
        "initial_states": initial_states,
        "final_states": states.copy(),
        "flip_activity": flip_activity,
        "n_pos": int(np.sum(pos_mask)),
        "n_neg": int(np.sum(neg_mask)),
        "rho_pos": float(np.mean(pos_mask)),
        "rho_neg": float(np.mean(neg_mask)),
        "T": int(T),
        "burn_in": int(burn_in),
        "seed": seed,
        "assignment_seed": int(assign_seed),
        "strategy_pos": str(strategy_pos),
        "strategy_neg": str(strategy_neg),
    }

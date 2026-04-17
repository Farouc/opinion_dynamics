"""Single-step zealot voter dynamics."""

from __future__ import annotations

import numpy as np

from src.utils import get_neighbor_lookup


def step_voter(G, states: np.ndarray, zealot_mask: np.ndarray, rng: np.random.Generator) -> None:
    """Perform one asynchronous voter-model update.

    Rules:
    1. pick random node i
    2. if i is zealot, skip
    3. pick random neighbor j
    4. set x_i = x_j
    """
    n = states.shape[0]
    i = int(rng.integers(n))

    if zealot_mask[i]:
        return

    neighbor_lookup = get_neighbor_lookup(G)
    neighbors_i = neighbor_lookup[i]
    if neighbors_i.size == 0:
        return

    j = int(neighbors_i[rng.integers(neighbors_i.size)])
    states[i] = states[j]


def step_voter_with_delta(
    G,
    states: np.ndarray,
    zealot_mask: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Perform one update and return the magnetization increment Δm.

    Returns
    -------
    float
        Change in magnetization caused by this step. Zero means no state change.
    """
    n = states.shape[0]
    i = int(rng.integers(n))

    if zealot_mask[i]:
        return 0.0

    neighbor_lookup = get_neighbor_lookup(G)
    neighbors_i = neighbor_lookup[i]
    if neighbors_i.size == 0:
        return 0.0

    old_val = int(states[i])
    j = int(neighbors_i[rng.integers(neighbors_i.size)])
    new_val = int(states[j])
    if new_val == old_val:
        return 0.0

    states[i] = new_val
    return float((new_val - old_val) / n)

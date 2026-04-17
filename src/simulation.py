"""Simulation engine for the zealot voter model."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from tqdm import tqdm

from src.observables import magnetization
from src.utils import assign_zealots
from src.voter_model import step_voter

LOGGER = logging.getLogger(__name__)


def _ensure_integer_graph(G: nx.Graph) -> nx.Graph:
    if set(G.nodes()) == set(range(G.number_of_nodes())):
        return G
    LOGGER.warning("Converting graph labels to consecutive integers for array indexing.")
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def run_simulation(
    G,
    rho: float,
    T: int,
    burn_in: int,
    seed: int | None = None,
    record: bool = True,
    zealot_state: int = +1,
    show_progress: bool = True,
    print_magnetization: bool = False,
    magnetization_interval: int = 1000,
):
    """Run zealot voter dynamics and record magnetization time series.

    Parameters
    ----------
    G : networkx.Graph
    rho : float
        Fraction of zealots.
    T : int
        Number of asynchronous update steps.
    burn_in : int
        Burn-in used downstream for observables.
    seed : int | None
        Seed controlling zealot assignment and dynamics.
    record : bool
        If True, return full state trajectory.

    Returns
    -------
    dict with keys:
    - magnetization: np.ndarray shape (T,)
    - trajectory: np.ndarray shape (T, N) or None
    - zealot_mask: np.ndarray shape (N,)
    - initial_states: np.ndarray shape (N,)
    - final_states: np.ndarray shape (N,)
    - rho, T, burn_in, seed
    """
    if T <= 0:
        raise ValueError("T must be strictly positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if magnetization_interval <= 0:
        raise ValueError("magnetization_interval must be strictly positive.")

    G_int = _ensure_integer_graph(G)
    n = G_int.number_of_nodes()

    rng = np.random.default_rng(seed)
    zealot_seed = int(rng.integers(np.iinfo(np.int32).max))
    zealot_mask, states = assign_zealots(
        G_int,
        rho=rho,
        state=zealot_state,
        seed=zealot_seed,
    )

    initial_states = states.copy()
    m_series = np.empty(T, dtype=float)
    trajectory = np.empty((T, n), dtype=np.int8) if record else None

    iterator = tqdm(
        range(T),
        desc=f"simulate rho={rho:.3f}",
        leave=False,
        disable=not show_progress,
    )

    cumulative_m = 0.0
    for t in iterator:
        step_voter(G_int, states, zealot_mask, rng)
        m_t = magnetization(states)
        m_series[t] = m_t
        cumulative_m += m_t

        if print_magnetization and (
            t == 0
            or (t + 1) % magnetization_interval == 0
            or t == T - 1
        ):
            m_running = cumulative_m / float(t + 1)
            tqdm.write(
                f"[magnetization] step={t + 1}/{T} "
                f"m(t)={m_t:.6f} running_mean={m_running:.6f}"
            )

        if record:
            trajectory[t] = states

    return {
        "magnetization": m_series,
        "trajectory": trajectory,
        "zealot_mask": zealot_mask,
        "initial_states": initial_states,
        "final_states": states.copy(),
        "rho": float(rho),
        "T": int(T),
        "burn_in": int(burn_in),
        "seed": seed,
    }

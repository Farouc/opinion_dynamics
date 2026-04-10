"""Shared utilities: configuration, logging, I/O, zealot assignment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure project-wide logging format once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")
    return cfg


def parse_rho_values(raw: Any) -> np.ndarray:
    """Parse rho values from config into a float array.

    Supported formats:
    - explicit list/tuple: [0.0, 0.05, ...]
    - dict: {start: 0.0, stop: 0.5, step: 0.05}
    """
    if isinstance(raw, dict):
        start = float(raw["start"])
        stop = float(raw["stop"])
        step = float(raw["step"])
        if step <= 0.0:
            raise ValueError("rho step must be positive.")
        count = int(round((stop - start) / step)) + 1
        vals = start + step * np.arange(count, dtype=float)
        return np.clip(vals, 0.0, 1.0)

    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.asarray(raw, dtype=float)
        return arr

    raise ValueError(
        "rho_values must be a sequence or a mapping with {start, stop, step}."
    )


def assign_zealots(
    G,
    rho: float,
    state: int = +1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign zealots and initialize states in {-1, +1}.

    Parameters
    ----------
    G : networkx.Graph
        Graph with integer labels [0, ..., n-1].
    rho : float
        Fraction of zealot nodes.
    state : int
        Fixed zealot state (+1 or -1).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    zealot_mask : np.ndarray
        Boolean mask of shape (n,).
    states : np.ndarray
        Initial states in {-1, +1}, with zealots fixed to `state`.
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be in [0, 1].")

    n = G.number_of_nodes()
    rng = np.random.default_rng(seed)

    n_zealots = int(round(rho * n))
    zealot_mask = np.zeros(n, dtype=bool)
    if n_zealots > 0:
        zealot_idx = rng.choice(n, size=n_zealots, replace=False)
        zealot_mask[zealot_idx] = True

    states = rng.choice(np.array([-1, +1], dtype=np.int8), size=n)
    zealot_state = 1 if state >= 0 else -1
    states[zealot_mask] = zealot_state

    return zealot_mask, states


def get_neighbor_lookup(G) -> list[np.ndarray]:
    """Return cached neighbor lookup list indexed by node id."""
    cache_key = "_neighbor_lookup"
    lookup = G.graph.get(cache_key)

    if lookup is not None and len(lookup) == G.number_of_nodes():
        return lookup

    n = G.number_of_nodes()
    lookup = [
        np.fromiter(G.neighbors(i), dtype=np.int32, count=G.degree(i))
        for i in range(n)
    ]
    G.graph[cache_key] = lookup
    return lookup


def save_dict_csv(path: str | Path, data: dict[str, Any], keys: list[str] | None = None) -> None:
    """Save equal-length 1D arrays from a dict into CSV columns."""
    keys = keys or list(data.keys())
    columns = [np.asarray(data[k]) for k in keys]

    lengths = {col.shape[0] for col in columns}
    if len(lengths) != 1:
        raise ValueError("All CSV columns must have the same length.")

    matrix = np.column_stack(columns)
    header = ",".join(keys)
    np.savetxt(path, matrix, delimiter=",", header=header, comments="")

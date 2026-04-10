"""Extreme Value Theory utilities for magnetization time series."""

from __future__ import annotations

import logging
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path("/tmp/matplotlib")
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genextreme
from tqdm import tqdm

from src.simulation import run_simulation
from src.utils import ensure_dir

LOGGER = logging.getLogger(__name__)


def compute_block_maxima(series, block_size: int) -> np.ndarray:
    """Compute block maxima for a 1D time series."""
    arr = np.asarray(series, dtype=float)
    if block_size <= 0:
        raise ValueError("block_size must be strictly positive.")

    n_blocks = arr.size // block_size
    if n_blocks <= 0:
        raise ValueError("Series too short for requested block_size.")

    trimmed = arr[: n_blocks * block_size]
    blocks = trimmed.reshape(n_blocks, block_size)
    return blocks.max(axis=1)


def fit_gev(maxima) -> dict[str, float]:
    """Fit a Generalized Extreme Value distribution to maxima."""
    maxima = np.asarray(maxima, dtype=float)
    if maxima.size < 3:
        raise ValueError("Need at least 3 maxima values for stable GEV fitting.")

    shape, loc, scale = genextreme.fit(maxima)
    return {
        "shape": float(shape),
        "location": float(loc),
        "scale": float(scale),
    }


def evt_analysis(
    G,
    rho_values,
    T: int,
    block_size: int,
    burn_in: int = 0,
    seed: int | None = None,
    show_progress: bool = True,
):
    """Run EVT analysis across rho values and return fitted GEV parameters."""
    rho_values = np.asarray(rho_values, dtype=float)

    xi = np.empty(rho_values.size, dtype=float)
    mu = np.empty(rho_values.size, dtype=float)
    sigma = np.empty(rho_values.size, dtype=float)
    n_blocks_used = np.empty(rho_values.size, dtype=int)

    rng = np.random.default_rng(seed)

    iterator = tqdm(
        enumerate(rho_values),
        total=rho_values.size,
        desc="evt sweep",
        disable=not show_progress,
    )

    for i, rho in iterator:
        run_seed = int(rng.integers(np.iinfo(np.int32).max))
        sim = run_simulation(
            G=G,
            rho=float(rho),
            T=T,
            burn_in=burn_in,
            seed=run_seed,
            record=False,
            show_progress=False,
        )
        series = sim["magnetization"]
        if burn_in > 0:
            if burn_in >= series.size:
                raise ValueError("burn_in must be smaller than T for EVT analysis.")
            series = series[burn_in:]

        maxima = compute_block_maxima(series, block_size=block_size)
        params = fit_gev(maxima)

        xi[i] = params["shape"]
        mu[i] = params["location"]
        sigma[i] = params["scale"]
        n_blocks_used[i] = maxima.size

    return {
        "rho_values": rho_values,
        "xi": xi,
        "mu": mu,
        "sigma": sigma,
        "n_blocks": n_blocks_used,
        "T": int(T),
        "burn_in": int(burn_in),
        "block_size": int(block_size),
    }


def plot_evt_results(results: dict, output_dir: str | Path):
    """Plot and save xi(rho)."""
    output_dir = ensure_dir(output_dir)

    rho = np.asarray(results["rho_values"], dtype=float)
    xi = np.asarray(results["xi"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(rho, xi, marker="o", color="tab:red")
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1.0)
    ax.set_xlabel(r"$\rho$ (zealot fraction)")
    ax.set_ylabel(r"GEV shape $\xi$")
    ax.set_title(r"Extreme-Value Shape Parameter $\xi(\rho)$")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / "xi_vs_rho.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    LOGGER.info("Saved EVT plot to %s", out_path)
    return out_path

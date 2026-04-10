"""Phase-transition analysis for the zealot voter model."""

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
from tqdm import tqdm

from src.observables import time_average, variance
from src.simulation import run_simulation
from src.utils import ensure_dir

LOGGER = logging.getLogger(__name__)


def sweep_rho(
    G,
    rho_values,
    T: int,
    burn_in: int,
    n_runs: int,
    seed: int | None = None,
    show_progress: bool = True,
):
    """Run repeated simulations over rho values and aggregate observables."""
    if n_runs <= 0:
        raise ValueError("n_runs must be strictly positive.")

    rho_values = np.asarray(rho_values, dtype=float)
    n_rho = rho_values.size

    mean_m_runs = np.empty((n_rho, n_runs), dtype=float)
    var_m_runs = np.empty((n_rho, n_runs), dtype=float)

    rng = np.random.default_rng(seed)

    outer = tqdm(
        enumerate(rho_values),
        total=n_rho,
        desc="rho sweep",
        disable=not show_progress,
    )

    for i, rho in outer:
        inner = tqdm(
            range(n_runs),
            desc=f"runs rho={rho:.3f}",
            leave=False,
            disable=not show_progress,
        )
        for run_idx in inner:
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
            mean_m_runs[i, run_idx] = time_average(series, burn_in)
            var_m_runs[i, run_idx] = variance(series, burn_in)

    results = {
        "rho_values": rho_values,
        "mean_m": mean_m_runs.mean(axis=1),
        "var_m": var_m_runs.mean(axis=1),
        "mean_m_std": mean_m_runs.std(axis=1, ddof=1 if n_runs > 1 else 0),
        "var_m_std": var_m_runs.std(axis=1, ddof=1 if n_runs > 1 else 0),
        "mean_m_sem": mean_m_runs.std(axis=1, ddof=1 if n_runs > 1 else 0) / np.sqrt(n_runs),
        "var_m_sem": var_m_runs.std(axis=1, ddof=1 if n_runs > 1 else 0) / np.sqrt(n_runs),
        "mean_m_runs": mean_m_runs,
        "var_m_runs": var_m_runs,
        "T": int(T),
        "burn_in": int(burn_in),
        "n_runs": int(n_runs),
    }
    return results


def estimate_critical_rho(results: dict) -> dict[str, float]:
    """Estimate critical rho using variance and magnetization slope criteria."""
    rho = np.asarray(results["rho_values"], dtype=float)
    mean_m = np.asarray(results["mean_m"], dtype=float)
    var_m = np.asarray(results["var_m"], dtype=float)

    idx_var_peak = int(np.argmax(var_m))
    rho_var_peak = float(rho[idx_var_peak])

    if rho.size > 1:
        dm_drho = np.gradient(mean_m, rho)
        idx_slope_peak = int(np.argmax(np.abs(dm_drho)))
        rho_slope_peak = float(rho[idx_slope_peak])
    else:
        dm_drho = np.array([0.0])
        idx_slope_peak = 0
        rho_slope_peak = rho_var_peak

    rho_c_estimate = float(0.5 * (rho_var_peak + rho_slope_peak))

    return {
        "rho_var_peak": rho_var_peak,
        "rho_slope_peak": rho_slope_peak,
        "rho_c_estimate": rho_c_estimate,
        "var_peak_index": float(idx_var_peak),
        "slope_peak_index": float(idx_slope_peak),
        "max_var": float(var_m[idx_var_peak]),
        "max_abs_dm_drho": float(np.max(np.abs(dm_drho))),
    }


def plot_phase_transition(results: dict, output_dir: str | Path):
    """Plot and save <m> vs rho and Var(m) vs rho."""
    output_dir = ensure_dir(output_dir)

    rho = np.asarray(results["rho_values"], dtype=float)
    mean_m = np.asarray(results["mean_m"], dtype=float)
    var_m = np.asarray(results["var_m"], dtype=float)
    mean_sem = np.asarray(results.get("mean_m_sem", np.zeros_like(mean_m)), dtype=float)
    var_sem = np.asarray(results.get("var_m_sem", np.zeros_like(var_m)), dtype=float)

    fig1, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.errorbar(rho, mean_m, yerr=mean_sem, marker="o", capsize=3)
    ax1.set_xlabel(r"$\rho$ (zealot fraction)")
    ax1.set_ylabel(r"$\langle m \rangle$")
    ax1.set_title("Mean Magnetization vs Zealot Density")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    m_path = output_dir / "mean_m_vs_rho.png"
    fig1.savefig(m_path, dpi=180)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
    ax2.errorbar(rho, var_m, yerr=var_sem, marker="o", capsize=3, color="tab:orange")
    ax2.set_xlabel(r"$\rho$ (zealot fraction)")
    ax2.set_ylabel(r"$\mathrm{Var}(m)$")
    ax2.set_title("Magnetization Variance vs Zealot Density")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    v_path = output_dir / "var_m_vs_rho.png"
    fig2.savefig(v_path, dpi=180)
    plt.close(fig2)

    LOGGER.info("Saved phase-transition plots to %s", output_dir)
    return {"mean_plot": m_path, "var_plot": v_path}

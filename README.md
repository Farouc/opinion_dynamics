# Criticality and Rare Events in the Zealot Voter Model

Research-grade, modular simulation code for:

1. Zealot voter dynamics on configurable graphs
2. Phase-transition detection versus zealot density `rho`
3. Extreme Value Theory (EVT) analysis of magnetization fluctuations

## Project Structure

```text
zealot_voter_criticality/
├── config/
│   └── default.yaml
├── src/
│   ├── graph_generation.py
│   ├── voter_model.py
│   ├── simulation.py
│   ├── observables.py
│   ├── phase_transition.py
│   ├── evt.py
│   └── utils.py
├── experiments/
│   ├── run_simulation.py
│   ├── sweep_rho.py
│   └── run_evt.py
├── results/
│   ├── raw/
│   └── figures/
├── notebooks/
│   └── exploratory.ipynb
├── requirements.txt
└── README.md
```

## Installation

```bash
cd zealot_voter_criticality
pip install -r requirements.txt
```

## Quick Start

Run one trajectory and plot `m(t)`:

```bash
python experiments/run_simulation.py --config config/default.yaml
```

Run phase-transition sweep:

```bash
python experiments/sweep_rho.py --config config/default.yaml
```

Run EVT pipeline:

```bash
python experiments/run_evt.py --config config/default.yaml
```

Write outputs to a custom folder (recommended for multi-run campaigns):

```bash
python experiments/sweep_rho.py --config config/default.yaml --results-dir results/campaign_2026-04-10/erdos
```

## Key Outputs

Saved under `results/`:

- `results/raw/simulation_rho_*.npz`
- `results/raw/phase_transition_results.npz`
- `results/raw/phase_transition_results.csv`
- `results/raw/critical_rho_estimate.json`
- `results/raw/evt_results.npz`
- `results/raw/evt_results.csv`
- `results/figures/mean_m_vs_rho.png`
- `results/figures/var_m_vs_rho.png`
- `results/figures/xi_vs_rho.png`

## Assumptions

- Zealots are fixed by default in state `+1`.
- Non-zealot initial states are i.i.d. uniform in `{-1, +1}`.
- `burn_in` is used for observable estimation and EVT post-processing.
- Default configuration uses explicit `rho_values`; alternatively, code supports `{start, stop, step}` format.

## Reproducibility

- All stochastic components use `numpy.random.default_rng(seed)`.
- Experiment scripts pass seed from config to graph generation, zealot assignment, and dynamics.

## Notes on EVT

`evt.py` implements classical block maxima with `scipy.stats.genextreme` fit and tracks `xi(rho)`.
A rise in fitted `xi` near estimated criticality can indicate heavier extreme fluctuations.

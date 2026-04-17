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

## Two-Camp Zealot Extension

The codebase now supports two competing zealot camps:
- positive zealots fixed at `+1`
- negative zealots fixed at `-1`
- free nodes following voter dynamics

### Single Two-Camp Run

```bash
python experiments/run_two_zealot_simulation.py \
  --config config/default.yaml \
  --graph-type erdos_renyi \
  --n 500 \
  --p 0.02 \
  --T 20000 \
  --burn-in 4000 \
  --pos-count 20 \
  --neg-count 30 \
  --strategy-pos highest_degree \
  --strategy-neg random \
  --print-graph-info \
  --print-magnetization \
  --results-dir results/two_camp_single
```

### Tipping Grid (Phase-Diagram Data)

```bash
python experiments/run_tipping_grid.py \
  --config config/default.yaml \
  --graph-type barabasi_albert \
  --n 800 \
  --m 3 \
  --T 30000 \
  --burn-in 6000 \
  --n-runs 8 \
  --n-pos-values 10,20,30,40 \
  --n-neg-values 10,20,30,40,50 \
  --strategy-pos random \
  --strategy-neg random \
  --results-dir results/two_camp_grid
```

Plot the saved phase diagram:

```bash
python experiments/plot_phase_diagram.py \
  --input-npz results/two_camp_grid/raw/tipping_grid_results.npz \
  --boundary-json results/two_camp_grid/raw/tipping_grid_boundary.json \
  --output-dir results/two_camp_grid/figures
```

### Strength/Strategy Comparison

```bash
python experiments/run_strength_comparison.py \
  --config config/default.yaml \
  --graph-type barabasi_albert \
  --n 600 \
  --m 3 \
  --T 25000 \
  --burn-in 5000 \
  --n-pos-values 15 \
  --n-neg-values 20,25 \
  --strategy-pos-list random,highest_degree,highest_eigenvector,farthest_spread \
  --strategy-neg-list random,highest_degree \
  --n-runs 10 \
  --results-dir results/two_camp_strength
```

### Export Dataset for Regression/Symbolic Search

```bash
python experiments/export_symbolic_dataset.py \
  --graph-types fully_connected,erdos_renyi,barabasi_albert \
  --n-values 150,250 \
  --erdos-p-values 0.01,0.02 \
  --barabasi-m-values 2,3 \
  --n-pos-values 6,10,14 \
  --n-neg-values 6,10,14,18 \
  --strategy-pos-list random,highest_degree,highest_eigenvector,farthest_spread,wl_cover \
  --strategy-neg-list random,highest_degree \
  --n-runs 6 \
  --results-dir results/two_camp_symbolic
```

Phase-1 interactive runs (override graph/rho/size directly from terminal):

```bash
# Fully connected graph with live magnetization prints
python experiments/run_simulation.py \
  --config config/default.yaml \
  --graph-type fully_connected \
  --n 500 \
  --rho 0.10 \
  --T 20000 \
  --print-graph-info \
  --print-magnetization \
  --magnetization-interval 1000 \
  --results-dir results/phase1_fc

# Erdos-Renyi
python experiments/run_simulation.py \
  --config config/default.yaml \
  --graph-type erdos_renyi \
  --n 1000 \
  --p 0.01 \
  --rho 0.05 \
  --results-dir results/phase1_er

# Barabasi-Albert
python experiments/run_simulation.py \
  --config config/default.yaml \
  --graph-type barabasi_albert \
  --n 1000 \
  --m 3 \
  --rho 0.05 \
  --results-dir results/phase1_ba
```

Run phase-transition sweep:

```bash
python experiments/sweep_rho.py --config config/default.yaml
```

Run EVT pipeline:

```bash
python experiments/run_evt.py --config config/default.yaml
```

Run fully-connected convergence-time symbolic regression (JSON + figure):

```bash
python experiments/fit_convergence_formula_fc.py \
  --n 400 \
  --T 80000 \
  --n-runs 6 \
  --rho-min 0.05 \
  --rho-max 0.50 \
  --rho-step 0.05 \
  --stable-window 500 \
  --results-dir results/fc_symbolic_v1
```

Use true PySR symbolic search (operators like `+ - * / log exp`) by switching backend:

```bash
python experiments/fit_convergence_formula_fc.py \
  --regression-method pysr \
  --pysr-niterations 800 \
  --pysr-population-size 60 \
  --pysr-maxsize 24 \
  --pysr-binary-operators "+,-,*,/" \
  --pysr-unary-operators "log,exp" \
  --results-dir results/fc_symbolic_pysr
```

Note: `pysr` is optional and typically requires both the Python package and Julia runtime.

This saves:

- `results/fc_symbolic_v1/raw/fc_convergence_dataset.json`
- `results/fc_symbolic_v1/raw/fc_symbolic_model.json`
- `results/fc_symbolic_v1/figures/fc_real_vs_symbolic_prediction.png`

Multi-N PySR protocol (fit `tau(N, rho)` and evaluate on held-out N):

```bash
python experiments/pysr_tau_multin.py \
  --n-values 100,200,300,400,500 \
  --rho-values 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 \
  --n-seeds 5 \
  --T-max 100000 \
  --stable-window 500 \
  --test-n-values 150,250 \
  --test-seeds 5 \
  --results-dir results/fc_tau_multin_pysr
```

This saves:

- `results/fc_tau_multin_pysr/raw/fc_tau_multin_training_dataset.json`
- `results/fc_tau_multin_pysr/raw/fc_tau_multin_pysr_model.json`
- `results/fc_tau_multin_pysr/raw/fc_tau_multin_eval_dataset.json`
- `results/fc_tau_multin_pysr/figures/rho_vs_tau_real_vs_pysr_N_150.png`
- `results/fc_tau_multin_pysr/figures/rho_vs_tau_real_vs_pysr_N_250.png`

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

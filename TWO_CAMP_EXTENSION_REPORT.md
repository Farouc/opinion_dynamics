# Two-Camp Zealot Voter Extension Report

Date: 2026-04-17  
Project: `opinion_dynamics`  
Scope: Extension of the existing one-sided zealot codebase to two competing zealot camps (+1 and -1), with structural tipping analysis.

---

## 1. Executive Summary

This extension adds a full two-camp research pipeline on top of the existing codebase (without removing one-camp functionality):

- Two disjoint zealot camps are now supported:
  - positive zealots fixed at `+1`
  - negative zealots fixed at `-1`
  - free nodes updated by voter dynamics
- Placement strategies are modular and extensible, including centrality-based and diversity/coverage-based methods.
- Tipping-focused observables are implemented (victory probability, crossing times, sign changes, time above threshold).
- Structural feature extraction is added at graph-level and camp-level.
- Lightweight WL (Weisfeiler-Lehman) color-refinement features are added.
- A candidate effective-strength framework computes multiple normalized and dimensionless predictors (not a single hardcoded formula).
- Multi-seed experiment orchestration and phase-diagram workflows are available.
- A tabular dataset exporter is available for downstream regression/symbolic regression.

The implementation is modular and reproducible, and it preserves existing graph generation and one-camp scripts.

---

## 2. Scientific Objective (Two-Camp Setting)

We now study two competing zealot sets:

- `Z_plus`: nodes fixed at `+1`
- `Z_minus`: nodes fixed at `-1`

Main question:

> Can a numerically smaller positive camp dominate global opinion due to structural placement advantage?

Secondary question:

> Which graph-aware, normalized structural quantities best predict tipping outcomes?

---

## 3. Theoretical Model and Definitions

### 3.1 State space and update rule

Each node has binary state `x_i(t) in {-1, +1}`.

Asynchronous update at each step:

1. Sample node `i` uniformly.
2. If `i` is zealot (`i in Z_plus union Z_minus`), skip.
3. Else sample neighbor `j` uniformly.
4. Set `x_i <- x_j`.

Zealot constraints:

- `x_i(t) = +1` for all `i in Z_plus`
- `x_i(t) = -1` for all `i in Z_minus`

### 3.2 Core observables

- Magnetization:
  - `m(t) = (1/N) * sum_i x_i(t)`
- Positive fraction:
  - `p_plus(t) = (1/N) * |{i: x_i(t)=+1}|`
- Negative fraction:
  - `p_minus(t) = (1/N) * |{i: x_i(t)=-1}|`

### 3.3 Asymptotic/tipping observables

After burn-in `t >= t_b`:

- `mean_magnetization = mean_t>=tb m(t)`
- `var_magnetization = Var_t>=tb m(t)`
- `victory_indicator = 1[mean_magnetization > 0]`
- `dominance_sign in {-1,0,+1}`
- Time to first sign change of `m(t)`
- Time to first crossing above threshold `theta`
- Fraction of time above/below threshold after burn-in
- Optional free-node flipping activity

### 3.4 Effective strength candidates

For each camp `S`, multiple candidates are computed, for example:

- `psi_size(S) = |S|`
- `psi_rho(S) = |S| / N`
- `psi_degree(S) = sum_{i in S} d_i`
- `psi_degree_norm(S) = psi_degree(S) / sum_i d_i`
- `psi_centrality(S)` (pagerank sum)
- `psi_wl(S)` (WL coverage fraction)
- `psi_dispersion(S)` (normalized spread score)
- Hybrid:
  - `psi_hybrid = alpha*psi_rho + beta*psi_degree_norm + gamma*psi_wl + delta*psi_dispersion`

For two camps, relative predictors are added:

- `delta_psi_* = psi_*(Z_plus) - psi_*(Z_minus)`
- ratio predictors `psi_*(Z_plus)/psi_*(Z_minus)`

No single formula is assumed universally correct.

---

## 4. What Was Added (Code Modules)

### 4.1 New source modules

- `src/zealot_assignment.py`
  - `assign_two_zealot_sets(...)`
  - `build_two_camp_assignment(...)`
  - Ensures disjoint masks, supports explicit node IDs, deterministic seeding.

- `src/placement_strategies.py`
  - `select_nodes_by_strategy(...)`
  - Implemented strategies:
    - `random`
    - `highest_degree`
    - `lowest_degree`
    - `highest_betweenness`
    - `highest_eigenvector`
    - `highest_pagerank`
    - `community_cover`
    - `wl_cover`
    - `wl_top_class`
    - `farthest_spread`
    - `hub_then_spread`
    - `explicit`

- `src/two_zealot_voter_model.py`
  - `step_two_zealot_voter(...)`
  - `step_two_zealot_voter_with_delta(...)`
  - `run_two_zealot_simulation(...)`

- `src/wl_features.py`
  - `wl_color_refinement(...)`
  - `compute_wl_coverage_features(...)`
  - `compute_two_camp_wl_features(...)`

- `src/graph_features.py`
  - `compute_graph_features(...)`
  - `compute_graph_context(...)`
  - `compute_camp_features(...)`
  - `compute_two_camp_comparison_features(...)`
  - Includes version-safe shortest-path fallback for NetworkX compatibility.

- `src/effective_strength.py`
  - `compute_effective_strength_candidates(...)`

- `src/tipping_analysis.py`
  - `run_two_camp_configuration(...)`
  - `run_tipping_grid(...)`
  - `extract_phase_boundary(...)`
  - `run_minimal_sanity_checks(...)`

- `src/symbolic_features_dataset.py`
  - `configuration_result_to_dataset_row(...)`
  - `save_dataset_rows(...)`

### 4.2 Extended existing source module

- `src/observables.py`
  - Added tipping metrics and asymptotic summaries while keeping previous APIs:
    - `positive_fraction`, `negative_fraction`
    - `dominance_sign`, `victory_indicator`
    - `time_to_first_sign_change`
    - `time_to_threshold_crossing`
    - `fraction_time_above_threshold`, `fraction_time_below_threshold`
    - `asymptotic_observables`, `tipping_observables`
    - `free_node_flip_activity`

### 4.3 New experiment scripts

- `experiments/run_two_zealot_simulation.py`
- `experiments/run_tipping_grid.py`
- `experiments/run_strength_comparison.py`
- `experiments/export_symbolic_dataset.py`
- `experiments/plot_phase_diagram.py`

### 4.4 Documentation and hygiene

- `README.md` updated with two-camp commands.
- `.gitignore` updated for Python caches.

---

## 5. Methodology by Pipeline Stage

### 5.1 Assignment and placement

- Positive and negative sets are selected independently but with enforced disjointness.
- Negative camp selection can exclude already selected positive nodes.
- Explicit node-list assignment is supported for controlled experiments.

### 5.2 Simulation

- Asynchronous voter updates on existing graph types.
- Optional O(1) updates for magnetization and positive fraction via state deltas.
- Deterministic randomness via `numpy.random.Generator` and derived per-run seeds.

### 5.3 Feature extraction

- Graph-level features include size, degree moments, clustering, path lengths, assortativity, spectral proxy (`lambda_2` when feasible).
- Camp-level features include size, degree/pagerank aggregates, dispersion, coverage measures.
- Comparison features include differences and ratios between camps.

### 5.4 WL structural features

- 1-WL refinement initialized by node degree.
- Features include class coverage, entropy, class-size descriptors, overlap between camps.

### 5.5 Effective strength candidates

- Computes multiple normalized candidate strengths per camp.
- Derives relative deltas/ratios for tipping prediction.

### 5.6 Aggregation

For each configuration over multiple seeds:

- mean and std of core targets
- positive wins count
- positive victory probability
- aggregated feature summaries

---

## 6. Command Cookbook

Use your environment first:

```bash
base
cd /mnt/D/mva_P2/interactions/opinion_dynamics
```

### 6.1 Single two-camp simulation

```bash
python3 experiments/run_two_zealot_simulation.py \
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

Notes:

- You can use `--pos-rho` / `--neg-rho` instead of counts.
- You can pass explicit lists with `--pos-nodes` and `--neg-nodes`.
- Strategy kwargs are JSON strings via `--strategy-kwargs-pos` and `--strategy-kwargs-neg`.

### 6.2 Tipping phase diagram grid

```bash
python3 experiments/run_tipping_grid.py \
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

Plot the phase diagram:

```bash
python3 experiments/plot_phase_diagram.py \
  --input-npz results/two_camp_grid/raw/tipping_grid_results.npz \
  --boundary-json results/two_camp_grid/raw/tipping_grid_boundary.json \
  --output-dir results/two_camp_grid/figures
```

### 6.3 Strategy / effective-strength comparison

```bash
python3 experiments/run_strength_comparison.py \
  --config config/default.yaml \
  --graph-type barabasi_albert \
  --n 600 \
  --m 3 \
  --T 25000 \
  --burn-in 5000 \
  --n-pos-values 15 \
  --n-neg-values 20,25 \
  --strategy-pos-list random,highest_degree,highest_eigenvector,farthest_spread,wl_cover,hub_then_spread \
  --strategy-neg-list random,highest_degree \
  --n-runs 10 \
  --results-dir results/two_camp_strength
```

### 6.4 Export regression/symbolic dataset

```bash
python3 experiments/export_symbolic_dataset.py \
  --graph-types fully_connected,erdos_renyi,barabasi_albert \
  --n-values 150,250 \
  --erdos-p-values 0.01,0.02 \
  --barabasi-m-values 2,3 \
  --n-pos-values 6,10,14 \
  --n-neg-values 6,10,14,18 \
  --strategy-pos-list random,highest_degree,highest_eigenvector,farthest_spread,wl_cover,hub_then_spread \
  --strategy-neg-list random,highest_degree \
  --n-runs 6 \
  --results-dir results/two_camp_symbolic
```

---

## 7. Regime Templates (A/B/C/D)

### Regime A: Positive minority vs negative majority

```bash
python3 experiments/run_strength_comparison.py \
  --graph-type barabasi_albert --n 800 --m 3 \
  --n-pos-values 10,15 \
  --n-neg-values 20,25,30 \
  --strategy-pos-list random,highest_degree,farthest_spread,wl_cover \
  --strategy-neg-list random \
  --n-runs 12 \
  --results-dir results/regime_A_minority_vs_majority
```

### Regime B: Equal size, different placement

```bash
python3 experiments/run_strength_comparison.py \
  --graph-type erdos_renyi --n 800 --p 0.01 \
  --n-pos-values 20 \
  --n-neg-values 20 \
  --strategy-pos-list random,highest_degree,highest_eigenvector,wl_cover,farthest_spread \
  --strategy-neg-list random,highest_degree \
  --n-runs 16 \
  --results-dir results/regime_B_equal_size
```

### Regime C: Sweep camp imbalance (`delta_n = n_pos - n_neg`)

```bash
python3 experiments/run_tipping_grid.py \
  --graph-type erdos_renyi --n 1000 --p 0.01 \
  --n-pos-values 10,20,30,40,50 \
  --n-neg-values 10,20,30,40,50 \
  --n-runs 10 \
  --results-dir results/regime_C_imbalance_grid
```

### Regime D: Graph family comparison

```bash
python3 experiments/export_symbolic_dataset.py \
  --graph-types fully_connected,erdos_renyi,barabasi_albert,grid_lattice \
  --n-values 200,400 \
  --L-values 20 \
  --erdos-p-values 0.01,0.02 \
  --barabasi-m-values 2,3,4 \
  --n-pos-values 8,12,16 \
  --n-neg-values 8,12,16,20 \
  --strategy-pos-list random,highest_degree,farthest_spread,wl_cover \
  --strategy-neg-list random,highest_degree \
  --n-runs 8 \
  --results-dir results/regime_D_graph_family
```

---

## 8. Output Files and Schemas

### 8.1 Single run (`run_two_zealot_simulation.py`)

- `results/.../raw/two_camp_simulation_*.npz`
  - arrays: `magnetization`, `positive_fraction`, `negative_fraction`, zealot masks, initial/final states, optional trajectory
- `results/.../raw/two_camp_summary_*.json`
  - config + asymptotic/tipping summaries + selected zealot nodes
- `results/.../figures/two_camp_timeseries_*.png`

### 8.2 Tipping grid (`run_tipping_grid.py`)

- `results/.../raw/tipping_grid_results.npz`
  - `n_pos_values`, `n_neg_values`, `mean_m_grid`, `positive_win_probability_grid`
- `results/.../raw/tipping_grid_records.json`
- `results/.../raw/tipping_grid_records.csv`
- `results/.../raw/tipping_grid_boundary.json`

### 8.3 Strength comparison (`run_strength_comparison.py`)

- `results/.../raw/strength_comparison_rows.json`
- `results/.../raw/strength_comparison_rows.csv`
- `results/.../raw/strength_predictive_rankings.json`
- figures:
  - `strength_candidate_correlations.png`
  - `strength_best_candidate_scatter.png`

### 8.4 Symbolic dataset (`export_symbolic_dataset.py`)

- `results/.../raw/symbolic_dataset_two_camp.json`
- `results/.../raw/symbolic_dataset_two_camp.csv`
- `results/.../raw/symbolic_dataset_two_camp_metadata.json`

---

## 9. Reproducibility and Determinism

- Main RNG backend: `numpy.random.default_rng(seed)`.
- Global seeds are propagated into derived per-run and per-assignment seeds.
- Multi-run aggregation uses explicit seed streams.
- WL/features are deterministic given graph and parameters.

---

## 10. Minimal Sanity Checks Implemented

Sanity helper:

```bash
python3 -c "import json; from src.tipping_analysis import run_minimal_sanity_checks; print(json.dumps(run_minimal_sanity_checks(seed=7), indent=2))"
```

Current smoke outcomes (example):

- Equal-size random on complete graph: near-balanced outcomes (`P_plus_win ~ 0.55`, mean `m` near zero).
- Positive hub-vs-random minority test on BA: hub placement strongly improves positive win probability versus random baseline.

These are sanity checks, not claims of universality.

---

## 11. Assumptions and Limitations

- WL module is lightweight 1-WL refinement (sufficient for structural partitioning but not exhaustive graph isomorphism analysis).
- Centrality computations are exact NetworkX variants; for very large graphs these can be costly.
- Spectral feature (`lambda_2`) is computed when feasible; may return `nan` for hard cases.
- Some aggregate metrics can be undefined for degenerate inputs (handled as `nan` where appropriate).
- This extension intentionally does not make EVT the central analysis path for the two-camp objective.

---

## 12. Practical Next Steps

1. Run Regime A/B/C/D campaigns with larger `n_runs` and higher `T` for robust statistics.
2. Compare predictive performance of `delta_psi_*` candidates against raw imbalance `n_pos - n_neg`.
3. Use `symbolic_dataset_two_camp.csv` for external regression/symbolic modeling.
4. Evaluate whether WL/diversity terms improve predictive power over degree-only baselines.
5. Estimate empirical tipping boundary where `mean_m` changes sign and/or `P_plus_win` crosses `0.5`.

---

## 13. Files Added/Modified in This Extension

Added:

- `src/zealot_assignment.py`
- `src/placement_strategies.py`
- `src/two_zealot_voter_model.py`
- `src/wl_features.py`
- `src/graph_features.py`
- `src/effective_strength.py`
- `src/tipping_analysis.py`
- `src/symbolic_features_dataset.py`
- `experiments/run_two_zealot_simulation.py`
- `experiments/run_tipping_grid.py`
- `experiments/run_strength_comparison.py`
- `experiments/export_symbolic_dataset.py`
- `experiments/plot_phase_diagram.py`

Modified:

- `src/observables.py` (extended for two-camp tipping metrics)
- `README.md` (new usage section)
- `.gitignore` (Python cache patterns)

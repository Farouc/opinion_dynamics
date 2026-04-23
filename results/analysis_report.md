# Two-Camp Zealot Voter Model — Results Analysis
**Project:** MVA Interactions | **Date:** 2026-04-23

---

## 0. Executive Summary
- Exp2 finds **92 minority-dominance cells** in total; max per setting is **36** for `barabasi_albert | highest_degree vs random`.
- Exp4 best overall metric is **ratio_psi_degree_norm** with logistic AUC **0.994** and steepness **a=37.960**.
- Exp5 regression `placement_advantage ~ degree_heterogeneity` gives slope **0.281** and R² **0.470**.
- Exp1: No high-variance metastable configurations above std>0.3 were detected.
- Exp3 WL-cover vs random yields average ΔP_win **-0.175** across tested graph instances.

---

## 1. Opinion Dynamics and Convergence (Exp 1)

### 1.1 Convergence speed by graph type and strategy
| graph | regime | strategy_pair | mean_convergence_time | mean_final_m | inter_seed_std |
| --- | --- | --- | --- | --- | --- |
| barabasi_albert | regimeA | highest_degree vs random | 853.000 | 0.548 | 0.114 |
| barabasi_albert | regimeA | hub_then_spread vs highest_degree | Data not available | -0.059 | 0.043 |
| barabasi_albert | regimeA | random vs random | 5340.000 | -0.339 | 0.220 |
| barabasi_albert | regimeB | highest_degree vs random | 437.200 | 0.725 | 0.060 |
| barabasi_albert | regimeB | hub_then_spread vs highest_degree | 2295.400 | 0.227 | 0.060 |
| barabasi_albert | regimeB | random vs random | 3836.600 | 0.018 | 0.041 |
| erdos_renyi | regimeA | highest_degree vs random | 3096.000 | 0.002 | 0.102 |
| erdos_renyi | regimeA | hub_then_spread vs highest_degree | 1717.200 | -0.325 | 0.103 |
| erdos_renyi | regimeA | random vs random | 5600.200 | -0.328 | 0.181 |
| erdos_renyi | regimeB | highest_degree vs random | 2743.400 | 0.266 | 0.076 |
| erdos_renyi | regimeB | hub_then_spread vs highest_degree | 5313.250 | -0.009 | 0.168 |
| erdos_renyi | regimeB | random vs random | 5726.600 | 0.112 | 0.136 |
| fully_connected | regimeA | highest_degree vs random | 271.800 | -0.424 | 0.092 |
| fully_connected | regimeA | hub_then_spread vs highest_degree | 408.400 | -0.252 | 0.139 |
| fully_connected | regimeA | random vs random | 657.800 | -0.290 | 0.148 |
| fully_connected | regimeB | highest_degree vs random | 2611.400 | -0.024 | 0.080 |
| fully_connected | regimeB | hub_then_spread vs highest_degree | 939.800 | 0.024 | 0.147 |
| fully_connected | regimeB | random vs random | 1799.600 | -0.070 | 0.151 |

### 1.2 Effect of placement strategy on convergence
- `barabasi_albert` fastest observed strategy is `highest_degree vs random` with mean convergence time 437.200.
- `erdos_renyi` fastest observed strategy is `hub_then_spread vs highest_degree` with mean convergence time 1717.200.
- `fully_connected` fastest observed strategy is `highest_degree vs random` with mean convergence time 271.800.
Higher spread in final magnetization across seeds indicates coexistence of opposite long-run outcomes under identical macro-parameters.

### 1.3 Metastability indicators
No configuration with inter-seed std(final m) > 0.3 was found.

## 2. Phase Diagram and Tipping Boundary (Exp 2)

### 2.1 Tipping boundary by strategy and graph
- `barabasi_albert | highest_degree vs random`: boundary has 0 sampled points, appears nonlinear/offset, and does not intersect near n_pos=n_neg.
- `barabasi_albert | hub_then_spread vs random`: boundary has 1 sampled points, appears nonlinear/offset, and does not intersect near n_pos=n_neg.
- `barabasi_albert | random vs random`: boundary has 14 sampled points, appears roughly linear-like, and does intersect near n_pos=n_neg.
- `erdos_renyi | highest_degree vs random`: boundary has 11 sampled points, appears roughly linear-like, and does intersect near n_pos=n_neg.
- `erdos_renyi | hub_then_spread vs random`: boundary has 12 sampled points, appears roughly linear-like, and does intersect near n_pos=n_neg.
- `erdos_renyi | random vs random`: boundary has 12 sampled points, appears roughly linear-like, and does intersect near n_pos=n_neg.

### 2.2 Boundary displacement due to placement
| graph | strategy_pair | mean_shift_npos | reduction_in_required_npos |
| --- | --- | --- | --- |
| barabasi_albert | highest_degree vs random | Data not available | Data not available |
| barabasi_albert | hub_then_spread vs random | -33.000 | 33.000 |
| erdos_renyi | highest_degree vs random | -10.731 | 10.731 |
| erdos_renyi | hub_then_spread vs random | -7.167 | 7.167 |
Negative `mean_shift_npos` means fewer positive zealots are needed to hit the P=0.5 boundary compared with random placement.

### 2.3 Minority dominance cells
| graph | strategy_pair | n_minority_win_cells | mean_P_win_in_those_cells |
| --- | --- | --- | --- |
| barabasi_albert | highest_degree vs random | 36 | 0.993 |
| barabasi_albert | hub_then_spread vs random | 35 | 0.979 |
| barabasi_albert | random vs random | 2 | 0.625 |
| erdos_renyi | highest_degree vs random | 13 | 0.837 |
| erdos_renyi | hub_then_spread vs random | 6 | 0.833 |
| erdos_renyi | random vs random | 0 | Data not available |

### 2.4 Critical region
| graph | strategy_pair | n_critical_cells | mean_crossing_time | critical_width_npos | critical_width_nneg |
| --- | --- | --- | --- | --- | --- |
| barabasi_albert | highest_degree vs random | 0 | Data not available | Data not available | Data not available |
| barabasi_albert | hub_then_spread vs random | 0 | Data not available | Data not available | Data not available |
| barabasi_albert | random vs random | 5 | 438.150 | 25.000 | 35.000 |
| erdos_renyi | highest_degree vs random | 3 | 2361.417 | 20.000 | 45.000 |
| erdos_renyi | hub_then_spread vs random | 3 | 1524.292 | 15.000 | 25.000 |
| erdos_renyi | random vs random | 4 | 1249.500 | 30.000 | 20.000 |
The band with |P_win-0.5|<=0.1 is used as the empirical critical zone; larger crossing times there indicate slowed dynamics near tipping.

## 3. WL-Based Structural Placement (Exp 3)

### 3.1 WL partition structure
| graph | n_classes | n_nodes | top1_class_frac | top5_class_frac | wl_entropy | effective_classes |
| --- | --- | --- | --- | --- | --- | --- |
| barabasi_albert | 295 | 300 | 0.007 | 0.033 | 5.681 | 293.148 |
| erdos_renyi | 300 | 300 | 0.003 | 0.017 | 5.704 | 300.000 |

### 3.2 Strategy ranking
| graph | strategy | P_plus_win | mean_m | psi_degree_norm | psi_wl_coverage | rank |
| --- | --- | --- | --- | --- | --- | --- |
| barabasi_albert | highest_degree | 1.000 | 0.632 | 0.258 | 0.051 | 1 |
| barabasi_albert | farthest_spread | 1.000 | 0.177 | 0.087 | 0.051 | 2 |
| barabasi_albert | wl_top_class | 1.000 | 0.617 | 0.258 | 0.051 | 3 |
| barabasi_albert | random | 0.350 | -0.069 | 0.052 | 0.051 | 4 |
| barabasi_albert | wl_cover | 0.000 | -0.379 | 0.025 | 0.051 | 5 |
| erdos_renyi | highest_degree | 1.000 | 0.169 | 0.100 | 0.050 | 1 |
| erdos_renyi | wl_top_class | 1.000 | 0.164 | 0.100 | 0.050 | 2 |
| erdos_renyi | farthest_spread | 0.050 | -0.146 | 0.048 | 0.050 | 3 |
| erdos_renyi | random | 0.000 | -0.127 | 0.050 | 0.050 | 4 |
| erdos_renyi | wl_cover | 0.000 | -0.597 | 0.012 | 0.050 | 5 |

### 3.3 Does WL coverage matter?
| graph | feature | Pearson_r_with_P_plus_win |
| --- | --- | --- |
| barabasi_albert | psi_degree_norm | 0.779 |
| barabasi_albert | psi_dispersion | -0.637 |
| barabasi_albert | psi_pagerank | 0.781 |
| barabasi_albert | psi_wl_coverage | Data not available |
| erdos_renyi | psi_degree_norm | 0.921 |
| erdos_renyi | psi_dispersion | -0.897 |
| erdos_renyi | psi_pagerank | 0.921 |
| erdos_renyi | psi_wl_coverage | Data not available |

### 3.4 Interpretation
- `barabasi_albert`: `wl_cover - random` gives ΔP_win=-0.350.
- `erdos_renyi`: `wl_cover - random` gives ΔP_win=0.000.
WL-cover is expected to help when structural diversity (coverage of distinct local roles) matters more than raw hub concentration.

## 4. Effective Strength Metrics (Exp 4)

### 4.1 Metric ranking by predictive power
| metric | Pearson_r | Spearman_r | logistic_AUC | logistic_steepness_a | rank |
| --- | --- | --- | --- | --- | --- |
| ratio_psi_degree_norm | 0.868 | 0.908 | 0.994 | 37.960 | 1 |
| ratio_psi_pagerank | 0.870 | 0.913 | 0.994 | 39.206 | 2 |
| ratio_psi_hybrid | 0.723 | 0.797 | 0.926 | 77.160 | 3 |
| ratio_psi_rho | 0.580 | 0.601 | 0.819 | 11.564 | 4 |
| ratio_psi_dispersion | -0.530 | -0.501 | 0.771 | -42.355 | 5 |
| ratio_psi_wl | 0.349 | 0.341 | 0.679 | 6.825 | 6 |

### 4.2 Best metric per graph type
| graph_type | best_metric | AUC | runner_up_metric | AUC_runner_up |
| --- | --- | --- | --- | --- |
| barabasi_albert | ratio_psi_degree_norm | 0.999 | ratio_psi_pagerank | 0.997 |
| erdos_renyi | ratio_psi_degree_norm | 0.993 | ratio_psi_pagerank | 0.993 |
| fully_connected | ratio_psi_rho | 0.988 | ratio_psi_degree_norm | 0.988 |

### 4.3 Does a single metric predict the winner?
The best metric is `ratio_psi_degree_norm` with AUC=0.994. Using the fitted centered logistic model, transition steepness is a=37.960; the balance point is centered at ratio_psi=0.5.
Interpretation: larger |a| implies a sharper switch in win probability around the critical ratio.

### 4.4 Minority dominance regime
- Minority-win configurations found: **71**.
- `ratio_psi_degree_norm` in minority-win cases: mean=0.663, std=0.110, Q25=0.570, Q75=0.772.
- Best separator between minority-win and majority-win positive cases: `ratio_psi_rho` with separation AUC=1.000.

### 4.5 The critical/metastable band
- Best-metric band for near-criticality: [0.489, 0.511], width=0.021, rows=73.
- In-band mean P_win=0.434, mean std_m=0.080.
- Mean crossing times in this band: Data not available (not logged in Exp4 raw dataset).

## 5. Graph Topology Effects (Exp 5)

### 5.1 Convergence speed by topology
| graph_family | parameter | mean_convergence_time | strategy |
| --- | --- | --- | --- |
| barabasi_albert | BA_m2 | 902.000 | highest_degree vs random |
| barabasi_albert | BA_m2 | 9217.867 | random vs random |
| barabasi_albert | BA_m3 | 1468.600 | highest_degree vs random |
| barabasi_albert | BA_m3 | 9198.143 | random vs random |
| barabasi_albert | BA_m5 | 1456.133 | highest_degree vs random |
| barabasi_albert | BA_m5 | 9984.714 | random vs random |
| erdos_renyi | ER_p0008 | 10862.933 | highest_degree vs random |
| erdos_renyi | ER_p0008 | 17070.200 | random vs random |
| erdos_renyi | ER_p0015 | 8449.786 | highest_degree vs random |
| erdos_renyi | ER_p0015 | 12852.500 | random vs random |
| erdos_renyi | ER_p003 | 12193.667 | highest_degree vs random |
| erdos_renyi | ER_p003 | 11432.000 | random vs random |
| fully_connected | FC_N200 | 5881.133 | highest_degree vs random |
| fully_connected | FC_N200 | 5767.333 | random vs random |

### 5.2 Placement advantage vs degree heterogeneity
| graph_config | degree_heterogeneity | P_win_highdeg | P_win_random | placement_advantage |
| --- | --- | --- | --- | --- |
| BA_m2 | 1.176 | 1.000 | 0.333 | 0.667 |
| BA_m3 | 0.991 | 1.000 | 0.467 | 0.533 |
| BA_m5 | 0.931 | 1.000 | 0.400 | 0.600 |
| ER_p0008 | 0.456 | 1.000 | 0.400 | 0.600 |
| ER_p0015 | 0.349 | 1.000 | 0.533 | 0.467 |
| ER_p003 | 0.265 | 1.000 | 0.400 | 0.600 |
| FC_N200 | 0.000 | 0.533 | 0.400 | 0.133 |
- Regression: slope=0.281, R²=0.470, intercept=0.347.
- Substantial advantage threshold (placement_advantage>0.1): heterogeneity >= 0.000.

### 5.3 Qualitative topology regimes
- **Fully connected**: mean placement advantage=0.133, mean convergence (highdeg/random)=5881.133/5767.333.
- **Erdős–Rényi**: mean placement advantage=0.556, range=[0.467, 0.600].
- **Barabási–Albert**: mean placement advantage=0.600, range=[0.533, 0.667], consistent with hub-amplified effects when heterogeneity is high.

### 5.4 Implications
Topology with higher degree heterogeneity is more sensitive to strategic placement, which implies real social networks with hubs may be disproportionately vulnerable to small but well-placed influence groups.

## 6. Cross-Experiment Synthesis

### 6.1 Consistent findings across experiments
- Strategic placement shifts tipping behavior beyond raw zealot counts (Exp2, Exp5).
- `ratio_psi_degree_norm` is the strongest global scalar predictor in Exp4 (AUC=0.994).
- Minority dominance is empirically present in phase-grid cells (Exp2) and configuration-level data (Exp4).

### 6.2 Contradictions or surprises
- WL-cover does not universally beat random: 1 graph instance(s) show negative ΔP_win.

### 6.3 Best overall predictor of zealot dominance
The metric **ratio_psi_degree_norm**, computed as a ratio between positive and total camp strength, achieves AUC=0.999 on BA and AUC=0.993 on ER in this run.

### 6.4 Open questions for future work
1. Do the same metric rankings hold for larger N and longer T with tighter confidence intervals?
2. Can crossing-time observables be logged in Exp4 to quantify metastability directly in metric space?
3. How robust are WL-based gains under alternative community/WL-depth definitions?
4. Is there a universal rescaling collapsing BA and ER tipping curves onto one master relation?
5. Which causal mechanism (hub capture vs structural coverage) dominates in real-world network data?

## 7. Data Availability

List of raw files successfully read:
- `results/exp1_timeseries_and_convergence/raw/trajectories_barabasi_albert_N500_m3_npos15_nneg30_regimeA_highest_degree_vs_random_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_barabasi_albert_N500_m3_npos15_nneg30_regimeA_highest_degree_vs_random_allseeds.json` (json, 319.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_barabasi_albert_N500_m3_npos15_nneg30_regimeA_hub_then_spread_vs_highest_degree_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_barabasi_albert_N500_m3_npos15_nneg30_regimeA_hub_then_spread_vs_highest_degree_allseeds.json` (json, 331.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_barabasi_albert_N500_m3_npos15_nneg30_regimeA_random_vs_random_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_barabasi_albert_N500_m3_npos15_nneg30_regimeA_random_vs_random_allseeds.json` (json, 313.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_barabasi_albert_N500_m3_npos25_nneg25_regimeB_highest_degree_vs_random_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_barabasi_albert_N500_m3_npos25_nneg25_regimeB_highest_degree_vs_random_allseeds.json` (json, 319.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_barabasi_albert_N500_m3_npos25_nneg25_regimeB_hub_then_spread_vs_highest_degree_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_barabasi_albert_N500_m3_npos25_nneg25_regimeB_hub_then_spread_vs_highest_degree_allseeds.json` (json, 329.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_barabasi_albert_N500_m3_npos25_nneg25_regimeB_random_vs_random_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_barabasi_albert_N500_m3_npos25_nneg25_regimeB_random_vs_random_allseeds.json` (json, 313.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_erdos_renyi_N500_p002_npos15_nneg30_regimeA_highest_degree_vs_random_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_erdos_renyi_N500_p002_npos15_nneg30_regimeA_highest_degree_vs_random_allseeds.json` (json, 320.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_erdos_renyi_N500_p002_npos15_nneg30_regimeA_hub_then_spread_vs_highest_degree_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_erdos_renyi_N500_p002_npos15_nneg30_regimeA_hub_then_spread_vs_highest_degree_allseeds.json` (json, 328.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_erdos_renyi_N500_p002_npos15_nneg30_regimeA_random_vs_random_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_erdos_renyi_N500_p002_npos15_nneg30_regimeA_random_vs_random_allseeds.json` (json, 311.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_erdos_renyi_N500_p002_npos25_nneg25_regimeB_highest_degree_vs_random_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_erdos_renyi_N500_p002_npos25_nneg25_regimeB_highest_degree_vs_random_allseeds.json` (json, 318.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_erdos_renyi_N500_p002_npos25_nneg25_regimeB_hub_then_spread_vs_highest_degree_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_erdos_renyi_N500_p002_npos25_nneg25_regimeB_hub_then_spread_vs_highest_degree_allseeds.json` (json, 330.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_erdos_renyi_N500_p002_npos25_nneg25_regimeB_random_vs_random_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_erdos_renyi_N500_p002_npos25_nneg25_regimeB_random_vs_random_allseeds.json` (json, 313.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_fully_connected_N200_npos15_nneg30_regimeA_highest_degree_vs_random_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_fully_connected_N200_npos15_nneg30_regimeA_highest_degree_vs_random_allseeds.json` (json, 308.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_fully_connected_N200_npos15_nneg30_regimeA_hub_then_spread_vs_highest_degree_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_fully_connected_N200_npos15_nneg30_regimeA_hub_then_spread_vs_highest_degree_allseeds.json` (json, 317.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_fully_connected_N200_npos15_nneg30_regimeA_random_vs_random_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_fully_connected_N200_npos15_nneg30_regimeA_random_vs_random_allseeds.json` (json, 301.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_fully_connected_N200_npos25_nneg25_regimeB_highest_degree_vs_random_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_fully_connected_N200_npos25_nneg25_regimeB_highest_degree_vs_random_allseeds.json` (json, 310.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_fully_connected_N200_npos25_nneg25_regimeB_hub_then_spread_vs_highest_degree_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_fully_connected_N200_npos25_nneg25_regimeB_hub_then_spread_vs_highest_degree_allseeds.json` (json, 317.0B): keys=12
- `results/exp1_timeseries_and_convergence/raw/trajectories_fully_connected_N200_npos25_nneg25_regimeB_random_vs_random_allseeds.npz` (npz, 1.5MB): magnetization:(5, 20000), positive_fraction:(5, 20000)
- `results/exp1_timeseries_and_convergence/raw/trajectories_fully_connected_N200_npos25_nneg25_regimeB_random_vs_random_allseeds.json` (json, 302.0B): keys=12
- `results/exp2_phase_diagram_tipping/raw/phase_grid_barabasi_albert_N800_m3_highest_degree_vs_random.npz` (npz, 3.1KB): n_pos_values:(8,), n_neg_values:(9,), mean_magnetization:(8, 9), P_plus_win:(8, 9), mean_crossing_time:(8, 9)
- `results/exp2_phase_diagram_tipping/raw/phase_grid_barabasi_albert_N800_m3_highest_degree_vs_random.csv` (csv, 4.4KB): rows=72, cols=9
- `results/exp2_phase_diagram_tipping/raw/phase_grid_barabasi_albert_N800_m3_hub_then_spread_vs_random.npz` (npz, 3.1KB): n_pos_values:(8,), n_neg_values:(9,), mean_magnetization:(8, 9), P_plus_win:(8, 9), mean_crossing_time:(8, 9)
- `results/exp2_phase_diagram_tipping/raw/phase_grid_barabasi_albert_N800_m3_hub_then_spread_vs_random.csv` (csv, 4.5KB): rows=72, cols=9
- `results/exp2_phase_diagram_tipping/raw/phase_grid_barabasi_albert_N800_m3_random_vs_random.npz` (npz, 3.1KB): n_pos_values:(8,), n_neg_values:(9,), mean_magnetization:(8, 9), P_plus_win:(8, 9), mean_crossing_time:(8, 9)
- `results/exp2_phase_diagram_tipping/raw/phase_grid_barabasi_albert_N800_m3_random_vs_random.csv` (csv, 4.2KB): rows=72, cols=9
- `results/exp2_phase_diagram_tipping/raw/phase_grid_erdos_renyi_N800_p0015_highest_degree_vs_random.npz` (npz, 3.1KB): n_pos_values:(8,), n_neg_values:(9,), mean_magnetization:(8, 9), P_plus_win:(8, 9), mean_crossing_time:(8, 9)
- `results/exp2_phase_diagram_tipping/raw/phase_grid_erdos_renyi_N800_p0015_highest_degree_vs_random.csv` (csv, 4.6KB): rows=72, cols=9
- `results/exp2_phase_diagram_tipping/raw/phase_grid_erdos_renyi_N800_p0015_hub_then_spread_vs_random.npz` (npz, 3.1KB): n_pos_values:(8,), n_neg_values:(9,), mean_magnetization:(8, 9), P_plus_win:(8, 9), mean_crossing_time:(8, 9)
- `results/exp2_phase_diagram_tipping/raw/phase_grid_erdos_renyi_N800_p0015_hub_then_spread_vs_random.csv` (csv, 4.7KB): rows=72, cols=9
- `results/exp2_phase_diagram_tipping/raw/phase_grid_erdos_renyi_N800_p0015_random_vs_random.npz` (npz, 3.1KB): n_pos_values:(8,), n_neg_values:(9,), mean_magnetization:(8, 9), P_plus_win:(8, 9), mean_crossing_time:(8, 9)
- `results/exp2_phase_diagram_tipping/raw/phase_grid_erdos_renyi_N800_p0015_random_vs_random.csv` (csv, 4.1KB): rows=72, cols=9
- `results/exp3_wl_cluster_placement/raw/wl_strategy_comparison_barabasi_albert_N300_m2.csv` (csv, 1.1KB): rows=5, cols=15
- `results/exp3_wl_cluster_placement/raw/wl_strategy_comparison_erdos_renyi_N300_p002.csv` (csv, 1.1KB): rows=5, cols=15
- `results/exp3_wl_cluster_placement/raw/wl_features_barabasi_albert_N300_m2.json` (json, 6.8KB): keys=4
- `results/exp3_wl_cluster_placement/raw/wl_features_erdos_renyi_N300_p002.json` (json, 6.8KB): keys=4
- `results/exp4_effective_strength_candidates/raw/strength_dataset_all_configs.csv` (csv, 525.5KB): rows=900, cols=45
- `results/exp4_effective_strength_candidates/raw/metric_correlations.json` (json, 860.0B): keys=3
- `results/exp5_graph_family_comparison/raw/graph_family_comparison_highest_degree_vs_random.csv` (csv, 1.5KB): rows=7, cols=16
- `results/exp5_graph_family_comparison/raw/graph_family_comparison_random_vs_random.csv` (csv, 1.5KB): rows=7, cols=16
- `results/exp5_graph_family_comparison/raw/degree_distributions_BA_m2.json` (json, 3.6KB): keys=6
- `results/exp5_graph_family_comparison/raw/degree_distributions_BA_m3.json` (json, 3.6KB): keys=6
- `results/exp5_graph_family_comparison/raw/degree_distributions_BA_m5.json` (json, 3.7KB): keys=6
- `results/exp5_graph_family_comparison/raw/degree_distributions_ER_p0008.json` (json, 3.6KB): keys=6
- `results/exp5_graph_family_comparison/raw/degree_distributions_ER_p0015.json` (json, 3.7KB): keys=6
- `results/exp5_graph_family_comparison/raw/degree_distributions_ER_p003.json` (json, 4.0KB): keys=6
- `results/exp5_graph_family_comparison/raw/degree_distributions_FC_N200.json` (json, 1.9KB): keys=6

Missing expected files/patterns:
- None.

*Generated automatically by `experiments/analyze_and_report.py`*

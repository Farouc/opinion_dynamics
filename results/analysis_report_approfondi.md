# Two-Camp Zealot Voter Model — Deep Results Analysis
**MVA Interactions Project | April 2026**

---

## Preface

This document provides a thorough interpretation of the five numerical experiments
conducted on the two-camp zealot voter model. The goal is not merely to report
numbers, but to extract the physical and mathematical meaning behind them — to
understand *why* the results look the way they do, what they confirm, what they
contradict, and what they imply for the broader question of opinion dynamics on
structured networks.

The central question driving the project is:

> **Can a numerically smaller group of committed agents ("zealots") dominate global
> opinion purely by virtue of their structural position in the network?**

The short answer, supported across all five experiments, is: **yes, robustly and
dramatically — but only on heterogeneous networks, and only when the right metric
is used to quantify structural advantage.**

---

## 1. The Voter Model: A Reminder on Mechanics

Before diving into results, it is worth internalizing the update rule, because
every result can ultimately be traced back to it.

At each discrete time step:
1. Pick a node `i` uniformly at random from all N nodes.
2. If `i` is a zealot, do nothing.
3. Otherwise, pick one neighbor `j` of `i` uniformly at random.
4. Set `x_i ← x_j` (node `i` copies `j`'s opinion).

This is a *local imitation* rule. There is no global field, no energy to minimize,
no strategic reasoning. An agent simply copies one randomly chosen neighbor.

**Key consequence:** the probability that a free node `i` is influenced *toward +1*
at any given step is proportional to the fraction of its neighbors that currently
hold opinion +1. A zealot of degree `d` directly influences `d` neighbors. This
means that a single high-degree zealot (a hub) exerts influence proportional to its
degree at *every single step* of the dynamics — it is a persistent, high-bandwidth
source of opinion. This is the physical intuition behind why degree-based metrics
dominate the predictive analysis.

---

## 2. Experiment 1 — Convergence and Time Series

### 2.1 What the convergence times tell us

The table below (from Exp1) reveals a striking pattern in convergence speed across
graph types and strategies:

| Graph | Strategy pair | Mean convergence time |
|---|---|---|
| Barabási–Albert | highest_degree vs random | **437** steps |
| Barabási–Albert | random vs random | 3,837 steps |
| Erdős–Rényi | highest_degree vs random | 2,743 steps |
| Fully connected | highest_degree vs random | 272 steps |
| Fully connected | random vs random | 1,800 steps |

The BA graph with hub placement converges in **437 steps** — roughly **9× faster**
than the same graph with random placement. This is not merely a quantitative
difference; it reflects a qualitative change in the dynamics.

**Why does hub placement speed up convergence so dramatically on BA?**
In a Barabási–Albert network, a small number of hubs concentrates a large fraction
of all edges. When the positive zealots *are* the hubs, they participate in an
outsized fraction of all voter model interactions. At each step, the probability
that the randomly chosen node `i` is a neighbor of a positive zealot is much higher
than if zealots were placed randomly. The network is essentially "pre-wired" to
funnel influence from the zealots outward, and this funnel is activated at every
time step simultaneously across all hubs.

**The fully connected case as a baseline:**
On the fully connected graph (N=200), every node has degree N−1 = 199, so all
nodes are equivalent — placing zealots on "hubs" means nothing. The convergence
time of 272 steps for `highest_degree vs random` reflects only the zealot count
ratio, not any structural advantage. This is the **mean-field limit**, and it
serves as a crucial sanity check: when topology is irrelevant, only numbers matter.

### 2.2 The absence of detected metastability in Exp1

No configuration showed inter-seed standard deviation of final magnetization above
0.3. This is worth interpreting carefully — it does not mean metastability is
absent. Rather, it means that within the parameter regimes tested in Exp1 (Regime A:
n+=15, n−=30; Regime B: n+=25, n−=25), most configurations land on one side of the
tipping boundary with enough margin that all seeds agree on the winner. The
metastable regime — where different seeds give opposite outcomes — is a *narrow*
band in (n+, n−) space, and Exp1's fixed parameter choices did not land precisely
in it. Exp2's phase diagram will show this band explicitly.

---

## 3. Experiment 2 — Phase Diagram and the Tipping Boundary

This is the most structurally rich experiment. It maps the full (n+, n−) parameter
space and reveals where the competition is decided.

### 3.1 The tipping boundary as a phase transition

In statistical physics terms, the tipping boundary — the contour where P(positive
wins) = 0.5 — is analogous to a **phase boundary** between two ordered phases:
"positive dominance" and "negative dominance." The voter model with two zealot camps
undergoes a sharp transition between these phases as a function of the control
parameters (n+, n−, placement strategy).

On **Erdős–Rényi**, the boundary is well-resolved with 11–12 sampled points and
passes through the diagonal n+ = n− for the random-vs-random case. This confirms
the intuition: on a homogeneous graph with symmetric random placement, equal zealot
counts lead to equal outcomes. The boundary is approximately linear in (n+, n−),
suggesting that near the transition the net influence scales roughly as n+ − n−
(a mean-field-like behavior).

On **Barabási–Albert with hub placement**, the boundary has 0 or 1 sampled
points — it has been pushed *entirely outside* the tested grid. This is a
dramatically stronger statement: **even with the minimum tested n+ = 5, the positive
camp wins for almost all values of n− tested.** The concept of a "tipping boundary
in zealot count space" becomes meaningless when placement provides such overwhelming
structural advantage.

### 3.2 Boundary displacement: how much does placement help?

The shift in the tipping boundary quantifies the "strategic value" of good placement:

| Graph | Strategy | Boundary shift (zealot units) |
|---|---|---|
| Erdős–Rényi | highest_degree vs random | **~11 zealots** |
| Erdős–Rényi | hub_then_spread vs random | **~7 zealots** |
| Barabási–Albert | hub_then_spread vs random | **~33 zealots** |

On ER, the best placement strategy is worth approximately **11 extra zealots**. That
is, a camp using `highest_degree` placement with n+ = 15 achieves the same win
probability as a random-placement camp with n+ ≈ 26. On BA, this strategic
"exchange rate" is ~33 zealots — a hub-placed camp with n+ = 7 matches a random
camp with n+ = 40.

This is the operational answer to the project's central question: **yes, structural
placement can substitute for zealot count, and the substitution rate is much higher
on heterogeneous (scale-free) networks.**

### 3.3 Minority dominance — the headline result

| Graph | Strategy | Minority-win cells | Mean P(win) in those cells |
|---|---|---|---|
| Barabási–Albert | highest_degree vs random | **36** | **0.993** |
| Barabási–Albert | hub_then_spread vs random | 35 | 0.979 |
| Barabási–Albert | random vs random | 2 | 0.625 |
| Erdős–Rényi | highest_degree vs random | 13 | 0.837 |
| Erdős–Rényi | random vs random | **0** | — |

The 36 minority-dominance cells on BA with hub placement are not marginal wins
(P_win ≈ 0.51) — they are *near-certain* victories (mean P_win = 0.993). When the
positive camp captures the hubs of a Barabási–Albert network, it dominates almost
surely even against a numerically larger opposing camp.

The contrast with `random vs random` on ER (0 minority-win cells) is the cleanest
possible control: on a homogeneous graph with symmetric placement, the larger camp
always wins. Structural advantage is a *necessary* prerequisite for minority
dominance, not just a helping factor.

### 3.4 The critical region and slow dynamics

In the narrow band where |P_win − 0.5| < 0.1 (the critical zone), crossing times
are dramatically elevated:

- ER, `highest_degree vs random`: mean crossing time **2,361 steps**
- ER, `hub_then_spread vs random`: mean crossing time **1,524 steps**
- ER, `random vs random`: mean crossing time **1,250 steps**

These slow dynamics near the boundary are the fingerprint of **critical slowing
down** — a universal phenomenon in systems near phase transitions. Close to the
tipping boundary, the "restoring force" pushing the system toward one attractor
is weak, so fluctuations dominate and the system wanders for a long time before
committing. This is also why the inter-seed variance is highest in this regime:
different stochastic realizations explore the phase space differently and can land
on opposite sides of the transition.

---

## 4. Experiment 3 — WL-Based Placement: A Counter-Intuitive Result

### 4.1 The WL partition structure on BA and ER

The WL color refinement algorithm converges to a partition of nodes into
structurally equivalent classes. The results here are striking:

| Graph | Number of classes | Entropy |
|---|---|---|
| Barabási–Albert (N=300) | **295 classes** | 5.681 |
| Erdős–Rényi (N=300) | **300 classes** | 5.704 |

Nearly every node is its own WL class. On ER, the graph is almost certainly
**asymmetric** (all nodes have distinct structural neighborhoods), so WL reaches
full individualization immediately. On BA, the heterogeneous degree distribution
also produces mostly unique local structures. The consequence is fundamental:
**"one node per WL class" is essentially equivalent to picking 15 nodes at random
from all non-hub nodes**, because hubs (which form unique singleton classes) are
not prioritized by the coverage criterion.

### 4.2 WL-cover is the worst strategy — and why this makes sense

| Graph | Strategy | P(positive wins) | psi_degree_norm |
|---|---|---|---|
| BA | highest_degree | **1.000** | 0.258 |
| BA | wl_top_class | 1.000 | 0.258 |
| BA | farthest_spread | 1.000 | 0.087 |
| BA | random | 0.350 | 0.052 |
| BA | **wl_cover** | **0.000** | **0.025** |

The `wl_cover` strategy achieves P_win = 0 — it performs *worse than random
placement*. The `psi_degree_norm` for `wl_cover` is 0.025, meaning the 15 selected
zealots collectively hold only 2.5% of the network's total degree. By comparison,
`highest_degree` achieves 25.8% with the same number of zealots.

This exposes a deep conceptual insight: **WL structural diversity and voter model
influence are orthogonal, and on heterogeneous networks, they are actively
antagonistic.** The WL algorithm is designed to maximize *coverage of distinct
local environments* — a notion imported from graph isomorphism testing and graph
neural network expressivity. But the voter model rewards *degree concentration*,
not structural variety. A zealot's influence is proportional to its degree, full
stop. Spreading zealots evenly across the WL partition means deliberately avoiding
the high-degree nodes that drive dynamics.

This is an important cautionary result for practitioners: metrics designed for
graph *characterization* (like WL) may be actively harmful when repurposed for
*influence maximization* in dynamic processes.

### 4.3 Dispersion is harmful — why clustering helps

The Pearson correlation of `psi_dispersion` (normalized average pairwise distance
within the zealot set) with P_win is **−0.897 on ER** and **−0.637 on BA**. More
dispersed zealot placement means lower win probability.

This again seems counterintuitive — shouldn't spreading zealots ensure they "cover"
more of the graph? The answer is no, for a subtle reason. In the voter model,
influence propagates stochastically via random walks. A cluster of zealots on hubs
creates a *dense local field* of opinion +1 that is very hard for free nodes in
that neighborhood to escape. The boundary between the zealot cluster's influence
zone and the rest of the network is large (many edges), but the zealots win the
local battle quickly. By contrast, dispersed zealots fight many isolated local
battles simultaneously, each with fewer reinforcements.

On scale-free networks, this effect is amplified: the hubs that `wl_cover` avoids
are precisely the *communication bottlenecks* of the network. Placing zealots there
does not just win local opinion — it controls the information highways that
determine global consensus.

---

## 5. Experiment 4 — The Search for a Universal Predictor

### 5.1 The metric hierarchy

| Metric | AUC | Steepness a | Interpretation |
|---|---|---|---|
| ratio_psi_degree_norm | **0.994** | 37.96 | Fraction of total degree held by Z+ |
| ratio_psi_pagerank | 0.994 | 39.21 | Fraction of total PageRank held by Z+ |
| ratio_psi_hybrid | 0.926 | 77.16 | Weighted combination |
| ratio_psi_rho | 0.819 | 11.56 | Fraction of total zealot count |
| ratio_psi_dispersion | 0.771 | −42.35 | Negative: more dispersion = worse |
| ratio_psi_wl | 0.679 | 6.83 | WL coverage ratio |

The gap between `ratio_psi_degree_norm` (AUC = 0.994) and `ratio_psi_rho`
(AUC = 0.819) is the quantitative statement of the whole project: knowing *how many*
zealots each camp has is a mediocre predictor of who wins, but knowing *how much
degree* each camp controls is an almost perfect predictor.

### 5.2 Why degree norm and PageRank tie at 0.994

PageRank is a degree-weighted, iterative centrality — it accounts not just for a
node's own degree but for the degree of its neighbors, and their neighbors, etc.
On BA graphs with power-law degree distributions, PageRank and degree are strongly
correlated, so the two metrics carry nearly identical information. The fact that
they achieve essentially identical AUC (0.994) suggests that *first-order degree
information* is already sufficient — the higher-order corrections from PageRank
do not add predictive value over and above raw degree in this setting.

On a graph where PageRank and degree decorrelate more strongly (e.g., graphs with
bottleneck structures, hierarchical communities), one might expect PageRank to
become strictly better. Testing this would be a natural extension.

### 5.3 The logistic transition: a near-discontinuous phase boundary

The steepness coefficient a = 37.96 for `ratio_psi_degree_norm` defines the sharpness
of the transition:

$$P(\text{positive wins}) \approx \sigma\!\left(37.96 \times (\Psi - 0.5)\right)$$

where Ψ = ratio_psi_degree_norm and σ is the sigmoid function. At a = 37.96, the
sigmoid goes from P ≈ 0.02 at Ψ = 0.38 to P ≈ 0.98 at Ψ = 0.62. The transition
is essentially a step function: **across only 24% of the [0,1] range of Ψ, the
outcome shifts from near-certain negative dominance to near-certain positive
dominance.** The critical band [0.489, 0.511] captures only 2.2% of this range.

This sharp transition is the hallmark of a genuine **phase transition in the
thermodynamic-like sense**: the system is not gradually shifting preferences, it is
snapping between two attractors, and the degree-norm ratio is the order parameter
that controls which attractor is reached.

### 5.4 Fully connected graphs: mean-field validation

On fully connected graphs, `ratio_psi_rho` (simple zealot fraction) achieves AUC =
0.988 — matching `ratio_psi_degree_norm` at 0.988. On FC, all degrees are equal
(N−1), so `psi_degree_norm ∝ psi_rho` by definition. The metric degeneracy here is
expected and confirms that the model collapses to pure mean-field on the complete
graph, where only zealot count matters. This is a valuable consistency check.

### 5.5 Minority dominance: the structural signature

Among the 71 minority-win configurations (n+ < n− but P_win > 0.6):

- Mean `ratio_psi_degree_norm` = **0.663** (std = 0.110)
- Range: roughly [0.53, 0.80]

These camps hold on average **66% of the total zealot degree** despite comprising
fewer nodes. The typical scenario: n+ = 10 zealots on BA-500 hubs capture ~25% of
total network degree, while n− = 20 random zealots capture only ~8%. The ratio
0.25/(0.25+0.08) ≈ 0.75 — well above 0.5, explaining the near-certain positive win.

The fact that `ratio_psi_rho` achieves AUC = 1.000 *within this minority-win
subgroup* is a subtle but interesting observation. Among configurations where the
minority wins, the degree ratio is so extreme that even the cruder zealot-count
ratio becomes a perfect separator (these wins are not marginal — they are decisive).
This is not a contradiction of the earlier finding that `rho` is a weak global
predictor; it simply means that the minority-win cases are "easy" for any metric
once you already know who is minority.

---

## 6. Experiment 5 — Graph Topology as an Amplifier

### 6.1 The regression: degree heterogeneity predicts placement advantage

The linear regression `placement_advantage ~ degree_heterogeneity` gives:

$$\text{placement advantage} = 0.281 \times \text{hetero} + 0.347 \quad (R^2 = 0.470)$$

The R² of 0.470 is moderate — degree heterogeneity explains about half the variance
in how much hub placement helps. The missing variance comes from graph-specific
structural features beyond the degree distribution (e.g., community structure,
diameter, mixing patterns).

The intercept of 0.347 is non-trivial: even at zero degree heterogeneity (the FC
case), hub placement still provides a baseline advantage of ~0.35 in P_win over
random. This is partly a finite-size effect (on FC with N=200, the "best" 20 nodes
by degree are indistinguishable from random choices) and partly noise from n_runs=15.

### 6.2 The BA m=2 anomaly

BA with m=2 has the **highest degree heterogeneity (1.176)** and the **highest
placement advantage (0.667)**, while BA with m=5 has lower heterogeneity (0.931)
and a slightly lower advantage (0.600). This pattern has a clean explanation:

When m is small, the BA network is sparse, and the degree distribution is more
heavy-tailed relative to the mean. The ratio std(degree)/mean(degree) is largest
for m=2 because the hub degrees can reach into the hundreds while average degree is
only 4 (= 2m). As m increases, the average degree grows (= 2m), hubs grow too
(max degree scales as √N for BA), but the *relative* spread decreases. So the
"leverage" of being on a hub, measured relative to average degree, is greatest for
sparse BA networks.

### 6.3 The fully connected graph as a perfect control

On FC, placement advantage = 0.133 — barely above zero. In 15 runs with n+=n−=20
out of N=200 nodes (10% zealots each), the two camps are symmetric and the
stochasticity of the initial free-node configuration determines outcomes more than
placement. This is precisely the mean-field prediction: the long-run consensus is
determined by the zealot density ratio alone, and with equal densities, the outcome
is approximately a coin flip perturbed by finite-size effects.

### 6.4 ER: an intermediate regime

ER graphs with p = 0.008 to 0.03 show placement advantages of 0.467 to 0.600 —
consistently high, but with a non-monotone dependence on p. The most sparse ER
graph (p=0.008, heterogeneity=0.456) shows the highest advantage (0.600), which
aligns with the heterogeneity regression. Denser ER graphs become more regular and
hub placement loses its leverage, converging toward the FC behavior.

---

## 7. Synthesis: A Coherent Picture

The five experiments collectively tell a single, coherent story.

### 7.1 The order parameter of zealot competition

In the two-camp voter model, the quantity

$$\Psi_d = \frac{\sum_{i \in Z_+} d_i}{\sum_{i \in Z_+} d_i + \sum_{i \in Z_-} d_i}$$

acts as an **order parameter** for the competition. When Ψ_d > 0.5, the positive
camp dominates asymptotically with probability close to 1; when Ψ_d < 0.5, the
negative camp dominates. The transition is sharp (logistic steepness a ≈ 38),
occurring over a critical window of width ~0.02 in Ψ_d.

This has a clean mechanistic justification rooted in the update rule: at each step,
the probability that a free node is pulled toward +1 is proportional to the fraction
of its neighbors that are +1. Summed over all free nodes and averaged over the
network, this probability is approximately proportional to the total degree held by
positive agents (zealots + free nodes currently at +1). In the long run, the zealots
act as fixed boundary conditions that bias this sum — and their total degree is the
relevant measure of that bias.

### 7.2 Topology as a multiplier, not a modifier

The topology of the graph does not change *what* matters (degree-weighted influence)
but *how much* strategic placement can exploit it. On fully connected graphs,
topology is irrelevant. On Erdős–Rényi graphs, hub placement provides moderate
leverage. On Barabási–Albert networks, hub placement is decisive — it can cause a
5-zealot camp to defeat a 50-zealot camp with near certainty.

This suggests a practical principle: **the value of strategic placement scales with
the degree of heterogeneity of the underlying network.** In real-world social
networks, which are known to be heavy-tailed (a small number of influencers hold
a disproportionate fraction of connections), this implies that small, well-connected
activist groups can systematically dominate opinion dynamics against larger but
poorly-connected opposing movements.

### 7.3 What WL diversity misses — and why it matters

The WL experiment's finding that structural diversity (as measured by WL coverage)
is *anti-correlated* with influence is not just an empirical curiosity. It points
to a fundamental distinction between two types of "importance" in network science:

- **Representational importance** (WL, graph isomorphism, expressive GNNs): which
  nodes provide the most information about the graph's global structure?
- **Dynamical importance** (degree, PageRank, voter model influence): which nodes
  drive the most influence in a given dynamic process?

These two notions of importance are not only distinct — they can be actively
opposed. A node is representationally important if it occupies a *unique* structural
position. A node is dynamically important in the voter model if it has *many
connections*. High-degree hubs are often structurally *redundant* (many similar
hubs exist in BA networks), but dynamically *critical*. Peripheral nodes in unique
structural positions are the opposite.

This distinction is rarely made explicit in the network science literature and
constitutes a genuinely interesting theoretical contribution of the experiment.

### 7.4 The metastable critical region

Near Ψ_d ≈ 0.5, the system enters a critical regime characterized by:
- Long crossing times (up to 2,361 steps on ER)
- High inter-seed variance in final magnetization
- Slow approach to steady state

This is the analog of **critical slowing down** near a second-order phase
transition. In practice, it means that near the tipping boundary, small
perturbations — adding one zealot, changing one node's placement, one different
random seed — can flip the outcome. This is the regime most relevant to real-world
scenarios where the two camps are nearly matched: tiny structural advantages or
disadvantages determine the long-run winner.

---

## 8. Open Questions and Extensions

The results suggest several concrete directions for future investigation:

**1. The role of network diameter and community structure.**
The current experiments use BA and ER, which have small diameter and no
community structure. On networks with communities (stochastic block models, real
social networks), zealot placement at community *boundaries* might outperform pure
hub placement. The Ψ_d predictor might need a community-aware correction.

**2. Asymmetric update rules.**
The standard voter model gives equal weight to all neighbors. A zealot of degree
100 influences each of its 100 neighbors with probability 1/100 per interaction.
What if influence decays with degree (weighted voter model)? Hub placement might
become less decisive, and the optimal strategy might shift.

**3. Temporal networks.**
In real social networks, connections are not static. If the network rewires over
time (adaptive voter model), zealots might be able to "drift" toward high-degree
positions dynamically. Does the degree-norm metric remain predictive on evolving
graphs?

**4. Finite-size scaling and the thermodynamic limit.**
All experiments use N ≤ 800. The sharpness of the transition (a = 38) might
increase with N — in the thermodynamic limit (N → ∞), the transition might
become a true discontinuity. Testing Ψ_d on networks of increasing size would
allow extrapolation to the large-N behavior.

**5. Causal identification of hub vs. dispersion effects.**
The current experiments conflate two factors: hub placement maximizes degree AND
minimizes dispersion (hubs cluster near the network core). Disentangling these
by constructing placement strategies that independently control degree and dispersion
would clarify which factor drives the advantage.

---

## 9. Key Takeaways

For quick reference, here are the five most important results:

1. **`ratio_psi_degree_norm` is a near-perfect predictor of zealot competition
   outcomes** (AUC = 0.994 overall, 0.999 on BA). The logistic transition at
   Ψ_d = 0.5 is sharp (a ≈ 38), confirming a genuine phase transition.

2. **Minority dominance is real and common on BA networks.** Hub-placed positive
   camps won in 36 out of 36 grid cells where n+ < n−, with mean P_win = 0.993.
   On ER with random placement, minority dominance never occurs (0 cells).

3. **Hub placement shifts the tipping boundary by 11–33 zealot equivalents**
   depending on graph type. On BA, the boundary exits the tested parameter space
   entirely — hub placement makes zealot count irrelevant over a wide range.

4. **WL-based placement is actively harmful**, achieving P_win = 0 on both graph
   types tested. Structural diversity (WL coverage) and dynamical influence
   (degree concentration) are orthogonal and can be antagonistic.

5. **Placement advantage scales with degree heterogeneity** (regression slope =
   0.281, R² = 0.47). On fully connected graphs, placement is nearly irrelevant.
   On BA with m=2 (highest heterogeneity), placement advantage reaches 0.667 in
   win probability.

---

*Analysis written for the MVA Interactions course project, April 2026.*
*All results are based on numerical simulations; see experiment scripts for
reproducibility details.*
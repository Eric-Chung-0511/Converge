# Converge — User Handbook

**Converge** is a Monte Carlo schedule risk analysis (SRA) engine built for CPM project networks. It goes beyond a standard "run simulations and draw a histogram" tool by quantifying three effects that most planning software ignores: **merge bias**, **true criticality**, and **correlation via shared risk drivers**. This handbook explains how to use every feature and what each number means.

---

## Table of Contents

1. [Quick Start (5 minutes)](#1-quick-start-5-minutes)
2. [Sidebar Controls](#2-sidebar-controls)
3. [Tab 1 — Data Input](#3-tab-1--data-input)
4. [Tab 2 — Results: S-Curve & Histogram](#4-tab-2--results-s-curve--histogram)
5. [Tab 3 — Criticality & Sensitivity](#5-tab-3--criticality--sensitivity)
6. [Tab 4 — Merge Bias](#6-tab-4--merge-bias)
7. [Tab 5 — Risk Drivers & Correlation](#7-tab-5--risk-drivers--correlation)
8. [Tab 6 — Validation Bench](#8-tab-6--validation-bench)
9. [Tab 7 — Distribution Comparison](#9-tab-7--distribution-comparison)
10. [Import & Export](#10-import--export)
11. [Key Concepts Explained](#11-key-concepts-explained)
12. [Glossary](#12-glossary)

---

## 1. Quick Start (5 minutes)

1. **Load an example** — In the left sidebar, select **"DC Commissioning (MEP → IST)"** from the example dropdown and click **Load Example**. This loads a synthetic but representative data-centre commissioning network with ~14 activities and pre-configured risk drivers. The durations are illustrative, not from a real project; the network is deliberately structured to expose merge bias at the L5 IST node.

2. **Run the simulation** — Click the blue **Run Simulation** button at the bottom of the sidebar. The engine runs 10,000 Monte Carlo iterations in a few seconds.

3. **Read the S-Curve** — Go to the **Results** tab. The S-curve shows the probability distribution of your project's completion date. The P50 line is where there is a 50 % chance of finishing on or before that date.

4. **Check the Criticality tab** — See which activities actually drive schedule risk versus which ones the deterministic CPM labels as critical. These are often different.

5. **Explore Merge Bias** — Go to the **Merge Bias** tab. This shows why your CPM date is almost certainly optimistic — even if every activity finishes "on time" on average.

---

## 2. Sidebar Controls

### Load Example
Select a pre-built network from the dropdown and click **Load Example**.

| Example | What it demonstrates |
|---|---|
| DC Commissioning (MEP → IST) | Merge bias from parallel subsystem paths; headline portfolio piece |
| Hulett (1996) Case 1 | Single-path baseline; validates against published P(CPM) ≈ 10–15 % |
| Hulett (1996) Case 2 | Two parallel paths; validates merge-bias shift (CPM date drops to < 5 % likely) |
| Hulett (1996) Case 3 | Criticality ≠ CPM critical path; Path B has ~69 % criticality despite 5 d float |
| Teaching Toy | Minimal 2-activity network for analytic agreement tests |

### Simulation Controls

| Control | What it does | Guidance |
|---|---|---|
| **Iterations (N)** | Number of Monte Carlo runs | 10,000 is the default and sufficient for most uses. Increase to 50,000–100,000 for smoother convergence plots or tighter percentile estimates. |
| **Sampling method** | Pure Random vs Latin Hypercube (LHS) | LHS gives better coverage of the input space with fewer iterations — recommended for networks with many risk drivers. |
| **Random seed** | Fixes the random number generator for reproducibility | Change the seed to verify that your results are stable (they should be, at N ≥ 10,000). |
| **Percentiles to display** | Which probability levels to mark on charts | Default: P50, P80, P90. You can add P10, P25, P75, P85, P95. |

### Calendar Dates (optional)
If you enter a **project start date**, all working-day results are converted to calendar dates for display. You may also enter a list of public holidays (one per line, YYYY-MM-DD format). This conversion is for presentation only — the engine always works in working days internally, so correctness does not depend on any holiday calendar.

---

## 3. Tab 1 — Data Input

### Activity Register

Each row in the table represents one CPM activity. The columns are:

| Column | Description |
|---|---|
| **ID** | A short unique identifier, e.g. `MEP`, `L3_COOL`. Used in predecessor lists. |
| **Name** | A descriptive label shown in charts. |
| **Predecessors** | Semicolon-separated IDs of activities that must finish before this one starts (Finish-to-Start logic only). Leave blank for the first activity. Example: `L3_COOL;L3_PWR` |
| **Distribution** | The probability distribution for this activity's duration. Options: `triangular`, `pert`, `uniform`, `normal`, `deterministic`. |
| **a / min / mean** | Lower parameter: minimum duration for triangular/PERT/uniform; mean for normal. |
| **m / likely / std** | Middle parameter: most likely (mode) for triangular/PERT; standard deviation for normal. Not used for uniform. |
| **b / max** | Upper parameter: maximum duration for triangular/PERT/uniform. Not used for normal. |

**Tips:**
- All durations are in **working days**.
- The network must be a directed acyclic graph (no cycles). Converge will detect and report cycles before running.
- You do not need to define a single "project start" activity — any activity with no predecessors is treated as starting at day 0.
- **Constraint dates are intentionally ignored.** Enforcing a hard constraint date during simulation would force the model to show a false 100 % on-time rate, defeating the purpose of the analysis.

### Risk Driver Register

Each row is a risk driver — a named uncertainty that can affect multiple activities simultaneously. This is what creates realistic correlation between activities.

| Column | Description |
|---|---|
| **ID** | Short unique identifier, e.g. `RD_SUPPLY`. |
| **Name** | Descriptive label, e.g. "Equipment delivery delay". |
| **Probability** | The chance this risk fires on any given iteration (0 to 1). E.g. 0.40 = 40 % chance. |
| **impact_type** | Distribution type for the impact multiplier. Usually `triangular`. |
| **impact_a** | Minimum impact multiplier. 1.0 = no change. |
| **impact_m** | Most likely multiplier. E.g. 1.15 = +15 % duration. |
| **impact_b** | Maximum multiplier. E.g. 1.40 = +40 % duration. |
| **assigned_activities** | Semicolon-separated activity IDs that this driver affects. |

**How it works:** On each iteration, the engine draws a random Bernoulli outcome for each driver. If the driver fires, one multiplier is sampled from its impact distribution and applied to *all* assigned activities simultaneously. This means activities sharing a driver move together in that iteration — producing correlation as an emergent outcome, not a manually entered coefficient.

---

## 4. Tab 2 — Results: S-Curve & Histogram

### S-Curve (CDF)

The S-curve is the cumulative distribution function (CDF) of project completion. The x-axis is project completion in working days; the y-axis is the probability of completing on or before that day.

**How to read it:**
- Find any day on the x-axis and read up to the curve → that is the probability of finishing by that day.
- The **P50 line** (dashed blue) is the median: 50 % of simulated outcomes finish on or before this date.
- The **P80 line** (dashed amber) means there is an 80 % chance of finishing by that date — a common owner / contract target.
- The **P90 line** (dashed red) is a high-confidence estimate used for contingency planning.
- The **CPM line** (dashed grey) is the deterministic schedule completion. Notice it nearly always falls well to the left of P50 — this gap is the merge bias and distribution-skew effect.

**What the "CPM: Xd (Y% likely)" label means:** The CPM completion date has only a Y% probability of being met. If Y is 10–30 %, the schedule is highly optimistic and the owner should expect a late finish unless contingency is added.

### Completion Histogram (PDF)

The histogram shows the same data as the S-curve but in a different form: the probability *density* of each possible completion day. The shape of this histogram tells you about the risk profile:

- **Narrow and symmetric** → uncertainty is well-bounded and roughly symmetric.
- **Wide and right-skewed** (long tail to the right) → there is significant upside risk; bad outcomes are much worse than the mode suggests.
- **Bimodal (two humps)** → two distinct scenarios are driving outcomes, often caused by a high-probability risk driver that creates two clusters.

The **P50, P80, P90 markers** on the histogram correspond exactly to the same lines on the S-curve.

---

## 5. Tab 3 — Criticality & Sensitivity

### What is Criticality Index?

The **Criticality Index (CI)** of an activity is the fraction of Monte Carlo iterations in which that activity lies on the *realised longest path* through the network. It answers the question: "In what percentage of futures is this activity actually driving the project finish date?"

This is fundamentally different from the deterministic CPM critical path, which is computed once using point-estimate durations.

**Why they differ:**
- CPM uses mean (or mode) durations to find *one* critical path.
- In reality, durations vary. A path with 5 days of float at the mode can still become the longest path in 40 % of simulated futures if its activities have high uncertainty.
- **The activity with the highest criticality index — not the CPM critical path activity — is the one most worth monitoring and accelerating.**

### Criticality Index Chart

The bar chart shows all activities sorted by criticality index (highest first). Bars are coloured:
- **Red** = on the deterministic CPM critical path
- **Blue** = not on the CPM critical path

Look for **blue bars that are tall** — these are near-critical activities that deterministic CPM overlooks but that are frequently the real driver of delay.

### Criticality & Sensitivity Table

The table shows, for each activity:

| Column | Meaning |
|---|---|
| **Criticality Index** | % of iterations where this activity is on the longest path |
| **Spearman Correlation** | Rank correlation between this activity's sampled duration and project completion. A high value means when this activity runs long, the whole project runs long. |
| **CPM Critical** | Whether this activity is on the deterministic critical path |
| **CPM Duration** | The deterministic (point-estimate) duration |
| **Simulated Mean** | The average sampled duration across all iterations (typically > CPM duration due to distribution skew) |

### Sensitivity Tornado

The tornado chart ranks activities by their Spearman rank correlation with total project completion. Activities at the top of the chart are the strongest drivers of schedule risk — focusing mitigation effort on these activities gives the highest return.

**Interpreting the values:**
- Correlation near **+1.0** → when this activity takes longer, the project always finishes later. It is the dominant driver.
- Correlation near **0** → this activity has little influence on total completion, regardless of how long it takes.

---

## 6. Tab 4 — Merge Bias

### The Core Concept

**Merge bias** is the systematic optimism in CPM schedules caused by parallel paths converging at a merge node. It is not a modelling error — it is a mathematical certainty proven by Jensen's inequality.

When multiple parallel paths must all complete before an activity can start (a merge point), the actual start of that activity equals the *maximum* of all path completions:

> **E[max(path₁, path₂, …)] ≥ max(E[path₁], E[path₂], …)**

CPM uses `max(E[path])`. Reality is `E[max(path)]`. The gap between them is merge bias.

In a data-centre commissioning network, L5 Integrated Systems Testing (IST) cannot start until *every* parallel subsystem path (power, cooling, generators, BMS) completes. Even if each path is "on time" on average, IST will start late in most scenarios because it only takes one slow path to delay everything.

### The Metrics Explained

| Metric | What it means |
|---|---|
| **CPM (mode-based)** | The deterministic schedule completion using point-estimate (modal) durations. This is what your Gantt chart shows. |
| **T(E[d]) mean-based** | Schedule completion recomputed using *mean* durations instead of mode durations. For right-skewed distributions (PERT, triangular with high tail), this is already later than the CPM date. |
| **Simulated Mean E[T]** | The average project completion across all Monte Carlo iterations. This is the true expected finish date. It accounts for both distribution skew and merge bias. |
| **True Merge Bias** | `E[T] − T(E[d])`: the extra delay caused purely by the merge-point effect (Jensen gap). This isolates the structural network effect from the distribution skew effect. |
| **P50 Completion** | The median simulated completion date (50th percentile). |

**Decomposition of the CPM gap:**

> CPM date → **(+skew/risk shift)** → T(E[d]) → **(+merge bias)** → E[T]

The total gap between CPM and the true mean is split into:
1. **Skew/risk shift** — because activity distributions have right-skewed tails (the mean > mode for PERT and triangular).
2. **Merge bias** — because parallel paths converge and the maximum of random variables exceeds the maximum of their means.

### CPM vs Simulation Distribution Chart

This histogram overlays four reference lines:
- **CPM (mode): Xd** (grey dashed) — the deterministic schedule date.
- **T(E[d]): Xd** (green dash-dot) — mean-based CPM; already later than modal CPM.
- **E[T]: Xd** (red solid) — the true simulated mean; the best single estimate of when the project will finish.
- **P50: Xd** (amber dotted) — the median outcome.

The further right E[T] is from CPM, the more the schedule is understating expected duration.

---

## 7. Tab 5 — Risk Drivers & Correlation

### Implied Correlation Matrix

The heatmap shows the **Pearson correlation coefficients** between all pairs of activities, computed directly from the simulated duration samples. You did not enter these correlations manually — they are an emergent output of the risk driver structure you defined.

**How to read the heatmap:**
- Values range from −1 (perfect negative correlation) to +1 (perfect positive correlation).
- **+1.0 (dark red)** → two activities are driven by exactly the same risk driver with no independent uncertainty; they always move together.
- **~0 (white)** → two activities have no shared risk drivers; their durations are statistically independent.
- Values between 0 and 1 reflect partial sharing — activities share some drivers but also have independent uncertainty.

**The key insight:** A single shared risk driver makes two activities perfectly correlated. Additional independent risk drivers dilute that correlation. This is why a hand-entered global correlation coefficient fails — it applies the same value to all pairs indiscriminately, with no connection to the underlying risk structure.

### Correlation Matrix (Table)

The same data in tabular form, useful for exporting or reporting.

---

## 8. Tab 6 — Validation Bench

The Validation Bench is what makes Converge's results credible. It proves the engine is mathematically correct against three independent reference types.

### Layer 1 — Analytic Agreement

**What is tested:** A single Triangular(a, m, b) activity is simulated. The simulated mean and variance are compared against the closed-form analytic solutions:

- Mean = (a + m + b) / 3
- Variance = (a² + m² + b² − am − ab − mb) / 18

**Pass criterion:** Simulated values must match analytic values within a tight tolerance (< 0.5 % relative error at N = 10,000).

**Why it matters:** If the sampling engine cannot reproduce a known analytic result, nothing else can be trusted. This is the foundational correctness test.

### Layer 2 — Merge Bias vs Numerical Integration

**What is tested:** Two parallel paths merge at a node. The simulated E[max(path₁, path₂)] is compared against a high-resolution numerical integration of the true expected maximum.

**Pass criterion:** Simulated merge bias must match the numerical reference within tolerance.

**Why it matters:** Proves that the network topology and longest-path algorithm are correctly computing the merge effect — not just that the sampling engine is right.

### Layer 3 — Convergence Rate (σ/√N)

**What is tested:** The Monte Carlo standard error is computed at increasing values of N (1,000 to 100,000). The log-log plot of SE vs N should show a slope of approximately −0.5, corresponding to the theoretical `σ/√N` convergence rate.

**Pass criterion:** The fitted slope on the log-log plot must be between −0.45 and −0.55.

**Why it matters:** This is the strongest correctness claim. It proves the engine is not just getting lucky — it converges at the exact rate predicted by the Central Limit Theorem. A slope significantly steeper than −0.5 would indicate correlation artefacts; shallower would indicate inefficiency.

### Published Benchmark: Hulett (1996)

**Source:** David T. Hulett, Ph.D., *"Schedule Risk Analysis Simplified"*, PMI / PM Network, July 1996.

The three Hulett cases serve as independently verifiable benchmarks. Any reviewer can retrieve the source article and check that Converge reproduces the published results:

| Case | Network | Published target to reproduce |
|---|---|---|
| Case 1 | Single path, 2 activities. Activity A101: Triangular(40, 50, 100) | CPM date is only ~10–15 % likely to be met |
| Case 2 | Two identical parallel paths merging (4 activities) | CPM date drops to < 5 % likely; mean shifts ~10 working days later than Case 1 |
| Case 3 | B101 shortened so Path A is CPM critical (130 d) but Path B has 125 d (5 d float) | Path B has ~69 % criticality; Path A has ~31 % — opposite of what CPM implies |

---

## 9. Tab 7 — Distribution Comparison

### PERT vs Triangular — Tail Risk

This tab lets you enter a three-point estimate (minimum, most likely, maximum) and compare the Triangular and PERT distributions side by side.

**Why this matters:**

PERT (Beta-PERT) assigns a weight of 4 to the most likely value, which holds the standard deviation **close to** `(b − a) / 6` regardless of how skewed the distribution is. For a highly asymmetric activity (e.g., min = 10 d, most likely = 12 d, max = 60 d), PERT significantly **understates tail risk** compared to Triangular.

The chart shows:
- The full probability density of each distribution
- Mean and standard deviation of each
- The difference in tail shape — particularly above the P80 region

**Practical implication:** If you use PERT for highly asymmetric activities, your P80 and P90 estimates will be too optimistic. The Distribution Comparison tab quantifies exactly how much. Triangular is more conservative and arguably more honest for activities with a hard minimum but no firm upper bound.

---

## 10. Import & Export

### Importing Data

**Activities (CSV or Excel):**
1. Go to the **Data** tab.
2. Under the Activity Register, click **Import Activities (CSV or Excel)**.
3. Upload a `.csv` or `.xlsx` file. Required columns: `id`, `name`, `predecessors`, `dist_type`, `param_a`, `param_m`, `param_b`.

**Risk Drivers (CSV or Excel):**
1. Under the Risk Driver Register, click **Import Risk Drivers**.
2. Required columns: `id`, `name`, `probability`, `impact_type`, `impact_a`, `impact_m`, `impact_b`, `assigned_activities`.

**Full project (JSON):**
In the sidebar under **Import / Export**, use **Load Project (JSON)** to restore a previously saved project including all activities, risk drivers, and settings.

### Exporting Data

**Activities / Risk Drivers:**
- Use the **Export Activities (CSV)** / **Export Activities (Excel)** buttons to download the current activity register.
- Same buttons are available for the risk driver register.

**Charts:**
- Every chart has an **"Export this chart"** expander below it.
- **Interactive HTML** is always available and recommended for sharing — it preserves zoom, pan, and hover tooltips.
- **Static PNG / SVG / PDF** are available via toggle. Note: static export requires kaleido and may not be available in all environments.

**Full project (JSON):**
Use **Save Project (JSON)** in the sidebar to save all activities, risk drivers, and settings in a single file. This is the recommended way to save and resume your work.

---

## 11. Key Concepts Explained

### Monte Carlo Simulation
Monte Carlo simulation runs the schedule many thousands of times. In each iteration, a random duration is sampled for every activity from its probability distribution. The engine then computes the longest path through the network for that iteration. After N iterations, the distribution of outcomes gives a statistically reliable picture of schedule risk. At N = 10,000, percentile estimates are stable to within a fraction of a working day.

### Why CPM Completion Dates Are Optimistic
There are two independent reasons:

1. **Distribution skew:** PERT and triangular distributions have a longer right tail than left tail — the mean is higher than the mode. CPM uses the mode; simulation uses the full distribution including the tail.

2. **Merge bias (Jensen's inequality):** At a merge node, the project must wait for the slowest of all incoming paths. The expected value of that slowest path exceeds the largest individual path mean (E[max] ≥ max(E[path])) — so even when every path is "on time" on average, the merge still lands late. This effect grows with the number of parallel paths and increases with path uncertainty.

### Latin Hypercube Sampling (LHS)
Pure random sampling can leave gaps in the probability space by chance, especially at low N. LHS divides the 0–1 range into N equal strata and ensures exactly one sample per stratum per dimension. This gives more uniform coverage and achieves the same statistical precision as pure random with fewer iterations — typically 20–40 % fewer iterations to reach the same SE.

### The Risk Driver Method (Hulett)
Traditional SRA adds correlation by entering a matrix of pairwise correlation coefficients. This is problematic because: (a) coefficients must be estimated subjectively, (b) the matrix must be positive semi-definite, and (c) the same coefficient applies to all magnitude scenarios.

The Risk Driver Method models correlation structurally: two activities are correlated *because* they share a common risk (e.g., the same supplier, the same weather window, the same regulatory approval). The correlation is not an input — it emerges from the structure of the risk register. The implied correlation matrix (shown in Tab 5) is an output you can inspect and validate.

---

## 12. Glossary

| Term | Definition |
|---|---|
| **CDF** | Cumulative Distribution Function. The S-curve. At any point x, the CDF gives P(completion ≤ x). |
| **CI (Criticality Index)** | Fraction of Monte Carlo iterations in which an activity lies on the realised longest (critical) path. |
| **CPM** | Critical Path Method. Deterministic scheduling using point-estimate durations to find the longest path. |
| **E[T]** | Expected (mean) completion time across all simulated iterations. The best single estimate of when the project will finish. |
| **Jensen's inequality** | For any convex function f, E[f(X)] ≥ f(E[X]). The `max` function is convex, so E[max(paths)] ≥ max(E[paths]). This is the mathematical proof of merge bias. |
| **LHS** | Latin Hypercube Sampling. A stratified sampling method that improves coverage of the input space. |
| **Merge bias** | The extra delay at a merge node caused by having to wait for the slowest parallel path. Proven by Jensen's inequality. |
| **PERT distribution** | Beta-PERT: a smooth, bell-shaped distribution parameterised by (min, most likely, max) with a shape parameter λ = 4. The industry standard for QRA. |
| **P50 / P80 / P90** | The 50th / 80th / 90th percentile of simulated completion. P80 means there is an 80 % probability of finishing on or before that date. |
| **PDF** | Probability Density Function. The histogram shape. Shows the relative likelihood of each completion outcome. |
| **QRA** | Quantitative Risk Analysis. The discipline of assigning probability distributions to uncertain inputs and propagating them through a model to characterise output uncertainty. |
| **Risk driver** | A named uncertainty that simultaneously affects multiple activities when it fires. The structural unit of correlation in the Risk Driver Method. |
| **Seed** | An integer that initialises the random number generator. The same seed always produces the same sequence of random numbers, making results reproducible. |
| **SE (Standard Error)** | A measure of how much the simulated mean varies due to finite sample size. SE ≈ σ/√N. At N = 10,000 and σ ≈ 15 d, SE ≈ 0.15 d — negligible. |
| **Sensitivity** | Spearman rank correlation between an activity's duration and total project completion. Measures how much influence that activity has on overall schedule risk. |
| **Triangular distribution** | A simple three-point distribution (min, most likely, max) with linear density. More conservative than PERT for the same three points. |
| **Working days** | Calendar days minus weekends and any user-specified holidays. All engine calculations are in working days. |

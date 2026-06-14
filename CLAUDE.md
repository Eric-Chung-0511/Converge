# CLAUDE.md — Converge

> **Converge** — a schedule risk analysis (SRA) engine with a built-in validation bench.
> Monte Carlo simulation for project schedule risk, built to a depth that proves correctness mathematically — not a tutorial demo. Runs locally as a self-contained web app.

---

## 0. Read this first — the one principle that governs every decision

This is **not** a generic "I ran a Monte Carlo and here is a histogram" project. The entire purpose of this app is to **demonstrate depth and differentiation** to a hiring manager in data-centre / mission-critical / EPC schedule risk analysis. Every feature decision must be judged against one question:

> *Would a senior planner or a QRA hiring manager look at this and conclude the author actually understands schedule risk at a level most planners do not?*

If a feature does not advance that goal, it is out of scope. If a feature advances it but is "basic", it must be deepened until it is provably rigorous. **When in doubt, choose rigour over breadth.** We are building one thing deep, not many things shallow.

The three intellectual differentiators this app must make undeniable:

1. **Merge bias** — quantify *why* a schedule where every path is "on time" still finishes late, and prove the result against an analytical / numerical reference.
2. **Criticality index != CPM critical path** — show that the activity the deterministic CPM labels "critical" is often **not** the activity that actually drives completion risk.
3. **Correlation via the Risk Driver Method (Hulett)** — model correlation as an *emergent structural property* of shared risk drivers, not as a hand-entered correlation coefficient.

Plus the trust layer that makes all three credible:

4. **A validation bench** — the engine is proven correct against known-answer tests (analytical solutions, convergence rate, published benchmark cases with citable sources).

---

## 1. Scope

### In scope (v1)
- Monte Carlo simulation of **schedule** completion risk for a CPM activity network.
- Activity duration uncertainty via probability distributions.
- Risk Driver Method for correlation.
- Merge bias quantification with analytical/numerical validation.
- Criticality index and sensitivity (tornado).
- Latin Hypercube Sampling (LHS) as an option alongside pure random sampling.
- A validation bench (unit tests + an in-app "Validation" view).
- Import / export for daily use (see section 6).
- Chart export to PDF / SVG / PNG / interactive HTML (see section 6).
- Built-in example networks, including verifiable benchmark cases from cited literature (see section 7).
- Runs locally as a self-contained web app (see section 9).

### Explicitly OUT of scope (do not build in v1; note as future work only)
- Cost risk / integrated cost-schedule / EVM.
- Resource loading / resource levelling.
- A full interactive CPM schedule editor (we edit activities in a table, not on a Gantt canvas).
- **Primavera P6 / XER import.** XER parsing is brittle and high-effort and adds nothing to the "I understand QRA" thesis. Mention in README as a future extension only.
- Probabilistic / conditional branching of discrete events beyond what the Risk Driver occurrence probability already provides.
- Relationship types other than Finish-to-Start.

### Domain framing for the headline example
The flagship example network models the back end of a data-centre / mission-critical delivery sequence — from electrical & instrumentation/control installation (MEP) through commissioning — because this sequence is the canonical real-world merge-bias scenario. Use the standard data-centre commissioning levels:

- **L3 — Pre-Functional Testing (PFT):** each subsystem (power distribution, cooling, generators, BMS/controls) tested **independently and in parallel**.
- **L4 — Functional Performance Testing (FPT):** each discipline tested under part-load / full-load / fault scenarios.
- **L5 — Integrated Systems Testing (IST):** **all** systems tested together at full load. IST cannot start until *every* parallel subsystem path (L3 -> L4) is complete.

L5 IST is therefore a merge point of multiple parallel paths: its expected start = `E[max(path_1, ..., path_k)]`, while deterministic CPM uses `max(E[path_1], ..., E[path_k])`. Because `max` is convex, `E[max] >= max(E[...])` (Jensen's inequality). The gap is the merge bias, and it is physically why energization / RFS dates run optimistic. This is the same structure as parallel-unit power-plant commissioning.

---

## 2. Architecture (non-negotiable)

**Strict separation of engine and UI.** The simulation engine must be a standalone, pure-Python package that knows nothing about Streamlit. The UI imports the engine. This is what allows the validation bench to run headless under `pytest`, and "the engine passes its mathematical validation tests" is itself a selling point.

```
converge/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── converge/
│   ├── __init__.py
│   ├── distributions.py      # duration distributions: Triangular, PERT, Uniform, Normal
│   ├── calendar.py           # working-day <-> calendar-date conversion (optional holidays)
│   ├── network.py            # CPM network: activities, FS dependencies, forward pass / longest path
│   ├── risk_drivers.py       # Risk Driver Method: drivers, assignment, multiplicative impacts
│   ├── sampling.py           # pure random + Latin Hypercube Sampling
│   ├── engine.py             # the Monte Carlo orchestrator (vectorised)
│   ├── results.py            # post-processing: percentiles, criticality, sensitivity, merge bias
│   ├── io.py                 # import/export: CSV, Excel, JSON round-trip
│   └── examples.py           # built-in + benchmark example networks
├── app/
│   ├── streamlit_app.py      # UI only; imports converge.*
│   └── plotting.py           # Plotly figure builders + export helpers
└── tests/
    ├── test_distributions.py
    ├── test_merge_bias.py
    ├── test_convergence.py
    ├── test_risk_drivers.py
    └── test_benchmarks.py
```

The engine must be **vectorised with NumPy**: run all `N` iterations as array operations, not a Python `for` loop over iterations. Document the time complexity. A naive per-iteration loop is unacceptable for the convergence demo (which needs large `N`).

---

## 3. Core engine requirements

### 3.1 Duration distributions (`distributions.py`)
Implement, each as a class/function exposing `sample(size, rng)` and, where closed forms exist, `mean()` and `variance()`:

- **Triangular** `(a, m, b)` — analytic mean `(a+m+b)/3`, analytic variance available; use for known-answer tests.
- **PERT / Beta-PERT** `(a, m, b)` with shape parameter `lambda` (default 4) — the industry-standard QRA distribution.
- **Uniform** `(a, b)` — teaching/contrast.
- **Normal** `(mu, sigma)`, truncated at 0 — teaching/contrast; also used to demonstrate the Central Limit Theorem on long path sums.

**Depth requirement — the PERT subtlety:** PERT effectively fixes the standard deviation at roughly `(b - a) / 6` regardless of skew, which **understates tail risk** for highly skewed activities. The UI must let the user place the same activity under Triangular vs PERT and view the tail difference side by side. Document this in code comments and in the in-app math notes. This is a deliberate "I know where the assumptions bite" signal.

### 3.2 CPM network (`network.py`)
- Activities with predecessors (Finish-to-Start is sufficient for v1; note others as future work).
- Deterministic forward pass to compute the CPM completion and the deterministic critical path.
- A per-iteration completion computed as the longest path through sampled durations.
- **Constraint dates must be stripped/ignored for the risk analysis.** A risk analysis exists to test whether contractual dates are feasible; if hard date constraints were enforced each iteration, the simulation would force success and invalidate itself. The engine works on logic + durations only. State this in comments — it is a basic QRA correctness point that signals domain literacy.
- Validate the network: detect cycles, dangling activities, missing predecessors — raise clear, typed errors.

### 3.3 Working-day / calendar handling (`calendar.py`)
- The engine computes everything in **working-day units** relative to a project start. This avoids coupling correctness to a holiday calendar.
- Provide an optional, simple conversion to calendar dates: a project start date, a 5-day work week by default, and an optional user-supplied holiday list. Keep it simple; this is presentation only, not part of the math.
- All published-benchmark validation (section 7) is checked in working-day / probability terms, never against region-specific calendar dates.

### 3.4 Risk Driver Method (`risk_drivers.py`) — the headline correlation model
Implement Hulett's risk driver approach. Do **not** implement a global correlation coefficient as the primary mechanism.

A **risk driver** has:
- `probability` — the chance the risk occurs on any given iteration.
- `impact` — a **multiplicative** factor range applied to assigned activity durations *if it occurs* (e.g. modelled as a distribution over a multiplier, default range like 1.05-1.30).
- `assigned_activities` — the set of activities this driver affects.

Per iteration: for each driver, draw whether it occurs (Bernoulli on `probability`); if it occurs, draw one multiplier and apply that **same** multiplier to **all** activities the driver is assigned to. Correlation between two activities then emerges naturally: activities sharing a driver move together; activities with additional independent drivers are less correlated. The app should be able to **report the resulting implied correlation matrix** (computed from the simulated samples) so the user can see correlation as an *output* of structure, not an input.

Document in comments the key fact: a single shared driver makes two activities 100% correlated; additional independent drivers dilute that correlation. This is the rigorous answer to "a single global coefficient has no discriminating power."

### 3.5 Sampling (`sampling.py`)
- Pure random sampling.
- **Latin Hypercube Sampling** — improves coverage of the input space and convergence efficiency, especially with many correlated drivers or limited iteration budgets. Expose as a toggle so the UI can show LHS vs random convergence side by side.
- All randomness through a seeded `numpy.random.Generator` for reproducibility. Seed must be user-settable in the UI.

### 3.6 Engine orchestration (`engine.py`)
- Default iterations `N = 10000`, user-configurable. Provide guidance in the UI that more iterations smooth the output and that convergence can be inspected in the Validation view.
- **Performance target:** a 30-50 activity network at `N = 10000` should complete in well under a few seconds on a normal laptop CPU thanks to vectorisation. No GPU, no model weights, no heavy dependencies.

### 3.7 Results (`results.py`)
- Completion distribution; percentiles **P10/P50/P80/P90** (configurable).
- **Criticality index**: fraction of iterations in which each activity lies on the realised critical (longest) path. Provide a comparison against the deterministic CPM critical path.
- **Sensitivity / tornado**: rank activities and risk drivers by their correlation with total completion (duration sensitivity and/or cruciality). Document which measure is used and why.
- **Merge bias metric**: deterministic CPM completion vs simulated mean/median completion, with the gap reported in working days and as a percentage, plus a marker comparing it to the analytical/numerical reference for the benchmark cases.

---

## 4. The validation bench — this is what makes it credible (`tests/` + in-app "Validation" view)

The app must be able to **prove it is correct**. Implement three layers, all as `pytest` tests AND surfaced in an in-app Validation view with plots.

1. **Analytic agreement (single activity).** For a Triangular `(a, m, b)`, the simulated mean and variance must match the closed-form mean/variance within a tolerance. Fail the test if not. Proves the sampling engine is correct.

2. **Merge bias against a reference.** For two (then k) parallel paths with known duration distributions merging at a node, compare the simulated `E[max(...)]` against an analytical result (where available) or a high-resolution numerical integration. Show that simulated completion systematically exceeds `max(E[path])`, and that the measured gap matches the reference within tolerance.

3. **Convergence rate.** Plot Monte Carlo standard error vs number of iterations `N` and demonstrate it decays as approximately `sigma / sqrt(N)` (slope approx -1/2 on a log-log plot). This is the strongest correctness claim: the error converges at the theoretically predicted rate. Include this plot in the Validation view.

The in-app Validation view should read like a short technical report: state the claim, show computed vs reference, show pass/fail, with the math explained in plain language (Jensen's inequality for merge bias; CLT for long-path sums; `sigma/sqrt(N)` for MC error).

---

## 5. UI (Streamlit, `app/`)

- Use `st.data_editor` for the **activity table** (id, name, predecessors, distribution type, a/m/b or mu/sigma) and a separate `st.data_editor` for the **Risk Driver register** (id, name, probability, impact range, assigned activities). These two editable tables are the interactive core.
- Controls: number of iterations `N`, sampling method (random vs LHS), random seed, percentile selection, optional project start date + holidays.
- Result views (each as a clearly labelled section/tab):
  - **S-curve (CDF)** with P50/P80/P90 marked by vertical reference lines — the primary owner-communication chart.
  - **Histogram (PDF)** of completion outcomes.
  - **Tornado** sensitivity chart.
  - **Criticality index** table, shown alongside the deterministic CPM critical path so the difference is visible.
  - **Merge bias card**: CPM vs simulated completion, gap in working days and %, with the validation marker.
  - **Validation view**: the three-layer bench from section 4, with the convergence log-log plot.
  - **Implied correlation matrix** (computed from samples) to show risk-driver correlation as an emergent output.
- Include concise **in-app math notes** next to each advanced view (merge bias, criticality, risk drivers, convergence). These notes are part of the differentiation — they show the author can *communicate* the findings, which the target JD explicitly requires ("expertly interpret and communicate findings").
- UI must be clean and presentable (this is a portfolio piece). Follow good visual hierarchy; do not over-clutter. Use Plotly for all charts (interactive by default).

---

## 6. Import / export — must be convenient for daily use

- **Project save/load:** full project (activities + risk drivers + settings) as a single **JSON** file, with a documented, versioned schema. Round-trip must be lossless.
- **Tabular import/export:** the activity table and the risk-driver register must each import/export as **CSV and Excel (.xlsx)**, so a planner can prepare data in a spreadsheet and load it. Provide downloadable template files with headers and one example row. Validate on import with clear error messages pointing to the offending row/column.
- **Chart export:** every Plotly chart must be exportable to **PNG, SVG, and PDF**, plus a standalone **interactive HTML** export. Provide a one-click "export this chart" affordance. (Use Plotly's `write_image` via `kaleido` for static formats and `write_html` for interactive; add `kaleido` to requirements.)
- **Results export:** simulation results (percentiles, criticality table, merge-bias summary) exportable to CSV/Excel, and a one-click **PDF summary report** of the run is desirable.

All file handling must have comprehensive error handling: malformed files, missing columns, wrong types, empty inputs — never crash; surface a clear message.

---

## 7. Example & benchmark networks (`examples.py`) — must be traceable and verifiable

Ship the app with example networks so it is immediately usable and so its correctness is independently verifiable by anyone who reviews it. **Every benchmark must carry its source citation and URL in the example metadata and in the README, so a reviewer can open the source and check the numbers themselves. Never fabricate a reference value; only encode numbers you can trace to the cited source, and clearly flag any input that had to be reconstructed rather than read directly from the source.**

1. **Flagship — "DC Commissioning (MEP -> IST)":** an L3 -> L4 -> L5 network with several parallel subsystem paths (power distribution, cooling, generators, BMS/controls) merging at IST. This is the headline merge-bias demonstrator. 30-50 activities; choose the count that makes the merge bias clearly visible without cluttering the UI. This is an illustrative network (not a published benchmark), so label it as such.

2. **Primary verifiable benchmark — Hulett, "Schedule Risk Analysis Simplified" (PMI / PM Network, July 1996).**
   Source URL: https://www.pmi.org/learning/library/schedule-risk-analysis-simplified-10742
   Author: David T. Hulett, Ph.D. Encode the three cases and validate against the **published outputs** (in probability / working-day terms, NOT against the article's 1996 US calendar dates, which depend on a holiday calendar):
   - **Case 1 (single path, 2 activities):** Activity A101 is a Triangular with three-point estimate **(40, 50, 100)** working days — this is stated explicitly in the article (footnote 3 gives mean = (40+50+100)/3 = 63.3 days). Activity A102 has CPM duration 80 days; its low/high range is shown only in the article's Figure 2 (an image, not in the text). Do NOT invent A102's range: either obtain it from the source figure and cite that it came from Figure 2, or treat A102's range as a clearly-flagged reconstructed assumption. The **published target** to reproduce: the CPM completion date is only about **10-15% likely** to be met.
   - **Case 2 (two identical parallel paths, 4 activities; merge bias):** B-path is identical to the A-path. CPM completion is unchanged, but the published target to reproduce is: the CPM date drops to **under 5% likely**, and the **mean completion shifts later than Case 1 by roughly ten calendar days purely from adding one identical parallel path**. This is the canonical published merge-bias result.
   - **Case 3 (criticality != CPM critical path):** With B101 shortened to 45 days, Path A is the CPM critical path (130 days) while Path B is near-critical (125 days, 5 days float). The published target to reproduce: **Path B is the highest-risk path with about 69% criticality, versus about 31% for the CPM-critical Path A.**
   For each case, the in-app benchmark view must show **"computed vs published"** with the source link, so a reviewer sees Converge reproduces an authoritative reference.

3. **Teaching toy:** a minimal 2-3 activity case used by the analytic-agreement test in section 4.

If, during the build, a cleaner or more fully specified public SRA benchmark is found, it may be added alongside (not instead of) the Hulett case, with full citation and URL. The requirement is traceability, not a specific source.

---

## 8. Coding standards

- Python 3.11+. Production quality: comprehensive error handling, no bare `except`, typed custom exceptions where useful.
- **Full type hints** on all public functions; `mypy`-clean as a goal.
- Docstrings on every module, class, and public function (NumPy or Google style), stating the math where relevant.
- **All `print` statements, log messages, and f-strings must be written in English** (code-facing text is English even though project discussion is in Chinese).
- Comments should explain the *why* and the *math*, not restate the code. Where an algorithm implements a mathematical result (Jensen, CLT, `sigma/sqrt(N)`, risk-driver correlation), name the concept in a comment.
- Vectorise with NumPy; avoid Python-level loops over iterations. Use SciPy for distributions/analytic checks, Pandas for tabular I/O, Plotly for charts.
- Reproducibility: a single seeded `numpy.random.Generator` threaded through sampling.
- `requirements.txt` pinned. Core deps: `streamlit`, `numpy`, `scipy`, `pandas`, `plotly`, `kaleido`, `openpyxl`, `pytest`.
- Tests must pass (`pytest`) and the validation bench tests are part of the suite, not optional.

---

## 9. Run & distribution — local first

The primary delivery is a **self-contained app that runs locally** and a **GitHub repository that serves as the portfolio front door**. Hugging Face is explicitly NOT the primary target: the audience here is planners / QRA hiring managers / PMC firms, not the ML community, and the app uses no model and no GPU.

**Tier 1 (required) — GitHub + one-line local run.**
- `git clone`, then `pip install -r requirements.txt`, then `streamlit run app/streamlit_app.py`; the browser opens a local web app at `localhost`. All data stays on the user's machine.
- The **README is the front door**: a one-paragraph "what this is and why it is not a basic Monte Carlo demo" framing, the three differentiators, a screenshots section, the in-app math notes summarised, the benchmark citations with URLs, and a clearly labelled "Future work" section (P6/XER import, cost-schedule integration, additional relationship types, probabilistic branching).

**Tier 2 (recommended) — download-and-run, no Python setup.**
- Make it installable as a package (`pip install` from the GitHub repo) exposing a console entry point (e.g. `converge` launches the Streamlit app).
- Additionally provide a **PyInstaller** build producing a single double-click executable so a non-technical planner can run it locally with no Python environment at all. Document the build steps.

**Tier 3 (optional, future) — Docker.**
- Provide a `Dockerfile` for a one-command `docker run` with a pinned environment. List as future work; not required for v1.

**Hugging Face Spaces** may optionally be kept as a *secondary* hosted mirror for people who want a quick look without cloning — but it is not the main deliverable and the README should point primarily to the local-run instructions.

---

## 10. Build order (suggested)

1. `distributions.py` + `test_distributions.py` (analytic agreement passing first — establishes trust).
2. `network.py` (deterministic CPM + longest-path, constraint stripping) + `calendar.py` with a tiny example.
3. `sampling.py` (random first, LHS second).
4. `engine.py` vectorised Monte Carlo wiring distributions + network.
5. `results.py` (percentiles, criticality, merge bias) + `test_merge_bias.py` + `test_convergence.py`.
6. `risk_drivers.py` + `test_risk_drivers.py` + implied correlation reporting.
7. `io.py` (JSON round-trip, CSV/Excel, templates).
8. `examples.py` (flagship + cited benchmarks) + `test_benchmarks.py`.
9. `app/` Streamlit UI + `plotting.py` (charts + PDF/SVG/HTML export), with the Validation view last.
10. README + Tier 2 packaging (console entry point + PyInstaller).

At each step, prefer correctness and a passing validation bench over adding the next feature. The validation bench is the product's credibility; keep it green.

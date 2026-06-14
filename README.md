# Converge — Schedule Risk Analysis Engine

**Monte Carlo schedule risk analysis with a built-in mathematical validation bench.**
Runs locally as a self-contained web app. No cloud, no account, no GPU — your schedule data never leaves your machine.

> **Why this is not another "Monte Carlo histogram" demo.** Most schedule simulations stop at "here is a distribution of finish dates." Converge is built around the three results that actually separate quantitative risk analysis from deterministic planning — and it **proves its engine is correct** against analytical solutions, theoretical convergence rates, and a published, citable benchmark.

---

## The three differentiators

### 1. Merge bias — decomposed, not just measured

When parallel paths converge at a milestone (e.g. all commissioning subsystems must finish before Integrated Systems Testing), the expected completion is `E[max(path₁ … pathₖ)]`, while deterministic CPM effectively uses `max(E[path])`. Because `max` is convex, Jensen's inequality guarantees:

```
E[max(X₁, …, Xₖ)]  ≥  max(E[X₁], …, E[Xₖ])
```

Converge goes further than reporting "simulation is later than CPM." That naive gap conflates **two different effects**, and the app separates them:

| Component | Formula | Cause |
|---|---|---|
| **Mean shift (skew bias)** | `T(E[d]) − T_CPM(mode)` | CPM uses most-likely durations; right-skewed distributions have mean > mode |
| **True merge bias (Jensen gap)** | `E[T(d)] − T(E[d])` | Parallel-path convergence only — exactly **zero** for a serial chain |

This decomposition is falsifiable, and the test suite falsifies it: for a purely serial network the measured Jensen gap is 0.000 days (completion is linear in durations, so Jensen is tight), while adding one identical parallel path produces a +9.4 working-day gap with the CPM date unchanged — reproducing Hulett's published result (see benchmarks).

### 2. Criticality index ≠ CPM critical path

The activity that deterministic CPM calls "critical" is often **not** the activity driving completion risk. Converge computes each activity's **criticality index** (the fraction of iterations in which it lies on the realised longest path) and displays it next to the CPM critical path. In the Hulett benchmark, the path with 5 days of float carries ~69% of the risk while the "critical" path carries ~31% — a planner managing by float alone watches the wrong path.

### 3. Correlation as an output, not an input — the Risk Driver Method (Hulett)

Instead of a hand-entered correlation coefficient (which applies one number uniformly to every activity pair and has no discriminating power), Converge models **risk drivers**: named risk events with an occurrence probability and a multiplicative impact, assigned to the activities they physically affect. When a driver fires, the *same* multiplier hits all of its activities — so correlation **emerges from structure**. The app reports the implied correlation matrix computed from the simulated samples, so you can verify that correlation is a result, not an assumption.

---

## The validation bench

Credibility is the product. The engine proves itself three ways, in `pytest` (67 tests) **and** in an in-app Validation view:

1. **Analytic agreement** — simulated mean/variance of a Triangular distribution match the closed-form values within CLT tolerance.
2. **Merge bias vs numerical integration** — simulated `E[max(X₁, X₂)]` matches a `scipy.integrate.quad` reference of the survival-function integral within 0.5 working days.
3. **Convergence rate** — Monte Carlo standard error decays as `σ/√N`: the measured log-log slope is ≈ −0.5. This single test would fail if the sampling were biased, the RNG broken, or the vectorisation wrong.

### Published benchmark — Hulett (1996)

Source: Hulett, D. T., *"Schedule Risk Analysis Simplified,"* PM Network, PMI, July 1996.
<https://www.pmi.org/learning/library/schedule-risk-analysis-simplified-10742>

| Case | Published result | Converge (N = 100,000) |
|---|---|---|
| 1 — single path | CPM date only ~10–15% likely | 16.8% *(A102 range reconstructed — see note)* |
| 2 — identical parallel path added | CPM date < 5% likely; mean ~10 calendar days later than Case 1 | 2.8%; +9.4 working days, **entirely** Jensen gap |
| 3 — near-critical path | Path with 5d float carries ~69% criticality vs ~31% for CPM path | reproduced within tolerance |

*Traceability note:* A101's three-point estimate (40, 50, 100) is stated explicitly in the article. A102's range appears only in the article's Figure 2 (an image); the values used here (60, 80, 110) are a **clearly flagged reconstruction**, which is why Case 1 validates against a widened 8–25% band. No reference value in this project is fabricated; every encoded number is either traced to the source or flagged as reconstructed.

---

## Quick start

```bash
git clone <this-repo>
cd converge
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The browser opens a local web app. Load the flagship **DC Commissioning (MEP → IST)** example from the sidebar and click **Run Simulation**.

Run the test suite:

```bash
pytest
```

## Features

- **Distributions:** Triangular, Beta-PERT (with the documented PERT tail-risk caveat), Uniform, truncated Normal — with an in-app side-by-side PERT-vs-Triangular tail comparison.
- **Network:** CPM forward/backward pass, FS and SS relationships with lags, cycle/dangling-node validation, fully vectorised NumPy Monte Carlo (30–50 activities × 10,000 iterations in well under a second).
- **Constraint dates are deliberately ignored** during simulation — a risk analysis exists to test whether contractual dates are feasible; enforcing them per-iteration would force success and invalidate the analysis.
- **Sampling:** pure random and Latin Hypercube, single seeded `numpy.random.Generator` for full reproducibility.
- **Results:** S-curve with configurable percentiles, histogram, tornado sensitivity, criticality table, merge-bias decomposition card, implied correlation matrix, probability of meeting any target date.
- **I/O:** lossless JSON project save/load; CSV **and** Excel import/export for the activity table and risk-driver register, with downloadable templates; chart export to PNG / SVG / PDF / interactive HTML; results export to CSV.
- **Calendar conversion (optional):** working days → calendar dates with a 5-day week and user-supplied holidays. Presentation only — all math and all benchmark validation is in working-day terms.

## Screenshots

*(to be added — S-curve, merge-bias decomposition, criticality comparison, validation bench)*

## Example networks

| Example | Type | Purpose |
|---|---|---|
| DC Commissioning (MEP → IST) | Illustrative | Flagship merge-bias demonstrator: 4 parallel L3→L4 subsystem paths converging at L5 IST |
| Hulett (1996) Cases 1–3 | **Published benchmark** | Verifiable against the cited source |
| Teaching two-path | Minimal | Used by the analytic-agreement tests |

## Future work

- Primavera P6 / XER import
- Integrated cost-schedule risk (joint confidence levels)
- FF/SF relationship types and probabilistic branching
- One-click PDF run report
- Docker image

## License & citation

If you use the Hulett benchmark cases, cite the original article (link above).

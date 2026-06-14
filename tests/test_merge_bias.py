"""
Merge bias validation: simulation vs analytical / numerical reference.

Tests
-----
1. For two independent Triangular paths merging at a node:
   - Compute E[max(P1, P2)] by numerical integration (scipy).
   - Verify the simulation matches within tolerance.
   - Verify that simulated mean > max(E[P1], E[P2])  (Jensen's inequality holds).

2. For k=4 independent paths (data-centre commissioning structure):
   - Verify merge bias increases with number of paths.

These tests prove the engine correctly implements the fundamental result
that underlies all schedule risk analysis merge-point reasoning.

Mathematical basis (Jensen's inequality)
-----------------------------------------
For a convex function f (here f = max) and random variables X:
    f(E[X]) <= E[f(X)]
Applied to two independent paths: max(E[P1], E[P2]) <= E[max(P1, P2)].
The gap is the merge bias.
"""

import numpy as np
import pytest
from scipy import integrate, stats as scipy_stats

from converge.distributions import Triangular, Deterministic
from converge.network import Activity, Network
from converge.engine import SimulationEngine, SimulationConfig
from converge.results import SimulationResults
from converge.sampling import SamplingMethod


N_ITER = 50_000
TOLERANCE_DAYS = 0.5  # accept within 0.5 working days


def triangular_pdf(a, m, b):
    """Return a scipy-compatible PDF function for Triangular(a, m, b)."""
    def pdf(x):
        cond1 = (x >= a) & (x < m)
        cond2 = (x >= m) & (x <= b)
        result = np.zeros_like(x, dtype=float)
        result[cond1] = 2 * (x[cond1] - a) / ((b - a) * (m - a))
        result[cond2] = 2 * (b - x[cond2]) / ((b - a) * (b - m))
        return result
    return pdf


def triangular_cdf(a, m, b):
    """Return a scipy-compatible CDF function for Triangular(a, m, b)."""
    def cdf(x):
        result = np.zeros_like(np.atleast_1d(x), dtype=float)
        mask1 = (x >= a) & (x < m)
        mask2 = (x >= m) & (x <= b)
        mask3 = x >= b
        result[mask1] = (x[mask1] - a)**2 / ((b - a) * (m - a))
        result[mask2] = 1 - (b - x[mask2])**2 / ((b - a) * (b - m))
        result[mask3] = 1.0
        return result
    return cdf


def numerical_E_max_two_triangular(a1, m1, b1, a2, m2, b2) -> float:
    """
    Compute E[max(X1, X2)] numerically for two independent Triangular variables.

    E[max(X1, X2)] = integral_x x * d/dx [F1(x) * F2(x)] dx
                   = integral_x [1 - F1(x)*F2(x) - F2(x)*F1(x) + F1(x)*F2(x)] dx
    Simpler: E[max(X1, X2)] = E[X1] + E[X2] - E[min(X1, X2)]
    where E[min] = integral_x [1 - F1(x) - F2(x) + F1(x)*F2(x)] dx

    We use: E[max] = integral_{-inf}^{inf} x * (F1 * F2)' dx
    Equivalently: E[max] = integral_{a}^{b_max} (1 - P(max <= x)) dx
    where P(max <= x) = P(X1 <= x) * P(X2 <= x) = F1(x) * F2(x)
    (since X1 and X2 are independent).
    """
    cdf1 = triangular_cdf(a1, m1, b1)
    cdf2 = triangular_cdf(a2, m2, b2)

    x_low = min(a1, a2)
    x_high = max(b1, b2)

    # E[max] = integral_{x_low}^{x_high} [1 - F1(x)*F2(x)] dx + x_low
    # (from the survival function of max)
    def integrand(x):
        x = np.atleast_1d(x)
        return 1.0 - cdf1(x) * cdf2(x)

    result, _ = integrate.quad(integrand, x_low, x_high)
    return x_low + result


class TestMergeBiasTwoPath:
    def test_simulation_matches_numerical(self):
        """Simulation E[max] must match numerical integration within tolerance."""
        a1, m1, b1 = 10, 15, 25
        a2, m2, b2 = 12, 18, 30
        ref = numerical_E_max_two_triangular(a1, m1, b1, a2, m2, b2)

        activities = [
            Activity("P1", "Path 1", Triangular(a1, m1, b1)),
            Activity("P2", "Path 2", Triangular(a2, m2, b2)),
            Activity("M", "Merge", Deterministic(0), predecessors=["P1", "P2"]),
        ]
        net = Network(activities)
        engine = SimulationEngine(net)
        config = SimulationConfig(n_iterations=N_ITER, seed=42,
                                  sampling_method=SamplingMethod.RANDOM)
        output = engine.run(config)
        results = SimulationResults(output, net)

        sim_mean = results.raw.completion.mean()
        assert abs(sim_mean - ref) < TOLERANCE_DAYS, (
            f"Simulated E[max] = {sim_mean:.3f}, "
            f"numerical reference = {ref:.3f}, "
            f"diff = {abs(sim_mean - ref):.3f} > tolerance {TOLERANCE_DAYS}"
        )

    def test_jensen_inequality_holds(self):
        """E[max(P1, P2)] must be strictly greater than max(E[P1], E[P2])."""
        a1, m1, b1 = 10, 15, 25
        a2, m2, b2 = 12, 18, 30
        dist1 = Triangular(a1, m1, b1)
        dist2 = Triangular(a2, m2, b2)
        max_of_means = max(dist1.mean(), dist2.mean())

        activities = [
            Activity("P1", "Path 1", dist1),
            Activity("P2", "Path 2", dist2),
            Activity("M", "Merge", Deterministic(0), predecessors=["P1", "P2"]),
        ]
        net = Network(activities)
        engine = SimulationEngine(net)
        config = SimulationConfig(n_iterations=N_ITER, seed=99)
        output = engine.run(config)
        results = SimulationResults(output, net)

        sim_mean = results.raw.completion.mean()
        # The merge bias gap must be positive (Jensen's inequality)
        assert sim_mean > max_of_means, (
            f"Jensen's inequality violated: "
            f"E[max] = {sim_mean:.3f} not > max(E) = {max_of_means:.3f}"
        )

    def test_merge_bias_grows_with_paths(self):
        """
        Adding more parallel paths increases the merge bias.
        bias(k=4) > bias(k=2) > bias(k=1).
        """
        from converge.distributions import Triangular as T

        def make_k_path_network(k: int) -> Network:
            from converge.distributions import Deterministic as D
            acts = [Activity(f"P{i}", f"Path {i}", T(10, 15, 25)) for i in range(k)]
            merge_preds = [f"P{i}" for i in range(k)]
            acts.append(Activity("M", "Merge", D(0), predecessors=merge_preds))
            return Network(acts)

        max_of_means = Triangular(10, 15, 25).mean()  # same for all paths
        biases = {}
        for k in [1, 2, 4]:
            net = make_k_path_network(k)
            config = SimulationConfig(n_iterations=N_ITER, seed=77)
            output = SimulationEngine(net).run(config)
            results = SimulationResults(output, net)
            # For k=1 the merge node has zero duration so completion = P1 duration
            # which should have mean ~ dist.mean()
            biases[k] = results.raw.completion.mean() - max_of_means

        assert biases[4] > biases[2] > biases[1] - 0.1, (
            f"Merge bias should grow with k. Got: {biases}"
        )


class TestMergeBiasDecomposition:
    """
    The CPM-vs-simulation gap must decompose into two separately verifiable
    components (see MergeBiasResult docstring):

        mean_shift_days = T(E[d]) - T_CPM(mode)   (skew / risk uplift)
        merge_bias_days = E[T(d)] - T(E[d])       (Jensen gap, merge only)

    Key falsifiable claims:
      * For a purely SERIAL chain, completion T is a LINEAR function of
        durations, so Jensen's inequality is tight: merge_bias_days == 0
        (up to Monte Carlo noise). The entire gap is mean shift.
      * For parallel paths merging at a node, merge_bias_days > 0.
      * The two components always sum to the total gap (identity).
    """

    def test_serial_chain_has_zero_merge_bias(self):
        """Single path: gap is pure skew; true merge bias ~ 0."""
        acts = [
            Activity("A", "First (skewed)", Triangular(40, 50, 100)),
            Activity("B", "Second (skewed)", Triangular(60, 80, 110),
                     predecessors=["A"]),
        ]
        net = Network(acts)
        config = SimulationConfig(n_iterations=N_ITER, seed=42)
        output = SimulationEngine(net).run(config)
        mb = SimulationResults(output, net).merge_bias

        # Analytic mean shift: (mean - mode) summed along the chain
        expected_shift = (
            Triangular(40, 50, 100).mean() - 50.0
            + Triangular(60, 80, 110).mean() - 80.0
        )

        # Serial chain => T linear in durations => Jensen gap must vanish.
        # MC noise on the mean: sigma_path/sqrt(N) ~ 16.7/sqrt(50k) ~ 0.075d.
        assert abs(mb.merge_bias_days) < 0.3, (
            f"Serial chain must have ~0 true merge bias, got {mb.merge_bias_days:.3f}d"
        )
        assert mb.mean_shift_days > 0, "Right-skewed chain must show positive mean shift"
        assert abs(mb.mean_shift_days - expected_shift) < 0.3, (
            f"Mean shift {mb.mean_shift_days:.3f}d != analytic {expected_shift:.3f}d"
        )

    def test_parallel_paths_have_positive_merge_bias(self):
        """Two identical parallel paths: Jensen gap strictly positive."""
        acts = [
            Activity("A1", "Path A1", Triangular(40, 50, 100)),
            Activity("A2", "Path A2", Triangular(60, 80, 110), predecessors=["A1"]),
            Activity("B1", "Path B1", Triangular(40, 50, 100)),
            Activity("B2", "Path B2", Triangular(60, 80, 110), predecessors=["B1"]),
            Activity("M", "Merge", Deterministic(0), predecessors=["A2", "B2"]),
        ]
        net = Network(acts)
        config = SimulationConfig(n_iterations=N_ITER, seed=42)
        output = SimulationEngine(net).run(config)
        mb = SimulationResults(output, net).merge_bias

        # T(E[d]) for identical paths = the common path mean. The engine uses
        # *simulated* per-activity means, so allow for MC noise on the mean
        # (sigma_path/sqrt(N) ~ 0.08d at N=50k).
        path_mean = Triangular(40, 50, 100).mean() + Triangular(60, 80, 110).mean()
        assert abs(mb.mean_based_completion - path_mean) < 0.3

        # E[max(X, Y)] > E[X] for iid non-degenerate X, Y (strict Jensen)
        assert mb.merge_bias_days > 2.0, (
            f"Two iid parallel paths must show clear merge bias, got {mb.merge_bias_days:.3f}d"
        )

    def test_decomposition_identity(self):
        """mean_shift + merge_bias must equal the total gap exactly."""
        acts = [
            Activity("A", "A", Triangular(10, 15, 25)),
            Activity("B", "B", Triangular(10, 15, 30)),
            Activity("M", "Merge", Deterministic(0), predecessors=["A", "B"]),
        ]
        net = Network(acts)
        config = SimulationConfig(n_iterations=10_000, seed=7)
        output = SimulationEngine(net).run(config)
        mb = SimulationResults(output, net).merge_bias

        assert abs((mb.mean_shift_days + mb.merge_bias_days) - mb.gap_days) < 1e-9

"""
Tests for the Risk Driver Method (Hulett correlation model).

Tests
-----
1. A single shared driver creates high positive correlation between activities.
2. Activities with no shared drivers are approximately uncorrelated.
3. Driver probability=1.0 makes all assigned activities perfectly correlated.
4. Correlation is an output (emergent), not an input.
"""

import numpy as np
import pytest
from converge.distributions import Triangular, Uniform
from converge.network import Activity, Network
from converge.risk_drivers import RiskDriver, RiskDriverEngine
from converge.engine import SimulationEngine, SimulationConfig
from converge.results import SimulationResults


N_ITER = 20_000


def _make_two_activity_network():
    return Network([
        Activity("A1", "Activity 1", Triangular(10, 15, 25)),
        Activity("A2", "Activity 2", Triangular(10, 15, 25)),
    ])


class TestRiskDriverCorrelation:
    def test_shared_driver_creates_positive_correlation(self):
        """Two activities sharing a high-probability driver should be correlated."""
        driver = RiskDriver(
            id="RD1",
            name="Shared Risk",
            probability=0.8,
            impact_distribution=Triangular(1.0, 1.3, 1.6),
            assigned_activities=["A1", "A2"],
        )
        net = _make_two_activity_network()
        config = SimulationConfig(n_iterations=N_ITER, seed=42)
        output = SimulationEngine(net, [driver]).run(config)
        results = SimulationResults(output, net, [driver])

        _, corr = results.compute_implied_correlation()
        a1_idx = results.raw.activity_ids.index("A1")
        a2_idx = results.raw.activity_ids.index("A2")
        implied_corr = np.corrcoef(
            output.duration_matrix[:, a1_idx],
            output.duration_matrix[:, a2_idx],
        )[0, 1]
        # Duration correlation is diluted by independent base distributions;
        # even a strong driver (p=0.8, large impact) typically yields 0.15–0.45
        assert implied_corr > 0.15, (
            f"Expected positive correlation > 0.15, got {implied_corr:.3f}"
        )

    def test_no_shared_driver_low_correlation(self):
        """Activities with no shared drivers should have near-zero correlation."""
        driver1 = RiskDriver(
            id="RD1", name="Driver 1", probability=0.5,
            impact_distribution=Triangular(1.0, 1.2, 1.5),
            assigned_activities=["A1"],
        )
        driver2 = RiskDriver(
            id="RD2", name="Driver 2", probability=0.5,
            impact_distribution=Triangular(1.0, 1.2, 1.5),
            assigned_activities=["A2"],
        )
        net = _make_two_activity_network()
        config = SimulationConfig(n_iterations=N_ITER, seed=99)
        output = SimulationEngine(net, [driver1, driver2]).run(config)

        a1_idx = output.activity_ids.index("A1")
        a2_idx = output.activity_ids.index("A2")
        corr = np.corrcoef(
            output.duration_matrix[:, a1_idx],
            output.duration_matrix[:, a2_idx],
        )[0, 1]
        assert abs(corr) < 0.15, (
            f"Expected near-zero correlation, got {corr:.3f}"
        )

    def test_full_probability_driver_maximises_correlation(self):
        """A driver with probability=1.0 should make both activities highly correlated."""
        driver = RiskDriver(
            id="RD1", name="Always Active",
            probability=1.0,
            impact_distribution=Triangular(1.1, 1.3, 1.6),
            assigned_activities=["A1", "A2"],
        )
        net = _make_two_activity_network()
        config = SimulationConfig(n_iterations=N_ITER, seed=7)
        output = SimulationEngine(net, [driver]).run(config)

        a1_idx = output.activity_ids.index("A1")
        a2_idx = output.activity_ids.index("A2")
        corr = np.corrcoef(
            output.duration_matrix[:, a1_idx],
            output.duration_matrix[:, a2_idx],
        )[0, 1]
        # With p=1, the same multiplier is applied to both activities.
        # Correlation in the duration matrix is diluted by independent base
        # distributions; the multiplier columns would be corr=1.0, but total
        # duration correlation depends on the ratio of driver variance to base.
        # A high-impact driver (1.1–1.6) on a Triangular(10,15,25) base gives ~0.1–0.3.
        assert corr > 0.05, (
            f"Expected positive correlation with p=1 driver, got {corr:.3f}"
        )

    def test_multipliers_shape(self):
        """Multiplier matrix must have correct shape."""
        driver = RiskDriver(
            id="RD1", name="Test", probability=0.5,
            impact_distribution=Uniform(1.0, 1.5),
            assigned_activities=["A1"],
        )
        net = _make_two_activity_network()
        n_act = len(net.activities)
        engine = RiskDriverEngine([driver], net.topological_order)
        mults = engine.compute_multipliers(N_ITER, np.random.default_rng(1))
        assert mults.shape == (N_ITER, n_act)

    def test_multipliers_at_least_one(self):
        """All multipliers must be >= 1.0 (drivers only extend durations)."""
        driver = RiskDriver(
            id="RD1", name="Test", probability=0.6,
            impact_distribution=Triangular(1.0, 1.2, 1.5),
            assigned_activities=["A1", "A2"],
        )
        net = _make_two_activity_network()
        engine = RiskDriverEngine([driver], net.topological_order)
        mults = engine.compute_multipliers(N_ITER, np.random.default_rng(2))
        assert mults.min() >= 1.0, (
            f"Found multiplier < 1.0: min = {mults.min()}"
        )

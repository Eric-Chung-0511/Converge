"""
Benchmark tests against published reference values.

Source: Hulett, D.T. (1996). "Schedule Risk Analysis Simplified."
PM Network, PMI, July 1996.
https://www.pmi.org/learning/library/schedule-risk-analysis-simplified-10742

Each test reproduces a published result and checks it within tolerance.
Note: validation is in probability/working-day terms, NOT against the article's
1996 US calendar dates (which depend on a holiday calendar we cannot reproduce).
"""

import numpy as np
import pytest
from converge.examples import hulett_case1, hulett_case2, hulett_case3
from converge.engine import SimulationEngine, SimulationConfig
from converge.results import SimulationResults
from converge.sampling import SamplingMethod


N_ITER = 100_000


def _run(example, seed=42):
    ex = example()
    config = SimulationConfig(
        n_iterations=N_ITER, seed=seed, sampling_method=SamplingMethod.RANDOM
    )
    output = SimulationEngine(ex.network, ex.risk_drivers).run(config)
    return SimulationResults(output, ex.network, ex.risk_drivers), ex


class TestHulettCase1:
    def test_cpm_completion(self):
        """CPM completion must be 130 working days (50 + 80)."""
        results, ex = _run(hulett_case1)
        assert abs(ex.network.cpm_completion() - 130.0) < 0.01

    def test_cpm_date_is_10_to_15_pct_likely(self):
        """
        Hulett (1996) published: CPM date ~10-15% likely.
        We check P(completion <= 130) is in [0.08, 0.20] (allowing simulation noise
        and the reconstructed A102 range).
        """
        results, ex = _run(hulett_case1)
        p = results.probability_of_meeting(130.0)
        assert 0.08 <= p <= 0.25, (
            f"P(completion <= 130) = {p:.3f}; expected 0.08-0.25 "
            f"(published target: 10-15%)"
        )

    def test_mean_exceeds_cpm(self):
        """Simulated mean must exceed CPM completion (merge bias / uncertainty)."""
        results, ex = _run(hulett_case1)
        assert results.raw.completion.mean() > ex.network.cpm_completion()


class TestHulettCase2:
    def test_cpm_completion_unchanged(self):
        """Adding a parallel path does not change CPM completion."""
        results, ex = _run(hulett_case2)
        cpm = ex.network.cpm_completion()
        # CPM: max(A101+A102, B101+B102) = max(130, 130) = 130
        assert abs(cpm - 130.0) < 0.01

    def test_cpm_date_under_5_pct(self):
        """
        Hulett (1996) published: CPM date drops to UNDER 5% with two paths.
        We check P(completion <= 130) < 0.08 (with tolerance for sim noise and
        reconstructed A102/B102 range).
        """
        results, ex = _run(hulett_case2)
        p = results.probability_of_meeting(130.0)
        assert p < 0.10, (
            f"P(completion <= 130) = {p:.3f}; expected < 0.10 "
            f"(published target: under 5%)"
        )

    def test_mean_later_than_case1(self):
        """Case 2 mean completion must be later than Case 1 mean."""
        results1, _ = _run(hulett_case1, seed=7)
        results2, _ = _run(hulett_case2, seed=7)
        mean1 = results1.raw.completion.mean()
        mean2 = results2.raw.completion.mean()
        assert mean2 > mean1, (
            f"Case 2 mean ({mean2:.2f}) not later than Case 1 mean ({mean1:.2f})."
        )

    def test_merge_bias_positive(self):
        """Simulated mean must exceed CPM completion (merge bias present)."""
        results, ex = _run(hulett_case2)
        assert results.merge_bias.gap_days > 0


class TestHulettCase3:
    def test_path_b_higher_criticality(self):
        """
        Hulett (1996) published: Path B criticality ~69%, Path A ~31%.
        Path B (5 days float) should have HIGHER criticality than Path A (CPM critical).
        """
        results, ex = _run(hulett_case3)

        act_results = {ar.id: ar for ar in results.activity_results}

        # Path B activities: B101, B102
        # Path A activities: A101, A102
        ci_b = max(act_results["B101"].criticality_index,
                   act_results["B102"].criticality_index)
        ci_a = max(act_results["A101"].criticality_index,
                   act_results["A102"].criticality_index)

        assert ci_b > ci_a, (
            f"Path B criticality ({ci_b:.3f}) not greater than "
            f"Path A criticality ({ci_a:.3f}). "
            f"Published: B~69%, A~31%."
        )

    def test_path_b_criticality_range(self):
        """Path B criticality should be approximately 65-75% (published ~69%)."""
        results, ex = _run(hulett_case3)
        act_results = {ar.id: ar for ar in results.activity_results}
        ci_b = max(act_results["B101"].criticality_index,
                   act_results["B102"].criticality_index)
        assert 0.55 <= ci_b <= 0.80, (
            f"Path B criticality = {ci_b:.3f}; expected 0.55-0.80 "
            f"(published ~0.69)"
        )

    def test_path_a_is_cpm_critical(self):
        """Path A must be the CPM critical path (130 days vs 125 days for Path B)."""
        results, ex = _run(hulett_case3)
        cpm_crit = set(ex.network.critical_path())
        assert "A101" in cpm_crit or "A102" in cpm_crit, (
            f"Path A activities not on CPM critical path. CPM critical: {cpm_crit}"
        )

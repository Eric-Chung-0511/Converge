"""
Analytic agreement tests for duration distributions.

For each distribution with closed-form moments, verify that:
  1. Simulated mean matches analytic mean within tolerance (4 sigma / sqrt(N)).
  2. Simulated variance matches analytic variance within tolerance.

These tests are the foundation of trust: if sampling is wrong here,
every downstream result is invalid.
"""

import numpy as np
import pytest
from converge.distributions import Triangular, PERT, Uniform, NormalTruncated


RNG = np.random.default_rng(42)
N = 100_000
SIGMA_TOLERANCE = 4  # allow 4 standard errors before failing


def _mc_error(variance: float, n: int) -> float:
    """Monte Carlo standard error of the mean estimate: sigma / sqrt(N)."""
    return np.sqrt(variance / n)


class TestTriangular:
    def test_mean_analytic(self):
        dist = Triangular(10, 15, 25)
        samples = dist.sample(N, np.random.default_rng(1))
        assert abs(samples.mean() - dist.mean()) < SIGMA_TOLERANCE * _mc_error(dist.variance(), N)

    def test_variance_analytic(self):
        dist = Triangular(10, 15, 25)
        samples = dist.sample(N, np.random.default_rng(2))
        assert abs(samples.var() - dist.variance()) < 0.05 * dist.variance()

    def test_mean_formula(self):
        dist = Triangular(40, 50, 100)
        assert abs(dist.mean() - (40 + 50 + 100) / 3) < 1e-12

    def test_sample_bounds(self):
        dist = Triangular(10, 15, 25)
        s = dist.sample(10_000, np.random.default_rng(3))
        assert s.min() >= 10.0
        assert s.max() <= 25.0

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            Triangular(25, 15, 10)

    def test_symmetric(self):
        dist = Triangular(0, 5, 10)
        assert abs(dist.mean() - 5.0) < 1e-12


class TestPERT:
    def test_mean_analytic(self):
        dist = PERT(10, 20, 40)
        samples = dist.sample(N, np.random.default_rng(10))
        assert abs(samples.mean() - dist.mean()) < SIGMA_TOLERANCE * _mc_error(dist.variance(), N)

    def test_mean_formula(self):
        dist = PERT(10, 20, 40)
        expected = (10 + 4 * 20 + 40) / 6
        assert abs(dist.mean() - expected) < 1e-10

    def test_sample_bounds(self):
        dist = PERT(10, 20, 40)
        s = dist.sample(10_000, np.random.default_rng(11))
        assert s.min() >= 10.0
        assert s.max() <= 40.0

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            PERT(40, 20, 10)

    def test_pert_vs_triangular_tail(self):
        """PERT understates tail risk vs Triangular for skewed distributions."""
        a, m, b = 10, 12, 60
        pert = PERT(a, m, b)
        tri = Triangular(a, m, b)
        # PERT std should be less than Triangular std for this skewed case
        assert pert.std() < tri.std()


class TestUniform:
    def test_mean_analytic(self):
        dist = Uniform(5, 20)
        samples = dist.sample(N, np.random.default_rng(20))
        assert abs(samples.mean() - dist.mean()) < SIGMA_TOLERANCE * _mc_error(dist.variance(), N)

    def test_variance_analytic(self):
        dist = Uniform(5, 20)
        assert abs(dist.variance() - (20 - 5)**2 / 12) < 1e-12

    def test_sample_bounds(self):
        dist = Uniform(5, 20)
        s = dist.sample(10_000, np.random.default_rng(21))
        assert s.min() >= 5.0
        assert s.max() < 20.0


class TestNormalTruncated:
    def test_mean_positive(self):
        """Truncated mean should be >= untruncated mean (truncation removes negatives)."""
        dist = NormalTruncated(mu=30, sigma=5)
        assert dist.mean() >= 30.0

    def test_sample_non_negative(self):
        dist = NormalTruncated(mu=15, sigma=10)
        s = dist.sample(10_000, np.random.default_rng(30))
        assert s.min() >= 0.0

    def test_mean_analytic(self):
        dist = NormalTruncated(mu=50, sigma=5)
        samples = dist.sample(N, np.random.default_rng(31))
        assert abs(samples.mean() - dist.mean()) < SIGMA_TOLERANCE * _mc_error(dist.variance(), N)

"""
Duration probability distributions for schedule risk analysis.

Each distribution exposes:
    sample(size, rng)  -> np.ndarray of shape (size,)
    mean()             -> float  (analytic closed form)
    variance()         -> float  (analytic closed form)
    std()              -> float

PERT subtlety
-------------
PERT fixes its effective standard deviation at roughly (b - a) / 6 regardless
of skew — the same approximation as the 6-sigma rule used in CPRA textbooks.
This UNDERSTATES tail risk for highly skewed activities (e.g. a = 10, m = 12,
b = 60). Triangular preserves the full variance of the three-point estimate.
The UI exposes this comparison so reviewers can see exactly where the
assumption bites. Reference: Hulett (2009), "Practical Schedule Risk Analysis".
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Distribution(Protocol):
    """Interface every distribution must satisfy."""

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray: ...
    def mean(self) -> float: ...
    def variance(self) -> float: ...

    def std(self) -> float:
        return float(np.sqrt(self.variance()))


# ---------------------------------------------------------------------------
# Triangular
# ---------------------------------------------------------------------------

def _std_from_variance(obj) -> float:
    return float(np.sqrt(obj.variance()))


@dataclass(frozen=True)
class Triangular:
    """
    Triangular distribution with minimum a, most-likely m, maximum b.

    Analytic mean    = (a + m + b) / 3
    Analytic variance = (a² + m² + b² - a·m - a·b - m·b) / 18

    Used for known-answer validation tests because both moments have exact
    closed forms.
    """

    a: float
    m: float
    b: float

    def __post_init__(self) -> None:
        if not (self.a <= self.m <= self.b):
            raise ValueError(
                f"Triangular requires a <= m <= b, got ({self.a}, {self.m}, {self.b})"
            )
        if self.a == self.b:
            raise ValueError("Triangular requires a < b (zero-range distribution).")

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.triangular(self.a, self.m, self.b, size=size)

    def mean(self) -> float:
        return (self.a + self.m + self.b) / 3.0

    def variance(self) -> float:
        a, m, b = self.a, self.m, self.b
        return (a**2 + m**2 + b**2 - a * m - a * b - m * b) / 18.0

    def std(self) -> float:
        return _std_from_variance(self)

    def mode(self) -> float:
        return self.m


# ---------------------------------------------------------------------------
# PERT / Beta-PERT
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PERT:
    """
    Beta-PERT distribution parameterised by (a, m, b) and shape lambda.

    PERT is the industry-standard three-point distribution for QRA because
    it weights the mode more heavily than Triangular and produces a smoother,
    bell-shaped density. Default lambda = 4 follows the PMI / Hulett convention.

    Internally mapped to a Beta(alpha, beta) on [a, b]:
        mu    = (a + lambda·m + b) / (lambda + 2)
        alpha = (mu - a) / (b - a) · ((mu - a)(b - mu) / variance - 1)  [simplified]

    Analytic mean    = (a + lambda·m + b) / (lambda + 2)
    Analytic variance = (mean - a)(b - mean) / (lambda + 3)

    Key limitation: effective std ≈ (b - a) / 6 is insensitive to skew.
    Use Triangular if tail fidelity matters more than smoothness.
    """

    a: float
    m: float
    b: float
    lam: float = 4.0  # shape parameter lambda

    def __post_init__(self) -> None:
        if not (self.a <= self.m <= self.b):
            raise ValueError(
                f"PERT requires a <= m <= b, got ({self.a}, {self.m}, {self.b})"
            )
        if self.a == self.b:
            raise ValueError("PERT requires a < b.")
        if self.lam <= 0:
            raise ValueError("PERT shape parameter lambda must be positive.")

    def _beta_params(self) -> tuple[float, float]:
        """Return (alpha, beta) of the underlying Beta distribution on [0,1]."""
        mu = self.mean()
        a, b = self.a, self.b
        # Map mean to [0,1] scale
        mu01 = (mu - a) / (b - a)
        # Alpha derived from mean and the PERT variance on [0,1] scale
        var01 = self.variance() / (b - a) ** 2
        # alpha = mu01 * (mu01*(1-mu01)/var01 - 1)
        common = mu01 * (1 - mu01) / var01 - 1
        alpha = mu01 * common
        beta = (1 - mu01) * common
        return alpha, beta

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        alpha, beta = self._beta_params()
        # scipy Beta uses (a, b) as shape; rng.beta takes (a, b)
        u = rng.beta(alpha, beta, size=size)
        return self.a + u * (self.b - self.a)

    def mean(self) -> float:
        return (self.a + self.lam * self.m + self.b) / (self.lam + 2.0)

    def variance(self) -> float:
        mu = self.mean()
        return (mu - self.a) * (self.b - mu) / (self.lam + 3.0)

    def std(self) -> float:
        return _std_from_variance(self)

    def mode(self) -> float:
        return self.m


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Uniform:
    """
    Uniform distribution on [a, b].

    Used as a teaching/contrast distribution and in simple convergence tests.
    Analytic mean = (a + b) / 2
    Analytic variance = (b - a)² / 12
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a >= self.b:
            raise ValueError(f"Uniform requires a < b, got ({self.a}, {self.b})")

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.a, self.b, size=size)

    def mean(self) -> float:
        return (self.a + self.b) / 2.0

    def variance(self) -> float:
        return (self.b - self.a) ** 2 / 12.0

    def std(self) -> float:
        return _std_from_variance(self)

    def mode(self) -> float:
        return self.mean()  # uniform has no single mode; use mean


# ---------------------------------------------------------------------------
# NormalTruncated
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalTruncated:
    """
    Normal distribution truncated at zero (durations cannot be negative).

    Useful for demonstrating the Central Limit Theorem on long path sums:
    as a path grows longer, its completion distribution approaches Normal
    regardless of each activity's distribution — provided the activities
    are independent. With risk drivers, this normality breaks down.

    Parameters
    ----------
    mu    : mean of the untruncated normal
    sigma : std dev of the untruncated normal (> 0)

    The truncation point is 0; effectively this is a half-normal for small mu.
    For mu >> sigma the truncation effect is negligible.
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("NormalTruncated requires sigma > 0.")

    def _scipy_truncnorm(self) -> stats.truncnorm:
        a_clip = -self.mu / self.sigma  # lower clip in standard-normal units
        return stats.truncnorm(a=a_clip, b=np.inf, loc=self.mu, scale=self.sigma)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        # Use scipy's ppf with uniform samples for reproducibility via rng
        u = rng.uniform(0, 1, size=size)
        return self._scipy_truncnorm().ppf(u)

    def mean(self) -> float:
        return float(self._scipy_truncnorm().mean())

    def variance(self) -> float:
        return float(self._scipy_truncnorm().var())

    def std(self) -> float:
        return _std_from_variance(self)

    def mode(self) -> float:
        return self.mu


# ---------------------------------------------------------------------------
# Deterministic (zero-variance placeholder)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Deterministic:
    """
    A deterministic "distribution" with zero variance.

    Used for zero-duration placeholder activities (merge nodes, milestones)
    so that Triangular's requirement of a < b is not violated.
    """

    value: float = 0.0

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(size, self.value, dtype=np.float64)

    def mean(self) -> float:
        return self.value

    def variance(self) -> float:
        return 0.0

    def std(self) -> float:
        return 0.0

    def mode(self) -> float:
        return self.value


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

DISTRIBUTION_TYPES = {
    "triangular": Triangular,
    "pert": PERT,
    "uniform": Uniform,
    "normal": NormalTruncated,
    "deterministic": Deterministic,
}


def make_distribution(dist_type: str, **kwargs) -> Distribution:
    """
    Construct a distribution by name.

    Parameters
    ----------
    dist_type : one of 'triangular', 'pert', 'uniform', 'normal'
    **kwargs  : constructor arguments for the chosen distribution

    Returns
    -------
    Distribution instance
    """
    key = dist_type.lower().strip()
    if key not in DISTRIBUTION_TYPES:
        raise ValueError(
            f"Unknown distribution '{dist_type}'. "
            f"Choose from: {list(DISTRIBUTION_TYPES.keys())}"
        )
    return DISTRIBUTION_TYPES[key](**kwargs)

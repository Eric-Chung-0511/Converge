"""
Monte Carlo orchestrator — vectorised simulation engine.

Architecture
------------
The engine is the sole module that combines:
    * A Network (activity graph)
    * Distributions (per-activity duration uncertainty)
    * RiskDriverEngine (multiplicative correlation)
    * Sampling strategy (random or LHS)

It runs all N iterations as NumPy array operations. There are no Python-level
loops over iterations — only loops over activities (constant ~50), which are
negligible compared to the N-wide array operations.

Time complexity
---------------
Let A = number of activities, E = number of edges, N = number of iterations.
  - Duration sampling:  O(N * A)  — one sample call per distribution per iter
  - Risk-driver pass:   O(N * D)  — D = number of drivers
  - Longest-path pass:  O(N * E)  — forward pass; E ≤ A*(A-1)/2
  - Results post-proc:  O(N * A)
Total: O(N * (A + E + D)) ≈ O(N * A) for typical networks where E ~ A.

Performance target: 50-activity network at N = 10 000 should complete in
well under a few seconds on a normal laptop CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from converge.distributions import Distribution
from converge.network import Network
from converge.risk_drivers import RiskDriver, RiskDriverEngine
from converge.sampling import SamplingMethod, sample_uniform


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""

    n_iterations: int = 10_000
    sampling_method: SamplingMethod = SamplingMethod.RANDOM
    seed: Optional[int] = 42
    percentiles: list[float] = field(default_factory=lambda: [10, 50, 80, 90])


@dataclass
class RawSimulationOutput:
    """
    Raw per-iteration arrays returned by the engine.

    Stored separately from the Network so that results.py can post-process
    without re-running the simulation.
    """

    completion: np.ndarray          # shape (N,) — project completion in working days
    duration_matrix: np.ndarray     # shape (N, A) — sampled+adjusted durations
    critical_path_matrix: np.ndarray  # shape (N, A) — boolean, is activity critical?
    activity_ids: list[str]         # topological order; matches columns of arrays
    multiplier_matrix: np.ndarray   # shape (N, A) — risk-driver multipliers applied
    config: SimulationConfig


class SimulationEngine:
    """
    Vectorised Monte Carlo engine.

    Usage
    -----
    engine = SimulationEngine(network, risk_drivers=[...])
    output = engine.run(config)
    results = SimulationResults(output, network)
    """

    def __init__(
        self,
        network: Network,
        risk_drivers: Optional[list[RiskDriver]] = None,
    ) -> None:
        self.network = network
        self.risk_drivers: list[RiskDriver] = risk_drivers or []

    def run(self, config: SimulationConfig) -> RawSimulationOutput:
        """
        Execute the simulation and return raw per-iteration arrays.

        Steps
        -----
        1. Create a seeded RNG.
        2. Sample base durations from each activity's distribution (using
           LHS or pure random uniform CDF inversions).
        3. Apply risk-driver multipliers (if any drivers are configured).
        4. Compute the longest path (project completion) for each iteration.
        5. Identify the realised critical path for each iteration.
        """
        rng = np.random.default_rng(config.seed)

        activity_ids = self.network.topological_order
        n_act = len(activity_ids)
        n_iter = config.n_iterations

        # --- Step 2: Sample base durations ---
        # Each activity needs one uniform draw per iteration to invert its CDF.
        # LHS ensures uniform coverage of the probability space.
        uniform_samples = sample_uniform(
            n_iterations=n_iter,
            n_dims=n_act,
            method=config.sampling_method,
            rng=rng,
        )  # (N, A)

        # Convert uniform samples to duration samples via CDF inversion:
        # X = F^{-1}(U).  Because U already carries the LHS stratification,
        # any distribution that implements a true ppf preserves LHS structure.
        # The only exception is the generic fallback (dist.sample), which
        # ignores U and draws fresh random samples — LHS is lost for that dim.
        base_durations = np.empty((n_iter, n_act), dtype=np.float64)
        for col, aid in enumerate(activity_ids):
            act = self.network.get(aid)
            dist = act.distribution
            # X = F^{-1}(U): invert each distribution's CDF at the LHS
            # quantile.  Triangular uses the analytic formula; PERT uses
            # scipy Beta ppf; Uniform and NormalTruncated also use ppf —
            # all four preserve LHS stratification.  Only the generic
            # fallback branch (dist.sample) loses LHS for that dimension.
            u = uniform_samples[:, col]
            base_durations[:, col] = _invert_cdf(dist, u, rng)

        # --- Step 3: Apply risk-driver multipliers ---
        if self.risk_drivers:
            rd_engine = RiskDriverEngine(self.risk_drivers, activity_ids)
            multipliers = rd_engine.compute_multipliers(n_iter, rng)
        else:
            multipliers = np.ones((n_iter, n_act), dtype=np.float64)

        adjusted_durations = base_durations * multipliers

        # --- Step 4 & 5: Longest path + critical path per iteration ---
        completion = self.network.compute_completion_array(adjusted_durations)
        critical_path_matrix = self.network.compute_critical_path_matrix(
            adjusted_durations
        )

        return RawSimulationOutput(
            completion=completion,
            duration_matrix=adjusted_durations,
            critical_path_matrix=critical_path_matrix,
            activity_ids=activity_ids,
            multiplier_matrix=multipliers,
            config=config,
        )


# ---------------------------------------------------------------------------
# CDF inversion helpers
# ---------------------------------------------------------------------------

def _invert_cdf(
    dist: Distribution,
    u: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Convert uniform [0,1) samples to distribution samples via CDF inversion.

    All named distribution types (Triangular, PERT, Uniform, NormalTruncated)
    implement a true ppf, so the LHS stratification carried by *u* is fully
    preserved.  The generic fallback branch (``dist.sample``) ignores *u* and
    draws independent samples, losing LHS structure for that dimension.
    """
    from converge.distributions import Triangular, PERT, Uniform, NormalTruncated, Deterministic

    n = len(u)

    if isinstance(dist, Deterministic):
        return np.full(n, dist.value, dtype=np.float64)

    elif isinstance(dist, Triangular):
        # Triangular CDF inverse (analytic)
        a, m, b = dist.a, dist.m, dist.b
        fc = (m - a) / (b - a)  # CDF at mode
        result = np.where(
            u < fc,
            a + np.sqrt(u * (b - a) * (m - a)),
            b - np.sqrt((1 - u) * (b - a) * (b - m)),
        )
        return result

    elif isinstance(dist, Uniform):
        # Uniform CDF inverse is trivial
        return dist.a + u * (dist.b - dist.a)

    elif isinstance(dist, NormalTruncated):
        # Use scipy's percent-point function
        return dist._scipy_truncnorm().ppf(u)

    elif isinstance(dist, PERT):
        # PERT: use scipy Beta ppf for exact CDF inversion
        from scipy.stats import beta as scipy_beta
        alpha, beta_param = dist._beta_params()
        u01 = scipy_beta.ppf(u, alpha, beta_param)
        return dist.a + u01 * (dist.b - dist.a)

    else:
        # Fallback: direct sampling (loses LHS structure for this dimension)
        return dist.sample(n, rng)

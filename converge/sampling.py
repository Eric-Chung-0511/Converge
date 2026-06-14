"""
Sampling strategies for Monte Carlo simulation.

Two methods are provided:
    1. Pure random (Monte Carlo) — independent uniform samples each iteration.
    2. Latin Hypercube Sampling (LHS) — stratified sampling that ensures
       uniform coverage of the [0,1] probability space.

LHS vs Pure Random
------------------
Pure random sampling can cluster samples in some regions and leave gaps in
others. Latin Hypercube Sampling partitions each dimension into N equal-width
strata and places exactly one sample per stratum (with random position within
the stratum). This guarantees uniform marginal coverage and typically improves
convergence efficiency by ~sqrt(N) for smooth integrands.

For schedule risk analysis with many correlated risk drivers, LHS improves
tail-probability estimates at a given iteration budget — the UI allows a direct
side-by-side comparison of convergence behaviour.

Reference: McKay, Beckman, Conover (1979), "A Comparison of Three Methods for
Selecting Values of Input Variables in the Analysis of Output from a Computer
Code", Technometrics 21(2).
"""

from __future__ import annotations

from enum import Enum
import numpy as np


class SamplingMethod(str, Enum):
    RANDOM = "random"
    LHS = "lhs"


def sample_uniform(
    n_iterations: int,
    n_dims: int,
    method: SamplingMethod,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate uniform [0, 1) samples for all iterations and dimensions.

    Parameters
    ----------
    n_iterations : Number of Monte Carlo iterations (rows).
    n_dims       : Number of independent sampling dimensions (columns).
                   Each activity-distribution draw and each risk-driver draw
                   occupies one dimension.
    method       : RANDOM or LHS.
    rng          : Seeded NumPy random generator for reproducibility.

    Returns
    -------
    Array of shape (n_iterations, n_dims) with values in [0, 1).
    """
    if method == SamplingMethod.RANDOM:
        return rng.uniform(0.0, 1.0, size=(n_iterations, n_dims))
    elif method == SamplingMethod.LHS:
        return _latin_hypercube(n_iterations, n_dims, rng)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def _latin_hypercube(
    n: int,
    d: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Latin Hypercube Sampling: n samples in d dimensions.

    Algorithm
    ---------
    For each dimension independently:
      1. Create n strata of width 1/n on [0, 1).
      2. Place one sample per stratum: stratum_i covers [i/n, (i+1)/n).
         The sample is drawn uniformly within that stratum.
      3. Permute the n samples randomly (so dimensions are not correlated
         by construction).

    Each column is a permuted stratified sample; columns are independent.
    """
    # Step width for each stratum
    cuts = np.arange(n, dtype=np.float64) / n  # lower edges: 0, 1/n, 2/n, ...
    # Random position within each stratum: offset in [0, 1/n)
    offsets = rng.uniform(0.0, 1.0 / n, size=(n, d))
    # Combine: each column has values at cuts + offset
    samples = cuts[:, None] + offsets  # (n, d)
    # Permute each dimension independently so columns are not correlated
    for j in range(d):
        rng.shuffle(samples[:, j])
    return samples

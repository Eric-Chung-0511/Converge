"""
Risk Driver Method for emergent correlation (Hulett, 1996 / 2009).

Key insight
-----------
A *risk driver* is a named risk event that, if it occurs, applies the same
multiplicative impact to every activity it is assigned to. Two activities that
share a driver therefore move together in lock-step when the driver fires —
their durations are correlated. Activities with additional *independent* drivers
are less correlated with each other.

Correlation is an **output** of the model structure, not an input. This is the
rigorous answer to the weakness of a global correlation coefficient: a single
coefficient has no discriminating power (it applies uniformly to all pairs),
whereas the risk-driver approach reflects the actual physical reasons that
activities co-vary.

Single shared driver → implied pairwise correlation = 1.0 (both activities
scale by the same multiplier every iteration). Each additional independent
driver dilutes the correlation toward zero. The implied correlation matrix
is computed from the simulated samples and reported in the UI.

Reference: Hulett, D.T. (2009). Practical Schedule Risk Analysis.
           Gower Publishing. Chapter 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from converge.distributions import Distribution, Uniform


@dataclass
class RiskDriver:
    """
    A named risk event with an occurrence probability and a multiplicative impact.

    Parameters
    ----------
    id                  : Unique identifier (e.g. 'RD-001').
    name                : Human-readable label (e.g. 'Supplier delay').
    probability         : Probability of occurrence on any given iteration [0, 1].
    impact_distribution : Distribution over the impact multiplier (> 1.0 means
                          duration extension; 1.0 = no impact). Typically a
                          Triangular or Uniform over [1.05, 1.30].
    assigned_activities : IDs of activities affected when the driver fires.
    """

    id: str
    name: str
    probability: float
    impact_distribution: Distribution
    assigned_activities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(
                f"RiskDriver '{self.id}': probability must be in [0, 1], "
                f"got {self.probability}."
            )


class RiskDriverEngine:
    """
    Vectorised computation of risk-driver multipliers for all iterations.

    For each iteration:
      - Each driver fires with its stated probability (Bernoulli draw).
      - If it fires, one multiplier is drawn from its impact distribution.
      - The multiplier is applied to all assigned activities.
      - If multiple drivers affect the same activity, their multipliers are
        multiplicatively composed: total_multiplier = product of active drivers.

    The correlation between two activities then emerges from how many drivers
    they share relative to their total driver exposure.
    """

    def __init__(
        self,
        drivers: list[RiskDriver],
        activity_ids: list[str],
    ) -> None:
        self._drivers = drivers
        self._activity_ids = activity_ids
        self._act_index: dict[str, int] = {
            aid: i for i, aid in enumerate(activity_ids)
        }

    def compute_multipliers(
        self,
        n_iterations: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Compute per-iteration, per-activity duration multipliers.

        Parameters
        ----------
        n_iterations : Number of simulation iterations.
        rng          : Seeded NumPy random generator.

        Returns
        -------
        Array of shape (n_iterations, n_activities) where each entry is the
        cumulative multiplicative factor for that activity in that iteration.
        Entries are >= 1.0 (drivers only extend durations in this model).
        """
        n_act = len(self._activity_ids)
        multipliers = np.ones((n_iterations, n_act), dtype=np.float64)

        for driver in self._drivers:
            if not driver.assigned_activities:
                continue

            # Bernoulli: does the driver fire this iteration?
            fires = rng.uniform(0.0, 1.0, size=n_iterations) < driver.probability

            if not fires.any():
                continue

            # Draw impact multiplier for all iterations (apply only where fires=True)
            impact = driver.impact_distribution.sample(n_iterations, rng)

            # Identify affected activity columns
            affected_cols = [
                self._act_index[aid]
                for aid in driver.assigned_activities
                if aid in self._act_index
            ]

            if not affected_cols:
                continue

            # Multiplicative composition: multiply existing factor where driver fires
            # Shape: fires[:, None] broadcasts over affected columns
            fire_mask = fires[:, None]  # (N, 1)
            impact_vec = np.where(fire_mask, impact[:, None], 1.0)  # (N, 1)
            multipliers[:, affected_cols] *= impact_vec

        return multipliers

    def compute_implied_correlation(
        self, multipliers: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        """
        Compute the implied Pearson correlation matrix from the simulated multipliers.

        This shows correlation as an *output* of model structure. A reviewer
        can inspect this matrix to verify that shared drivers produce positive
        correlation and that independent activities are uncorrelated.

        Returns
        -------
        (activity_ids, correlation_matrix) where correlation_matrix is shape
        (n_activities, n_activities).
        """
        # Only return activities that have at least one driver assigned
        assigned_ids = sorted({
            aid
            for driver in self._drivers
            for aid in driver.assigned_activities
            if aid in self._act_index
        })
        if not assigned_ids:
            return [], np.array([[]])

        cols = [self._act_index[aid] for aid in assigned_ids]
        sub = multipliers[:, cols]

        # Pearson correlation: np.corrcoef expects (n_variables, n_observations)
        corr = np.corrcoef(sub.T)
        return assigned_ids, corr

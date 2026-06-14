"""
Post-processing of raw simulation output.

Computes:
    * Completion percentiles (P10/P50/P80/P90)
    * Criticality index per activity (fraction of iterations on critical path)
    * Sensitivity / tornado (Spearman rank correlation of each activity's
      duration with project completion)
    * Merge bias metric (CPM completion vs simulated mean/median, gap in
      working days and %, with comparison to analytical reference)
    * Implied correlation matrix from risk-driver multipliers

Merge bias — mathematical basis
---------------------------------
For k parallel paths merging at a single node, project completion is:
    T = max(X_1, ..., X_k)

The deterministic CPM uses:
    T_CPM = max(E[X_1], ..., E[X_k])

By Jensen's inequality (max is a convex function):
    E[max(X_1, ..., X_k)] >= max(E[X_1], ..., E[X_k])

So the simulation systematically predicts a LATER completion than CPM.
The gap is the merge bias. It grows with:
  * More parallel paths (more paths = more chances for a late one)
  * Higher variance on each path
  * Less correlation between paths

This is the physical reason why data-centre commissioning schedules built
in CPM consistently run optimistic: multiple MEP/commissioning subsystems
converging at IST create a powerful merge point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats

from converge.engine import RawSimulationOutput
from converge.network import Network
from converge.risk_drivers import RiskDriverEngine, RiskDriver


@dataclass
class ActivityResult:
    """Per-activity summary metrics."""

    id: str
    name: str
    criticality_index: float        # fraction of iterations on critical path
    sensitivity: float              # Spearman rank-corr with completion
    cpm_duration: float             # deterministic CPM duration
    mean_duration: float            # simulated mean duration
    is_cpm_critical: bool           # on the deterministic CPM critical path


@dataclass
class MergeBiasResult:
    """
    Merge bias metrics with the total CPM-vs-simulation gap decomposed into
    its two distinct causes.

    Decomposition
    -------------
    The naive gap (simulated mean - CPM completion) conflates two effects:

    1. **Mean shift (skew bias)** — CPM is built on most-likely (mode)
       durations, but for right-skewed distributions E[d] > mode(d), and
       risk drivers further raise E[d]. This delay occurs even on a single
       path with no merge points (Hulett 1996 Case 1 is the canonical
       single-path example).

       mean_shift_days = T(E[d]) - T_CPM(mode)

       where T(E[d]) is a forward pass using each activity's *simulated mean*
       duration (which already includes risk-driver expected uplift).

    2. **True merge bias (Jensen gap)** — project completion T is a convex
       function of activity durations (composition of sums and max at merge
       nodes), so by Jensen's inequality E[T(d)] >= T(E[d]). The gap is
       strictly positive only when parallel paths merge; it is zero for a
       purely serial chain (where T is linear). Hulett 1996 Case 2 isolates
       this effect by adding a parallel path identical to Case 1.

       merge_bias_days = E[T(d)] - T(E[d])

    Reporting the two components separately — rather than calling the whole
    gap "merge bias" — is the technically correct treatment.
    """

    cpm_completion: float           # deterministic CPM finish, mode-based (working days)
    mean_based_completion: float    # T(E[d]): forward pass on simulated mean durations
    simulated_mean: float           # E[completion] from simulation
    simulated_median: float         # P50 from simulation
    mean_shift_days: float          # T(E[d]) - T_CPM : skew + risk-driver uplift
    merge_bias_days: float          # E[T] - T(E[d])  : true merge bias (Jensen gap)
    gap_days: float                 # total: simulated_mean - cpm_completion
    gap_pct: float                  # gap_days / cpm_completion * 100
    analytical_reference: Optional[float] = None  # if available for benchmark
    reference_gap_days: Optional[float] = None
    reference_source: Optional[str] = None


@dataclass
class SimulationResults:
    """
    Complete post-processed results from one simulation run.

    Instantiate by passing a RawSimulationOutput and the Network.
    """

    raw: RawSimulationOutput
    network: Network
    risk_drivers: Optional[list[RiskDriver]] = None

    def __post_init__(self) -> None:
        self._percentile_values: Optional[dict[float, float]] = None
        self._activity_results: Optional[list[ActivityResult]] = None
        self._merge_bias: Optional[MergeBiasResult] = None
        self._corr_ids: Optional[list[str]] = None
        self._corr_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Completion distribution
    # ------------------------------------------------------------------

    @property
    def completion(self) -> np.ndarray:
        """All N completion values (working days)."""
        return self.raw.completion

    def percentile(self, p: float) -> float:
        """Return the p-th percentile of project completion (0 <= p <= 100)."""
        return float(np.percentile(self.raw.completion, p))

    def percentiles(self, ps: Optional[list[float]] = None) -> dict[float, float]:
        """
        Return a dict of {percentile: completion_value} for the requested list.
        Defaults to the percentiles specified in the simulation config.
        """
        if ps is None:
            ps = list(self.raw.config.percentiles)
        return {p: self.percentile(p) for p in ps}

    def probability_of_meeting(self, target_working_days: float) -> float:
        """
        P(completion <= target) — the probability of finishing by the target date.
        """
        return float(np.mean(self.raw.completion <= target_working_days))

    # ------------------------------------------------------------------
    # Criticality index and sensitivity
    # ------------------------------------------------------------------

    @property
    def activity_results(self) -> list[ActivityResult]:
        """Compute and cache per-activity metrics."""
        if self._activity_results is not None:
            return self._activity_results

        cpm_critical_ids = set(self.network.critical_path())
        topo = self.raw.activity_ids
        n_iter = len(self.raw.completion)

        # Criticality index: fraction of iterations where activity is critical
        crit_index = self.raw.critical_path_matrix.mean(axis=0)  # (A,)

        # Sensitivity: Spearman rank correlation of each activity's duration
        # with project completion. Spearman is robust to non-linearity and is
        # the standard "cruciality" measure in SRA (Hulett 2009, §5.3).
        # We correlate sampled durations (not multipliers) with completion.
        sensitivity = np.array([
            stats.spearmanr(
                self.raw.duration_matrix[:, col],
                self.raw.completion,
            ).statistic
            for col in range(len(topo))
        ])

        results = []
        for col, aid in enumerate(topo):
            act = self.network.get(aid)
            results.append(ActivityResult(
                id=aid,
                name=act.name,
                criticality_index=float(crit_index[col]),
                sensitivity=float(sensitivity[col]),
                cpm_duration=act.cpm_duration,
                mean_duration=float(self.raw.duration_matrix[:, col].mean()),
                is_cpm_critical=(aid in cpm_critical_ids),
            ))

        self._activity_results = sorted(
            results, key=lambda r: r.criticality_index, reverse=True
        )
        return self._activity_results

    # ------------------------------------------------------------------
    # Merge bias
    # ------------------------------------------------------------------

    @property
    def merge_bias(self) -> MergeBiasResult:
        if self._merge_bias is not None:
            return self._merge_bias

        cpm_end = self.network.cpm_completion()
        sim_mean = float(self.raw.completion.mean())
        sim_median = float(np.median(self.raw.completion))
        gap = sim_mean - cpm_end
        gap_pct = (gap / cpm_end) * 100.0 if cpm_end > 0 else 0.0

        # Decomposition: forward pass on simulated per-activity MEAN durations
        # (duration_matrix already includes risk-driver multipliers, so the
        # mean shift captures both distribution skew and risk expected uplift).
        # By Jensen's inequality E[T(d)] >= T(E[d]) since T is convex in d;
        # the residual sim_mean - T(E[d]) is the true merge bias.
        mean_durations = {
            aid: float(self.raw.duration_matrix[:, col].mean())
            for col, aid in enumerate(self.raw.activity_ids)
        }
        mean_based_end = self.network.cpm_completion(mean_durations)

        self._merge_bias = MergeBiasResult(
            cpm_completion=cpm_end,
            mean_based_completion=mean_based_end,
            simulated_mean=sim_mean,
            simulated_median=sim_median,
            mean_shift_days=mean_based_end - cpm_end,
            merge_bias_days=sim_mean - mean_based_end,
            gap_days=gap,
            gap_pct=gap_pct,
        )
        return self._merge_bias

    def set_analytical_reference(
        self,
        reference_completion: float,
        source: str,
    ) -> None:
        """
        Attach an analytical or numerical reference value for the merge bias.

        Parameters
        ----------
        reference_completion : The analytically computed E[max(paths)] value.
        source               : Citation string shown in the UI.
        """
        mb = self.merge_bias  # trigger computation first
        mb.analytical_reference = reference_completion
        mb.reference_gap_days = reference_completion - mb.cpm_completion
        mb.reference_source = source

    # ------------------------------------------------------------------
    # Implied correlation matrix
    # ------------------------------------------------------------------

    def compute_implied_correlation(
        self,
    ) -> tuple[list[str], np.ndarray]:
        """
        Return the Pearson correlation matrix of activity durations.

        This reflects both risk-driver-induced correlation AND the structural
        correlation from shared path constraints. For a pure risk-driver
        analysis, this equals the correlation of the multiplier series.
        """
        if self._corr_ids is not None:
            return self._corr_ids, self._corr_matrix  # type: ignore[return-value]

        if self.risk_drivers:
            rd_engine = RiskDriverEngine(self.risk_drivers, self.raw.activity_ids)
            ids, corr = rd_engine.compute_implied_correlation(
                self.raw.multiplier_matrix
            )
        else:
            # Compute from raw durations (structural correlation from shared paths)
            ids = self.raw.activity_ids
            corr = np.corrcoef(self.raw.duration_matrix.T)

        self._corr_ids = ids
        self._corr_matrix = corr
        return ids, corr

    # ------------------------------------------------------------------
    # Convenience summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a flat dict suitable for display or export."""
        pcts = self.percentiles()
        mb = self.merge_bias
        return {
            "n_iterations": len(self.raw.completion),
            "cpm_completion_days": mb.cpm_completion,
            "mean_based_completion_days": mb.mean_based_completion,
            "mean_completion_days": mb.simulated_mean,
            "median_completion_days": mb.simulated_median,
            "mean_shift_days": mb.mean_shift_days,
            "merge_bias_days": mb.merge_bias_days,
            "total_gap_days": mb.gap_days,
            "total_gap_pct": mb.gap_pct,
            **{f"P{int(p)}": v for p, v in pcts.items()},
        }

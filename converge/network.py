"""
CPM network: activities, predecessor relationships (FS / SS), forward pass.

Supported relationship types
-----------------------------
* FS (Finish-to-Start) with optional lag L:
    B cannot start until A finishes + L working days.
    Forward pass: ES(B) = EF(A) + L

* SS (Start-to-Start) with optional lag L:
    B cannot start until A starts + L working days.
    Forward pass: ES(B) = ES(A) + L

Both types are fully vectorised so the Monte Carlo engine runs all N iterations
as NumPy array operations with no Python-level loop over iterations.

Calendar arithmetic convention
-------------------------------
Time is measured in working-day OFFSETS from project start (t=0).
A duration-D activity starting at ES=s occupies the continuous interval [s, s+D].
In calendar terms:
    First working day  = add_working_days(project_start, es)
    Last working day   = add_working_days(project_start, es + D - 1)
                       = add_working_days(project_start, ef - 1)

Example: project_start=Mon 5/29, A duration=2 → A occupies 5/29 and 6/1 (EF=2).
B (FS+0 from A, duration=3) → ES(B)=2, first day=6/2, last day=6/4 (EF=5).
Lag-3 on that FS: ES(B)=5, first day=6/5, last day=6/9.

Design notes
------------
* Constraint dates are INTENTIONALLY stripped. A risk analysis tests whether
  contractual dates are feasible; enforcing hard constraints inside the
  simulation would force success and invalidate the probability estimates.
  See Hulett (2009) §3.4.
* The per-iteration completion is the longest path through sampled durations,
  not the deterministic CPM critical path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np

from converge.distributions import Distribution


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class NetworkError(Exception):
    """Base class for network validation errors."""


class CycleError(NetworkError):
    """Raised when the network contains a cycle."""


class MissingPredecessorError(NetworkError):
    """Raised when an activity references a predecessor that does not exist."""


class InvalidRelationshipError(NetworkError):
    """Raised when an unsupported relationship type is specified."""


SUPPORTED_RELATIONSHIPS = {"FS", "SS"}


# ---------------------------------------------------------------------------
# Predecessor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Predecessor:
    """
    A typed predecessor relationship.

    Parameters
    ----------
    activity_id  : ID of the predecessor activity.
    relationship : 'FS' (Finish-to-Start) or 'SS' (Start-to-Start).
    lag          : Non-negative working-day lag. Default 0.

    String shorthand accepted by parse_predecessor():
        'A1'        → FS, lag=0
        'A1:FS:3'   → FS, lag=3
        'A1:SS'     → SS, lag=0
        'A1:SS:5'   → SS, lag=5
    """

    activity_id: str
    relationship: str = "FS"
    lag: float = 0.0

    def __post_init__(self) -> None:
        rel = self.relationship.upper()
        if rel not in SUPPORTED_RELATIONSHIPS:
            raise InvalidRelationshipError(
                f"Unsupported relationship '{self.relationship}'. "
                f"Use one of: {sorted(SUPPORTED_RELATIONSHIPS)}"
            )
        if self.lag < 0:
            raise ValueError(f"Lag must be non-negative, got {self.lag}.")
        # Normalise relationship to uppercase
        object.__setattr__(self, "relationship", rel)


# ---------------------------------------------------------------------------
# Predecessor string helpers
# ---------------------------------------------------------------------------

def parse_predecessor(pred_str: str) -> Predecessor:
    """
    Parse a single predecessor token such as 'A1', 'A1:FS:3', 'A1:SS:5'.
    """
    parts = [p.strip() for p in pred_str.strip().split(":")]
    activity_id = parts[0]
    relationship = parts[1].upper() if len(parts) > 1 else "FS"
    lag = float(parts[2]) if len(parts) > 2 else 0.0
    return Predecessor(activity_id, relationship, lag)


def parse_predecessors(preds_str: str) -> list[Predecessor]:
    """
    Parse a semicolon-separated predecessor string, e.g. 'A1;B1:SS:3'.
    Returns an empty list for empty / null inputs.
    """
    s = str(preds_str).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return []
    return [parse_predecessor(tok) for tok in s.split(";") if tok.strip()]


def format_predecessor(pred: Predecessor) -> str:
    """Serialise a Predecessor to the compact string form."""
    if pred.relationship == "FS" and pred.lag == 0.0:
        return pred.activity_id
    lag_s = f"{int(pred.lag)}" if pred.lag == int(pred.lag) else f"{pred.lag}"
    if pred.lag == 0.0:
        return f"{pred.activity_id}:{pred.relationship}"
    return f"{pred.activity_id}:{pred.relationship}:{lag_s}"


def format_predecessors(preds: list[Predecessor]) -> str:
    """Serialise a list of Predecessor objects to a semicolon-separated string."""
    return ";".join(format_predecessor(p) for p in preds)


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------

@dataclass
class Activity:
    """
    A single schedule activity.

    Parameters
    ----------
    id            : Unique string identifier (e.g. 'A1', 'IST').
    name          : Human-readable label.
    distribution  : Duration uncertainty distribution.
    predecessors  : List of Predecessor objects OR plain strings (auto-converted
                    to FS+0 for backward compatibility).
    deterministic_duration : Optional override for the CPM forward pass.
                   If None, the distribution mode() is used (industry standard:
                   CPM is built on 'most likely' durations, not means).
    """

    id: str
    name: str
    distribution: Distribution
    predecessors: list = field(default_factory=list)   # list[Union[str, Predecessor]]
    deterministic_duration: Optional[float] = None

    def __post_init__(self) -> None:
        # Normalise plain strings to Predecessor(FS+0) for backward compatibility
        normalised = []
        for p in self.predecessors:
            if isinstance(p, Predecessor):
                normalised.append(p)
            elif isinstance(p, str):
                normalised.append(parse_predecessor(p))
            else:
                raise TypeError(f"Predecessor must be str or Predecessor, got {type(p)}")
        self.predecessors = normalised

    @property
    def cpm_duration(self) -> float:
        """
        Duration used in the deterministic CPM forward pass.

        Industry standard: CPM is built on 'most likely' (mode) durations.
        If deterministic_duration is explicitly set it takes precedence.
        """
        if self.deterministic_duration is not None:
            return self.deterministic_duration
        if hasattr(self.distribution, "mode"):
            return self.distribution.mode()
        return self.distribution.mean()

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Activity) and self.id == other.id


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network:
    """
    Directed acyclic graph of activities with FS and SS relationships.

    Responsibilities
    ----------------
    * Validate structure (no cycles, no missing predecessors).
    * Deterministic CPM forward + backward pass (ES, EF, LS, LF, float).
    * Vectorised Monte Carlo longest-path computation (N iterations at once).
    * Vectorised critical-path identification per iteration.
    """

    def __init__(self, activities: list[Activity]) -> None:
        self._activities: dict[str, Activity] = {a.id: a for a in activities}
        self._validate()
        self._topo_order: list[str] = self._topological_sort()
        # Successor map: {aid: [(succ_id, relationship, lag), ...]}
        self._successors: dict[str, list[tuple[str, str, float]]] = (
            self._build_successor_map()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def activities(self) -> list[Activity]:
        return list(self._activities.values())

    def get(self, activity_id: str) -> Activity:
        if activity_id not in self._activities:
            raise KeyError(f"Activity '{activity_id}' not found in network.")
        return self._activities[activity_id]

    @property
    def topological_order(self) -> list[str]:
        return list(self._topo_order)

    # ------------------------------------------------------------------
    # Deterministic CPM
    # ------------------------------------------------------------------

    def cpm_forward_pass(
        self,
        durations: Optional[dict[str, float]] = None,
    ) -> dict[str, tuple[float, float]]:
        """
        Compute deterministic (ES, EF) for all activities.

        Parameters
        ----------
        durations : Optional override {activity_id: duration}. When omitted,
                    each activity's ``cpm_duration`` (industry-standard
                    most-likely / mode value) is used. Supplying per-activity
                    *mean* durations instead yields T(E[d]) — the deterministic
                    completion of the "expected" schedule, which is the correct
                    lower bound for the simulated mean by Jensen's inequality
                    (project completion is a convex function of durations:
                    a composition of sums and max operations).

        Returns
        -------
        {activity_id: (early_start, early_finish)}

        FS + lag L : ES(B) = EF(A) + L
        SS + lag L : ES(B) = ES(A) + L
        """
        es: dict[str, float] = {}
        ef: dict[str, float] = {}
        for aid in self._topo_order:
            act = self._activities[aid]
            act_es = 0.0
            for pred in act.predecessors:
                pid = pred.activity_id
                if pred.relationship == "FS":
                    act_es = max(act_es, ef[pid] + pred.lag)
                else:  # SS
                    act_es = max(act_es, es[pid] + pred.lag)
            es[aid] = act_es
            dur = durations[aid] if durations is not None else act.cpm_duration
            ef[aid] = act_es + dur
        return {aid: (es[aid], ef[aid]) for aid in self._activities}

    def cpm_completion(
        self,
        durations: Optional[dict[str, float]] = None,
    ) -> float:
        """
        Project completion (working days) under deterministic durations.

        With no argument: standard CPM completion (most-likely durations).
        With a {id: mean_duration} mapping: T(E[d]), the mean-based
        deterministic completion used to isolate true merge bias.
        """
        fp = self.cpm_forward_pass(durations)
        return max(v[1] for v in fp.values())

    def critical_path(self) -> list[str]:
        """
        Identify activities on the deterministic critical path (zero total float).

        Backward pass handles both FS and SS relationships:
            FS succ B, lag L : LF(A) ≤ LS(B) - L
            SS succ B, lag L : LS(A) ≤ LS(B) - L
                             → LF(A) ≤ LF(B) - dur(B) - L + dur(A)
        """
        fp = self.cpm_forward_pass()
        project_end = max(v[1] for v in fp.values())
        ef = {aid: fp[aid][1] for aid in self._activities}

        late_finish: dict[str, float] = {aid: project_end for aid in self._activities}

        for aid in reversed(self._topo_order):
            act = self._activities[aid]
            for (succ_id, rel, lag) in self._successors[aid]:
                succ = self._activities[succ_id]
                ls_succ = late_finish[succ_id] - succ.cpm_duration
                if rel == "FS":
                    late_finish[aid] = min(late_finish[aid], ls_succ - lag)
                else:  # SS
                    # LS(A) + lag ≤ LS(B)  →  LF(A) ≤ LS(B) - lag + dur(A)
                    late_finish[aid] = min(
                        late_finish[aid], ls_succ - lag + act.cpm_duration
                    )

        return [
            aid for aid in self._topo_order
            if abs(late_finish[aid] - ef[aid]) < 1e-9
        ]

    # ------------------------------------------------------------------
    # Vectorised Monte Carlo
    # ------------------------------------------------------------------

    def compute_completion_array(self, duration_matrix: np.ndarray) -> np.ndarray:
        """
        Vectorised longest-path computation for N iterations.

        Parameters
        ----------
        duration_matrix : shape (N, num_activities), columns in topological order.

        Returns
        -------
        Array of shape (N,) — project completion per iteration (working days).

        Time complexity: O(N * E) where E = number of edges.
        The inner loop is over activities (~50), not iterations (10 000+).
        """
        n_iter = duration_matrix.shape[0]
        n_act = len(self._topo_order)
        id_to_col = {aid: i for i, aid in enumerate(self._topo_order)}

        early_start = np.zeros((n_iter, n_act), dtype=np.float64)
        early_finish = np.zeros((n_iter, n_act), dtype=np.float64)

        for col, aid in enumerate(self._topo_order):
            act = self._activities[aid]
            es = np.zeros(n_iter, dtype=np.float64)
            for pred in act.predecessors:
                pcol = id_to_col[pred.activity_id]
                if pred.relationship == "FS":
                    es = np.maximum(es, early_finish[:, pcol] + pred.lag)
                else:  # SS
                    es = np.maximum(es, early_start[:, pcol] + pred.lag)
            early_start[:, col] = es
            early_finish[:, col] = es + duration_matrix[:, col]

        return early_finish.max(axis=1)

    def compute_critical_path_matrix(self, duration_matrix: np.ndarray) -> np.ndarray:
        """
        For each iteration, identify which activities are on the critical path.

        Returns
        -------
        Boolean array (N, num_activities), True where activity is critical.

        Backward pass derivation:
            FS succ B, lag L:  LF(A) ≤ LS(B) - L
            SS succ B, lag L:  LS(A) + L ≤ LS(B)
                               LF(A) - dur(A) + L ≤ LF(B) - dur(B)
                               LF(A) ≤ LF(B) - dur(B) - L + dur(A)
        Both reduce to elementwise NumPy minimum, fully vectorised.
        """
        n_iter = duration_matrix.shape[0]
        n_act = len(self._topo_order)
        id_to_col = {aid: i for i, aid in enumerate(self._topo_order)}

        # --- Forward pass ---
        early_start = np.zeros((n_iter, n_act), dtype=np.float64)
        early_finish = np.zeros((n_iter, n_act), dtype=np.float64)

        for col, aid in enumerate(self._topo_order):
            act = self._activities[aid]
            es = np.zeros(n_iter, dtype=np.float64)
            for pred in act.predecessors:
                pcol = id_to_col[pred.activity_id]
                if pred.relationship == "FS":
                    es = np.maximum(es, early_finish[:, pcol] + pred.lag)
                else:  # SS
                    es = np.maximum(es, early_start[:, pcol] + pred.lag)
            early_start[:, col] = es
            early_finish[:, col] = es + duration_matrix[:, col]

        project_end = early_finish.max(axis=1)  # (N,)

        # --- Backward pass ---
        late_finish = np.tile(project_end[:, None], (1, n_act))  # (N, n_act)

        for col in range(n_act - 1, -1, -1):
            aid = self._topo_order[col]
            dur_A = duration_matrix[:, col]          # (N,)
            for (succ_id, rel, lag) in self._successors[aid]:
                scol = id_to_col[succ_id]
                lf_B = late_finish[:, scol]          # (N,)
                ls_B = lf_B - duration_matrix[:, scol]  # (N,)
                if rel == "FS":
                    constraint = ls_B - lag
                else:  # SS: LF(A) ≤ LS(B) - lag + dur(A)
                    constraint = ls_B - lag + dur_A
                late_finish[:, col] = np.minimum(late_finish[:, col], constraint)

        total_float = late_finish - early_finish
        return total_float < 1e-9

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_successor_map(self) -> dict[str, list[tuple[str, str, float]]]:
        """Build {predecessor_id: [(successor_id, relationship, lag), ...]}."""
        succ: dict[str, list[tuple[str, str, float]]] = {
            aid: [] for aid in self._activities
        }
        for aid, act in self._activities.items():
            for pred in act.predecessors:
                succ[pred.activity_id].append((aid, pred.relationship, pred.lag))
        return succ

    def _validate(self) -> None:
        for act in self._activities.values():
            for pred in act.predecessors:
                if pred.activity_id not in self._activities:
                    raise MissingPredecessorError(
                        f"Activity '{act.id}' references unknown predecessor "
                        f"'{pred.activity_id}'."
                    )

    def _topological_sort(self) -> list[str]:
        """Kahn's algorithm; raises CycleError if the graph is not a DAG."""
        in_degree: dict[str, int] = {aid: 0 for aid in self._activities}
        successors: dict[str, list[str]] = {aid: [] for aid in self._activities}
        for aid, act in self._activities.items():
            for pred in act.predecessors:
                in_degree[aid] += 1
                successors[pred.activity_id].append(aid)

        queue = [aid for aid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for succ in successors[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(order) != len(self._activities):
            raise CycleError(
                "Network contains a cycle. Check predecessor relationships."
            )
        return order

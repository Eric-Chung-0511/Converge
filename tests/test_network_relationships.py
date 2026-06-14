"""
Mathematical validation of FS+lag and SS+lag relationship types.

Tests
-----
1. Deterministic CPM forward pass — FS with lag.
2. Deterministic CPM forward pass — SS with lag.
3. Deterministic CPM backward pass — critical path with SS relationships.
4. Monte Carlo mean — FS+lag serial chain (analytic: E[C] = E[A] + lag + E[B]).
5. Monte Carlo mean — SS+lag (numerical integration reference via scipy).
6. Predecessor string parsing round-trip ('A1:SS:5', 'A1:FS:3', 'A1').
7. Backward-compatibility: plain string predecessors still work.
8. Invalid relationship type raises InvalidRelationshipError.
9. Negative lag raises ValueError.

Mathematical basis
------------------
FS+lag (L): ES(B) = EF(A) + L
SS+lag (L): ES(B) = ES(A) + L

For a serial chain A → (FS+L) → B (no merge):
    completion = dur(A) + L + dur(B)   (every iteration, no branching)
    E[completion] = E[dur(A)] + L + E[dur(B)]   (linearity of expectation)

For A SS+L B with no other predecessors (A and B start simultaneously offset by L):
    EF(A) = dur(A),  EF(B) = L + dur(B)
    completion = max(EF(A), EF(B)) = max(dur(A), L + dur(B))
    E[completion] must be computed numerically: integral of the survival function
    of max(X, L+Y).

The SS case exercises the same merge-bias machinery as parallel FS paths.
"""

import math

import numpy as np
import pytest
from scipy import integrate

from converge.distributions import Triangular, Deterministic
from converge.network import (
    Activity, Network, Predecessor,
    parse_predecessor, parse_predecessors, format_predecessor, format_predecessors,
    InvalidRelationshipError,
)
from converge.engine import SimulationEngine, SimulationConfig
from converge.sampling import SamplingMethod


N_ITER = 50_000
SEED = 2024
TOL_DAYS = 0.4   # Monte Carlo tolerance: 0.4 working days at N=50,000


# ---------------------------------------------------------------------------
# Helper: analytic Triangular CDF
# ---------------------------------------------------------------------------

def _tri_cdf(x: np.ndarray, a: float, m: float, b: float) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.zeros_like(x)
    below_mode = (x >= a) & (x < m)
    above_mode = (x >= m) & (x <= b)
    at_top = x > b
    result[below_mode] = (x[below_mode] - a) ** 2 / ((b - a) * (m - a))
    result[above_mode] = 1.0 - (b - x[above_mode]) ** 2 / ((b - a) * (b - m))
    result[at_top] = 1.0
    return result


def _tri_mean(a: float, m: float, b: float) -> float:
    return (a + m + b) / 3.0


def _E_max_two(a1, m1, b1, shift1, a2, m2, b2, shift2) -> float:
    """
    E[max(X + shift1, Y + shift2)] where X ~ Tri(a1,m1,b1) and Y ~ Tri(a2,m2,b2)
    are independent.

    Using: E[max(U,V)] = int_{-inf}^{inf} [1 - F_U(t)*F_V(t)] dt + lower_bound
    """
    lo = min(a1 + shift1, a2 + shift2)
    hi = max(b1 + shift1, b2 + shift2)

    def survival(t):
        t = np.atleast_1d(t)
        fu = _tri_cdf(t - shift1, a1, m1, b1)
        fv = _tri_cdf(t - shift2, a2, m2, b2)
        return 1.0 - fu * fv

    val, _ = integrate.quad(survival, lo, hi)
    return lo + val


# ---------------------------------------------------------------------------
# 1. CPM forward pass — FS+lag
# ---------------------------------------------------------------------------

class TestFSLagCPM:
    def test_fs_lag_forward_pass(self):
        """
        A (mode=15) → FS+3 → B (mode=8).
        ES(A) = 0,  EF(A) = 15.
        ES(B) = 15 + 3 = 18,  EF(B) = 18 + 8 = 26.
        """
        acts = [
            Activity("A", "Activity A", Triangular(10, 15, 25)),
            Activity("B", "Activity B", Triangular(5, 8, 14),
                     predecessors=[Predecessor("A", "FS", 3)]),
        ]
        net = Network(acts)
        fp = net.cpm_forward_pass()
        es_a, ef_a = fp["A"]
        es_b, ef_b = fp["B"]

        assert es_a == pytest.approx(0.0)
        assert ef_a == pytest.approx(15.0)   # mode of A
        assert es_b == pytest.approx(18.0)   # 15 + lag 3
        assert ef_b == pytest.approx(26.0)   # 18 + mode 8

    def test_fs_lag5_cpm_completion(self):
        """Three-activity chain: A → (FS+5) → B → (FS+2) → C."""
        acts = [
            Activity("A", "A", Triangular(10, 15, 25)),
            Activity("B", "B", Triangular(5,  8,  14),
                     predecessors=[Predecessor("A", "FS", 5)]),
            Activity("C", "C", Triangular(3,  5,   9),
                     predecessors=[Predecessor("B", "FS", 2)]),
        ]
        net = Network(acts)
        # ES(A)=0, EF(A)=15; ES(B)=15+5=20, EF(B)=28; ES(C)=28+2=30, EF(C)=35
        fp = net.cpm_forward_pass()
        assert fp["C"][1] == pytest.approx(35.0)
        assert net.cpm_completion() == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# 2. CPM forward pass — SS+lag
# ---------------------------------------------------------------------------

class TestSSLagCPM:
    def test_ss_lag_forward_pass(self):
        """
        A (mode=15) → SS+3 → B (mode=8).
        ES(A) = 0,  EF(A) = 15.
        ES(B) = ES(A) + 3 = 3,  EF(B) = 3 + 8 = 11.
        Completion = max(15, 11) = 15.
        """
        acts = [
            Activity("A", "Activity A", Triangular(10, 15, 25)),
            Activity("B", "Activity B", Triangular(5, 8, 14),
                     predecessors=[Predecessor("A", "SS", 3)]),
        ]
        net = Network(acts)
        fp = net.cpm_forward_pass()
        es_a, ef_a = fp["A"]
        es_b, ef_b = fp["B"]

        assert es_a == pytest.approx(0.0)
        assert ef_a == pytest.approx(15.0)
        assert es_b == pytest.approx(3.0)    # SS + lag 3
        assert ef_b == pytest.approx(11.0)   # 3 + mode 8
        assert net.cpm_completion() == pytest.approx(15.0)

    def test_ss_zero_lag(self):
        """SS+0: both activities start simultaneously."""
        acts = [
            Activity("A", "A", Triangular(10, 15, 25)),
            Activity("B", "B", Triangular(5,  8,  14),
                     predecessors=[Predecessor("A", "SS", 0)]),
        ]
        net = Network(acts)
        fp = net.cpm_forward_pass()
        assert fp["B"][0] == pytest.approx(0.0)   # starts at project start

    def test_ss_long_b_drives_completion(self):
        """When B is longer, completion is driven by B not A."""
        acts = [
            Activity("A", "A", Triangular(5,  8, 14)),
            Activity("B", "B", Triangular(20, 30, 50),   # much longer
                     predecessors=[Predecessor("A", "SS", 0)]),
        ]
        net = Network(acts)
        fp = net.cpm_forward_pass()
        # ES(B)=0, EF(B)=30; EF(A)=8 → completion=30
        assert net.cpm_completion() == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 3. Critical path with SS relationships
# ---------------------------------------------------------------------------

class TestSSCriticalPath:
    def test_ss_critical_path_identification(self):
        """
        A → SS+3 → B.
        EF(A)=15 > EF(B)=11, so A is on the critical path (float=0), B is not.
        """
        acts = [
            Activity("A", "A", Triangular(10, 15, 25)),
            Activity("B", "B", Triangular(5,  8,  14),
                     predecessors=[Predecessor("A", "SS", 3)]),
        ]
        net = Network(acts)
        cp = net.critical_path()
        assert "A" in cp
        assert "B" not in cp

    def test_ss_both_critical(self):
        """
        A → SS+0 → B where EF(A) == EF(B): both should be critical.
        A: mode=10, EF=10. B: mode=10, SS+0 → ES(B)=0, EF(B)=10.
        """
        acts = [
            Activity("A", "A", Deterministic(10)),
            Activity("B", "B", Deterministic(10),
                     predecessors=[Predecessor("A", "SS", 0)]),
        ]
        net = Network(acts)
        cp = net.critical_path()
        assert "A" in cp
        assert "B" in cp


# ---------------------------------------------------------------------------
# 4. Monte Carlo mean — FS+lag serial chain (analytic reference)
# ---------------------------------------------------------------------------

class TestFSLagMonteCarlo:
    def test_fs_lag_mean_matches_analytic(self):
        """
        A → (FS+L) → B: completion = dur(A) + L + dur(B) per iteration.
        E[completion] = E[dur(A)] + L + E[dur(B)]  (linearity, no branching).
        """
        a_params = (10, 15, 25)
        b_params = (5,  8,  14)
        lag = 7.0
        expected_mean = _tri_mean(*a_params) + lag + _tri_mean(*b_params)

        acts = [
            Activity("A", "A", Triangular(*a_params)),
            Activity("B", "B", Triangular(*b_params),
                     predecessors=[Predecessor("A", "FS", lag)]),
        ]
        net = Network(acts)
        config = SimulationConfig(n_iterations=N_ITER, seed=SEED,
                                  sampling_method=SamplingMethod.RANDOM)
        output = SimulationEngine(net).run(config)
        sim_mean = output.completion.mean()

        assert abs(sim_mean - expected_mean) < TOL_DAYS, (
            f"FS+lag mean: sim={sim_mean:.3f}, analytic={expected_mean:.3f}, "
            f"diff={abs(sim_mean - expected_mean):.3f}"
        )

    def test_fs_zero_lag_matches_fs_chain(self):
        """FS+0 must behave identically to a plain FS predecessor."""
        a_params = (8, 12, 20)
        b_params = (4,  6, 10)

        acts_fs = [
            Activity("A", "A", Triangular(*a_params)),
            Activity("B", "B", Triangular(*b_params),
                     predecessors=[Predecessor("A", "FS", 0)]),
        ]
        acts_plain = [
            Activity("A", "A", Triangular(*a_params)),
            Activity("B", "B", Triangular(*b_params),
                     predecessors=["A"]),
        ]
        cfg = SimulationConfig(n_iterations=N_ITER, seed=SEED)
        out_fs   = SimulationEngine(Network(acts_fs)).run(cfg)
        out_plain = SimulationEngine(Network(acts_plain)).run(cfg)

        assert out_fs.completion.mean() == pytest.approx(out_plain.completion.mean(), abs=1e-9)


# ---------------------------------------------------------------------------
# 5. Monte Carlo mean — SS+lag (numerical integration reference)
# ---------------------------------------------------------------------------

class TestSSLagMonteCarlo:
    def test_ss_lag_mean_matches_numerical(self):
        """
        A (no predecessor) SS+L B:
            EF(A) = dur(A),  EF(B) = L + dur(B)
            completion = max(dur(A), L + dur(B))
        Compared against E[max(X, L+Y)] via numerical integration.
        """
        a_params = (10, 15, 25)
        b_params = (5,   8, 14)
        lag = 5.0

        ref = _E_max_two(*a_params, 0.0, *b_params, lag)

        acts = [
            Activity("A", "A", Triangular(*a_params)),
            Activity("B", "B", Triangular(*b_params),
                     predecessors=[Predecessor("A", "SS", lag)]),
        ]
        net = Network(acts)
        config = SimulationConfig(n_iterations=N_ITER, seed=SEED,
                                  sampling_method=SamplingMethod.RANDOM)
        output = SimulationEngine(net).run(config)
        sim_mean = output.completion.mean()

        assert abs(sim_mean - ref) < TOL_DAYS, (
            f"SS+lag mean: sim={sim_mean:.3f}, numerical={ref:.3f}, "
            f"diff={abs(sim_mean - ref):.3f}"
        )

    def test_ss_zero_lag_mean_matches_numerical(self):
        """SS+0 is max(dur(A), dur(B)): same as two parallel FS paths."""
        a_params = (10, 15, 25)
        b_params = (12, 18, 30)
        lag = 0.0

        ref = _E_max_two(*a_params, 0.0, *b_params, 0.0)

        acts = [
            Activity("A", "A", Triangular(*a_params)),
            Activity("B", "B", Triangular(*b_params),
                     predecessors=[Predecessor("A", "SS", 0)]),
        ]
        net = Network(acts)
        config = SimulationConfig(n_iterations=N_ITER, seed=SEED,
                                  sampling_method=SamplingMethod.RANDOM)
        output = SimulationEngine(net).run(config)
        sim_mean = output.completion.mean()

        assert abs(sim_mean - ref) < TOL_DAYS, (
            f"SS+0 mean: sim={sim_mean:.3f}, numerical={ref:.3f}, "
            f"diff={abs(sim_mean - ref):.3f}"
        )


# ---------------------------------------------------------------------------
# 6. Predecessor string parsing round-trip
# ---------------------------------------------------------------------------

class TestPredecessorParsing:
    def test_plain_id_is_fs_zero(self):
        p = parse_predecessor("A1")
        assert p.activity_id == "A1"
        assert p.relationship == "FS"
        assert p.lag == 0.0

    def test_fs_with_lag(self):
        p = parse_predecessor("A1:FS:3")
        assert p.activity_id == "A1"
        assert p.relationship == "FS"
        assert p.lag == 3.0

    def test_ss_no_lag(self):
        p = parse_predecessor("A1:SS")
        assert p.activity_id == "A1"
        assert p.relationship == "SS"
        assert p.lag == 0.0

    def test_ss_with_lag(self):
        p = parse_predecessor("A1:SS:5")
        assert p.activity_id == "A1"
        assert p.relationship == "SS"
        assert p.lag == 5.0

    def test_relationship_normalised_to_uppercase(self):
        p = parse_predecessor("A1:ss:2")
        assert p.relationship == "SS"

    def test_parse_predecessors_semicolon_list(self):
        preds = parse_predecessors("A1;B1:FS:3;C1:SS:5")
        assert len(preds) == 3
        assert preds[0].activity_id == "A1"
        assert preds[1].lag == 3.0
        assert preds[2].relationship == "SS"

    def test_parse_predecessors_empty(self):
        assert parse_predecessors("") == []
        assert parse_predecessors("nan") == []
        assert parse_predecessors("none") == []

    def test_format_round_trip(self):
        original = "A1;B1:FS:3;C1:SS:5"
        preds = parse_predecessors(original)
        serialised = format_predecessors(preds)
        assert serialised == original

    def test_format_fs_zero_omits_type_and_lag(self):
        p = Predecessor("A1", "FS", 0.0)
        assert format_predecessor(p) == "A1"

    def test_format_fs_nonzero_lag(self):
        p = Predecessor("A1", "FS", 3.0)
        assert format_predecessor(p) == "A1:FS:3"

    def test_format_ss_zero_lag(self):
        p = Predecessor("A1", "SS", 0.0)
        assert format_predecessor(p) == "A1:SS"


# ---------------------------------------------------------------------------
# 7. Backward compatibility — plain string predecessors
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_plain_string_predecessor_accepted(self):
        """Activity.__post_init__ must normalise plain strings to Predecessor(FS+0)."""
        acts = [
            Activity("A", "A", Triangular(10, 15, 25)),
            Activity("B", "B", Triangular(5,  8,  14), predecessors=["A"]),
        ]
        net = Network(acts)
        fp = net.cpm_forward_pass()
        assert fp["B"][0] == pytest.approx(15.0)   # ES(B) = EF(A) = 15

    def test_mixed_string_and_predecessor_objects(self):
        """Mixing str and Predecessor in the same predecessors list must work."""
        acts = [
            Activity("A", "A", Triangular(10, 15, 25)),
            Activity("B", "B", Triangular(5,  8,  14)),
            Activity("C", "C", Triangular(3,  5,   9),
                     predecessors=["A", Predecessor("B", "FS", 2)]),
        ]
        net = Network(acts)
        fp = net.cpm_forward_pass()
        # EF(A)=15, EF(B)+2=10; ES(C)=max(15, 10)=15
        assert fp["C"][0] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# 8 & 9. Validation errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_invalid_relationship_raises(self):
        with pytest.raises(InvalidRelationshipError):
            Predecessor("A1", "FF", 0)   # FF not supported

    def test_negative_lag_raises(self):
        with pytest.raises(ValueError):
            Predecessor("A1", "FS", -1)

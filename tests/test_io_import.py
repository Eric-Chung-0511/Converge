"""
Importer robustness tests (P6-friendly Excel/CSV import path).

These behaviours exist so that a planner can import a minimally prepared
P6 export (Activity ID / Name / Original Duration / Predecessors) through
the provided template without understanding QRA-specific fields:

1. Rows with a blank id are skipped (templates pre-fill formula rows).
2. dist_type blank or missing -> defaults to 'triangular'.
3. Zero-range three-point estimate (a == m == b) -> Deterministic
   (Triangular requires a < b; milestones have zero duration).
4. Explicit 'deterministic' dist_type takes its value from param_m.
5. Missing three-point values raise a clear, actionable error.
"""

import pytest

from converge.io import import_activities_csv
from converge.distributions import Triangular, Deterministic


HEADER = "id,name,predecessors,dist_type,param_a,param_m,param_b\n"


class TestImportRobustness:

    def test_blank_rows_skipped(self):
        csv = HEADER + (
            "A1,Task 1,,triangular,9,10,13\n"
            ",,,,,,\n"          # fully blank (template formula row)
            "A2,Task 2,A1,triangular,18,20,26\n"
            ",,,,,,\n"
        )
        acts = import_activities_csv(csv)
        assert [a.id for a in acts] == ["A1", "A2"]

    def test_dist_type_defaults_to_triangular(self):
        csv = HEADER + "A1,Task 1,,,9,10,13\n"
        acts = import_activities_csv(csv)
        assert isinstance(acts[0].distribution, Triangular)

    def test_dist_type_column_missing_defaults_to_triangular(self):
        csv = "id,name,param_a,param_m,param_b\nA1,Task 1,9,10,13\n"
        acts = import_activities_csv(csv)
        assert isinstance(acts[0].distribution, Triangular)

    def test_zero_range_estimate_becomes_deterministic(self):
        """Milestones: a == m == b must not crash Triangular validation."""
        csv = HEADER + "M1,Milestone,,triangular,0,0,0\n"
        acts = import_activities_csv(csv)
        assert isinstance(acts[0].distribution, Deterministic)
        assert acts[0].distribution.value == 0.0

    def test_explicit_deterministic_uses_param_m(self):
        csv = HEADER + "F1,Fixed Task,,deterministic,,12,\n"
        acts = import_activities_csv(csv)
        assert isinstance(acts[0].distribution, Deterministic)
        assert acts[0].distribution.value == 12.0

    def test_missing_three_point_values_raise_clear_error(self):
        csv = HEADER + "A1,Task 1,,triangular,,10,\n"
        with pytest.raises(Exception) as exc_info:
            import_activities_csv(csv)
        msg = str(exc_info.value)
        assert "param_a" in msg and "param_b" in msg

    def test_p6_style_predecessor_formats(self):
        """Semicolon lists with FS/SS lags parse into relationships."""
        csv = HEADER + (
            "A1,Task 1,,triangular,9,10,13\n"
            "A2,Task 2,A1:FS:5,triangular,9,10,13\n"
            "A3,Task 3,A1:SS:3;A2,triangular,9,10,13\n"
        )
        acts = import_activities_csv(csv)
        a2, a3 = acts[1], acts[2]
        assert (a2.predecessors[0].relationship, a2.predecessors[0].lag) == ("FS", 5.0)
        assert (a3.predecessors[0].relationship, a3.predecessors[0].lag) == ("SS", 3.0)
        assert a3.predecessors[1].activity_id == "A2"

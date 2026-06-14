"""
Import / export for project data (activities, risk drivers, settings).

Formats
-------
* JSON  — full project round-trip (lossless, versioned schema)
* CSV   — activities table and risk-driver register (for spreadsheet users)
* Excel — same as CSV but .xlsx via openpyxl

Schema version
--------------
All JSON exports include a "schema_version" field. Increment this when the
schema changes incompatibly. Import code must check and raise a clear error
on unsupported versions.

Validation
----------
All import functions validate every row/column and raise ImportError with a
message pointing to the offending row and field. Never crash silently.
"""

from __future__ import annotations

import json
import io
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from converge.distributions import make_distribution, Distribution
from converge.network import (
    Activity, Network, Predecessor,
    parse_predecessors, format_predecessors,
)
from converge.risk_drivers import RiskDriver

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# JSON full-project round-trip
# ---------------------------------------------------------------------------

def export_project_json(
    activities: list[Activity],
    risk_drivers: Optional[list[RiskDriver]],
    settings: Optional[dict] = None,
) -> str:
    """
    Serialise the full project to a JSON string.

    Parameters
    ----------
    activities   : List of Activity objects.
    risk_drivers : List of RiskDriver objects (may be None or empty).
    settings     : Optional dict of simulation settings (n_iterations, seed, etc.).

    Returns
    -------
    Formatted JSON string.
    """
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "activities": [_activity_to_dict(a) for a in activities],
        "risk_drivers": [_risk_driver_to_dict(rd) for rd in (risk_drivers or [])],
        "settings": settings or {},
    }
    return json.dumps(data, indent=2)


def import_project_json(json_str: str) -> dict:
    """
    Deserialise a project JSON string.

    Returns
    -------
    Dict with keys 'activities', 'risk_drivers', 'settings'.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ImportError(f"Invalid JSON: {e}") from e

    version = data.get("schema_version", "unknown")
    if version != SCHEMA_VERSION:
        raise ImportError(
            f"Unsupported schema version '{version}'. "
            f"This version of Converge supports '{SCHEMA_VERSION}'."
        )

    activities = [_dict_to_activity(d) for d in data.get("activities", [])]
    risk_drivers = [_dict_to_risk_driver(d) for d in data.get("risk_drivers", [])]
    settings = data.get("settings", {})

    return {
        "activities": activities,
        "risk_drivers": risk_drivers,
        "settings": settings,
    }


# ---------------------------------------------------------------------------
# CSV / Excel tabular import-export
# ---------------------------------------------------------------------------

ACTIVITY_COLUMNS = [
    "id", "name", "predecessors", "dist_type",
    "param_a", "param_m", "param_b",
    "param_mu", "param_sigma",
    "deterministic_duration",
]

RISK_DRIVER_COLUMNS = [
    "id", "name", "probability",
    "impact_type", "impact_a", "impact_b", "impact_m",
    "assigned_activities",
]


def export_activities_csv(activities: list[Activity]) -> str:
    """Return a CSV string of the activity table."""
    df = _activities_to_dataframe(activities)
    return df.to_csv(index=False)


def export_activities_excel(activities: list[Activity]) -> bytes:
    """Return Excel bytes of the activity table."""
    df = _activities_to_dataframe(activities)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Activities")
    return buf.getvalue()


def import_activities_csv(csv_str: str) -> list[Activity]:
    """Parse a CSV string into Activity objects with row-level validation."""
    try:
        df = pd.read_csv(io.StringIO(csv_str))
    except Exception as e:
        raise ImportError(f"Cannot parse CSV: {e}") from e
    return _dataframe_to_activities(df)


def import_activities_excel(excel_bytes: bytes) -> list[Activity]:
    """Parse an Excel file (bytes) into Activity objects."""
    try:
        df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Activities")
    except Exception as e:
        raise ImportError(f"Cannot parse Excel: {e}") from e
    return _dataframe_to_activities(df)


def export_risk_drivers_csv(drivers: list[RiskDriver]) -> str:
    df = _risk_drivers_to_dataframe(drivers)
    return df.to_csv(index=False)


def export_risk_drivers_excel(drivers: list[RiskDriver]) -> bytes:
    df = _risk_drivers_to_dataframe(drivers)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="RiskDrivers")
    return buf.getvalue()


def import_risk_drivers_csv(csv_str: str) -> list[RiskDriver]:
    try:
        df = pd.read_csv(io.StringIO(csv_str))
    except Exception as e:
        raise ImportError(f"Cannot parse CSV: {e}") from e
    return _dataframe_to_risk_drivers(df)


def import_risk_drivers_excel(excel_bytes: bytes) -> list[RiskDriver]:
    try:
        df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="RiskDrivers")
    except Exception as e:
        raise ImportError(f"Cannot parse Excel: {e}") from e
    return _dataframe_to_risk_drivers(df)


# ---------------------------------------------------------------------------
# Template generators
# ---------------------------------------------------------------------------

def get_activity_template_csv() -> str:
    """Return a CSV template with headers and one example row."""
    rows = [
        ACTIVITY_COLUMNS,
        [
            "A1", "Site Preparation", "",
            "triangular", "10", "15", "25", "", "", "",
        ],
        [
            "A2", "Foundation Work", "A1",
            "pert", "20", "30", "50", "", "", "",
        ],
    ]
    return "\n".join(",".join(str(v) for v in row) for row in rows)


def get_risk_driver_template_csv() -> str:
    """Return a CSV template for risk drivers."""
    rows = [
        RISK_DRIVER_COLUMNS,
        [
            "RD-001", "Supplier Delay", "0.3",
            "triangular", "1.0", "1.2", "1.5", "A1;A2",
        ],
    ]
    return "\n".join(",".join(str(v) for v in row) for row in rows)


# ---------------------------------------------------------------------------
# Internal conversion helpers
# ---------------------------------------------------------------------------

def _activity_to_dict(a: Activity) -> dict:
    dist = a.distribution
    dist_type = type(dist).__name__.lower().replace("truncated", "")
    params: dict[str, Any] = {}
    if hasattr(dist, "a"):
        params["a"] = dist.a  # type: ignore[attr-defined]
    if hasattr(dist, "m"):
        params["m"] = dist.m  # type: ignore[attr-defined]
    if hasattr(dist, "b"):
        params["b"] = dist.b  # type: ignore[attr-defined]
    if hasattr(dist, "mu"):
        params["mu"] = dist.mu  # type: ignore[attr-defined]
    if hasattr(dist, "sigma"):
        params["sigma"] = dist.sigma  # type: ignore[attr-defined]
    if hasattr(dist, "lam"):
        params["lam"] = dist.lam  # type: ignore[attr-defined]
    if hasattr(dist, "value"):
        params["value"] = dist.value  # type: ignore[attr-defined]
    return {
        "id": a.id,
        "name": a.name,
        "predecessors": format_predecessors(a.predecessors),
        "dist_type": dist_type,
        "dist_params": params,
        "deterministic_duration": a.deterministic_duration,
    }


def _dict_to_activity(d: dict) -> Activity:
    try:
        dist = make_distribution(d["dist_type"], **d["dist_params"])
        preds_raw = d.get("predecessors", "")
        preds = parse_predecessors(preds_raw) if preds_raw else []
        return Activity(
            id=d["id"],
            name=d["name"],
            distribution=dist,
            predecessors=preds,
            deterministic_duration=d.get("deterministic_duration"),
        )
    except Exception as e:
        raise ImportError(f"Cannot parse activity '{d.get('id', '?')}': {e}") from e


def _risk_driver_to_dict(rd: RiskDriver) -> dict:
    dist = rd.impact_distribution
    dist_type = type(dist).__name__.lower()
    params: dict[str, Any] = {}
    for attr in ("a", "m", "b", "mu", "sigma"):
        if hasattr(dist, attr):
            params[attr] = getattr(dist, attr)
    return {
        "id": rd.id,
        "name": rd.name,
        "probability": rd.probability,
        "impact_dist_type": dist_type,
        "impact_dist_params": params,
        "assigned_activities": rd.assigned_activities,
    }


def _dict_to_risk_driver(d: dict) -> RiskDriver:
    try:
        impact_dist = make_distribution(d["impact_dist_type"], **d["impact_dist_params"])
        return RiskDriver(
            id=d["id"],
            name=d["name"],
            probability=float(d["probability"]),
            impact_distribution=impact_dist,
            assigned_activities=d.get("assigned_activities", []),
        )
    except Exception as e:
        raise ImportError(
            f"Cannot parse risk driver '{d.get('id', '?')}': {e}"
        ) from e


def _activities_to_dataframe(activities: list[Activity]) -> pd.DataFrame:
    rows = []
    for a in activities:
        dist = a.distribution
        row = {
            "id": a.id,
            "name": a.name,
            "predecessors": format_predecessors(a.predecessors),
            "dist_type": type(dist).__name__.lower().replace("truncated", ""),
            # Deterministic stores its value in param_m so Excel/CSV round-trips
            "param_a": getattr(dist, "a", ""),
            "param_m": getattr(dist, "m", getattr(dist, "value", "")),
            "param_b": getattr(dist, "b", ""),
            "param_mu": getattr(dist, "mu", ""),
            "param_sigma": getattr(dist, "sigma", ""),
            "deterministic_duration": a.deterministic_duration or "",
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=ACTIVITY_COLUMNS)


def _blank(value: object) -> bool:
    """True if a cell value is empty / NaN / whitespace-only."""
    s = str(value).strip().lower()
    return s in ("", "nan", "none")


def _dataframe_to_activities(df: pd.DataFrame) -> list[Activity]:
    required = {"id", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ImportError(f"Missing required columns: {missing}")

    activities = []
    for idx, row in df.iterrows():
        # Skip fully blank rows (common when templates pre-fill formula rows
        # beyond the actual data range).
        if _blank(row.get("id")):
            continue

        row_num = idx + 2  # 1-based, skipping header
        try:
            # dist_type is optional: blank or missing defaults to triangular,
            # the standard QRA workhorse. (P6 exports have no such concept —
            # uncertainty shape is a risk-analysis input, not a schedule field.)
            dist_type = str(row.get("dist_type", "triangular")).strip().lower()
            if _blank(dist_type):
                dist_type = "triangular"

            params = {}
            if dist_type in ("triangular", "pert"):
                missing_params = [
                    p for p in ("param_a", "param_m", "param_b")
                    if _blank(row.get(p))
                ]
                if missing_params:
                    raise ImportError(
                        f"Three-point estimate incomplete: {missing_params} is "
                        f"blank. Fill optimistic/most-likely/pessimistic values, "
                        f"or use the template's auto-ranging formulas."
                    )
                a = float(row["param_a"])
                m = float(row["param_m"])
                b = float(row["param_b"])
                if a == m == b:
                    # Zero-range three-point estimate (e.g. a milestone or a
                    # fixed-duration activity): Triangular requires a < b, so
                    # treat as deterministic instead of failing.
                    dist_type = "deterministic"
                    params = {"value": m}
                else:
                    params = {"a": a, "m": m, "b": b}
                    if dist_type == "pert" and "param_lam" in row and not _blank(row["param_lam"]):
                        params["lam"] = float(row["param_lam"])
            elif dist_type in ("uniform",):
                params = {"a": float(row["param_a"]), "b": float(row["param_b"])}
            elif dist_type in ("normal", "normaltruncated"):
                dist_type = "normal"
                params = {"mu": float(row["param_mu"]), "sigma": float(row["param_sigma"])}
            elif dist_type == "deterministic":
                # Value taken from param_m (preferred) or param_a; defaults to 0
                # for pure logic nodes / milestones.
                if not _blank(row.get("param_m")):
                    params = {"value": float(row["param_m"])}
                elif not _blank(row.get("param_a")):
                    params = {"value": float(row["param_a"])}
                else:
                    params = {"value": 0.0}

            dist = make_distribution(dist_type, **params)

            preds_raw = str(row.get("predecessors", "")).strip()
            predecessors = parse_predecessors(preds_raw)

            det_dur_raw = str(row.get("deterministic_duration", "")).strip()
            det_dur = float(det_dur_raw) if det_dur_raw and det_dur_raw.lower() != "nan" else None

            activities.append(Activity(
                id=str(row["id"]).strip(),
                name=str(row["name"]).strip(),
                distribution=dist,
                predecessors=predecessors,
                deterministic_duration=det_dur,
            ))
        except Exception as e:
            raise ImportError(f"Row {row_num}: {e}") from e

    return activities


def _risk_drivers_to_dataframe(drivers: list[RiskDriver]) -> pd.DataFrame:
    rows = []
    for rd in drivers:
        dist = rd.impact_distribution
        row = {
            "id": rd.id,
            "name": rd.name,
            "probability": rd.probability,
            "impact_type": type(dist).__name__.lower(),
            "impact_a": getattr(dist, "a", ""),
            "impact_b": getattr(dist, "b", ""),
            "impact_m": getattr(dist, "m", ""),
            "assigned_activities": ";".join(rd.assigned_activities),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=RISK_DRIVER_COLUMNS)


def _dataframe_to_risk_drivers(df: pd.DataFrame) -> list[RiskDriver]:
    required = {"id", "name", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ImportError(f"Missing required columns: {missing}")

    drivers = []
    for idx, row in df.iterrows():
        if _blank(row.get("id")):
            continue  # skip pre-filled template rows / trailing blanks

        row_num = idx + 2
        try:
            impact_type = str(row.get("impact_type", "triangular")).strip().lower()
            if _blank(impact_type):
                impact_type = "triangular"
            if impact_type in ("triangular", "pert"):
                params = {
                    "a": float(row["impact_a"]),
                    "m": float(row.get("impact_m", row["impact_a"])),
                    "b": float(row["impact_b"]),
                }
            else:
                params = {
                    "a": float(row["impact_a"]),
                    "b": float(row["impact_b"]),
                }
                impact_type = "uniform"

            impact_dist = make_distribution(impact_type, **params)

            assigned_raw = str(row.get("assigned_activities", "")).strip()
            assigned = [a.strip() for a in assigned_raw.split(";") if a.strip()] if assigned_raw else []

            drivers.append(RiskDriver(
                id=str(row["id"]).strip(),
                name=str(row["name"]).strip(),
                probability=float(row["probability"]),
                impact_distribution=impact_dist,
                assigned_activities=assigned,
            ))
        except Exception as e:
            raise ImportError(f"Row {row_num}: {e}") from e

    return drivers


# ---------------------------------------------------------------------------
# Results export
# ---------------------------------------------------------------------------

def export_results_csv(summary: dict, activity_results: list) -> str:
    """Export simulation summary and criticality table as CSV."""
    rows = [["metric", "value"]] + [[k, v] for k, v in summary.items()]
    summary_csv = "\n".join(",".join(str(c) for c in r) for r in rows)

    act_rows = [["id", "name", "criticality_index", "sensitivity",
                 "cpm_duration", "mean_duration", "is_cpm_critical"]]
    for ar in activity_results:
        act_rows.append([
            ar.id, ar.name,
            f"{ar.criticality_index:.4f}",
            f"{ar.sensitivity:.4f}",
            ar.cpm_duration,
            f"{ar.mean_duration:.2f}",
            ar.is_cpm_critical,
        ])
    act_csv = "\n".join(",".join(str(c) for c in r) for r in act_rows)

    return f"# Simulation Summary\n{summary_csv}\n\n# Activity Results\n{act_csv}"

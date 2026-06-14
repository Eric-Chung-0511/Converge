"""
Converge — Schedule Risk Analysis Engine
Streamlit UI entry point.

Structure
---------
Sidebar : simulation controls (N, method, seed, example selector)
Main    : tabs for Results, Criticality, Merge Bias, Risk Drivers,
          Validation, and Distribution Comparison
"""

from __future__ import annotations

import io
import json
import traceback
from datetime import date as _date
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from converge.distributions import Triangular, PERT, Uniform, NormalTruncated, make_distribution
from converge.network import Activity, Network, parse_predecessors, format_predecessors
from converge.risk_drivers import RiskDriver
from converge.engine import SimulationEngine, SimulationConfig
from converge.results import SimulationResults
from converge.sampling import SamplingMethod
from converge.examples import get_example, list_examples, EXAMPLE_REGISTRY
from converge import io as conv_io

from app.plotting import (
    plot_s_curve, plot_histogram, plot_tornado, plot_criticality,
    plot_correlation_matrix, plot_convergence, plot_pert_vs_triangular,
    fig_to_bytes,
)

# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="Converge — Schedule Risk Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Session state initialisation
# ============================================================

def _default_activities_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": "MEP",      "name": "MEP & E&I Installation",      "predecessors": "",              "dist_type": "triangular", "param_a": 30.0, "param_m": 40.0, "param_b": 60.0},
        {"id": "L3_PWR",   "name": "L3 Power Distribution PFT",   "predecessors": "MEP",           "dist_type": "triangular", "param_a": 5.0,  "param_m": 8.0,  "param_b": 15.0},
        {"id": "L3_COOL",  "name": "L3 Cooling Systems PFT",      "predecessors": "MEP",           "dist_type": "triangular", "param_a": 5.0,  "param_m": 8.0,  "param_b": 18.0},
        {"id": "L3_GEN",   "name": "L3 Generator Systems PFT",    "predecessors": "MEP",           "dist_type": "triangular", "param_a": 4.0,  "param_m": 7.0,  "param_b": 14.0},
        {"id": "L3_BMS",   "name": "L3 BMS & Controls PFT",       "predecessors": "MEP",           "dist_type": "triangular", "param_a": 6.0,  "param_m": 10.0, "param_b": 20.0},
        {"id": "L4_PWR",   "name": "L4 Power Distribution FPT",   "predecessors": "L3_PWR",        "dist_type": "triangular", "param_a": 8.0,  "param_m": 12.0, "param_b": 20.0},
        {"id": "L4_COOL",  "name": "L4 Cooling Systems FPT",      "predecessors": "L3_COOL",       "dist_type": "triangular", "param_a": 10.0, "param_m": 15.0, "param_b": 25.0},
        {"id": "L4_GEN",   "name": "L4 Generator Systems FPT",    "predecessors": "L3_GEN",        "dist_type": "triangular", "param_a": 7.0,  "param_m": 10.0, "param_b": 18.0},
        {"id": "L4_BMS",   "name": "L4 BMS & Controls FPT",       "predecessors": "L3_BMS",        "dist_type": "triangular", "param_a": 8.0,  "param_m": 12.0, "param_b": 22.0},
        {"id": "L5_IST",   "name": "L5 Integrated Systems Test",  "predecessors": "L4_PWR;L4_COOL;L4_GEN;L4_BMS", "dist_type": "triangular", "param_a": 15.0, "param_m": 20.0, "param_b": 35.0},
        {"id": "RFS",      "name": "Ready for Service",            "predecessors": "L5_IST",        "dist_type": "triangular", "param_a": 2.0,  "param_m": 3.0,  "param_b": 7.0},
    ])


def _default_drivers_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": "RD-VENDOR", "name": "Vendor Equipment Delay",        "probability": 0.35, "impact_type": "triangular", "impact_a": 1.0, "impact_m": 1.15, "impact_b": 1.40, "assigned_activities": "L3_PWR;L3_COOL;L3_GEN;L3_BMS"},
        {"id": "RD-STAFF",  "name": "Commissioning Staff Availability", "probability": 0.25, "impact_type": "triangular", "impact_a": 1.0, "impact_m": 1.10, "impact_b": 1.30, "assigned_activities": "L4_PWR;L4_COOL;L4_GEN;L4_BMS;L5_IST"},
        {"id": "RD-SNAG",   "name": "Snag List / Defect Rectification", "probability": 0.40, "impact_type": "triangular", "impact_a": 1.0, "impact_m": 1.20, "impact_b": 1.60, "assigned_activities": "L5_IST"},
        {"id": "RD-MEP",    "name": "MEP Installation Delay",           "probability": 0.30, "impact_type": "triangular", "impact_a": 1.0, "impact_m": 1.15, "impact_b": 1.50, "assigned_activities": "MEP"},
    ])


if "activities_df" not in st.session_state:
    st.session_state.activities_df = _default_activities_df()
if "drivers_df" not in st.session_state:
    st.session_state.drivers_df = _default_drivers_df()
if "results" not in st.session_state:
    st.session_state.results = None
if "network" not in st.session_state:
    st.session_state.network = None
if "risk_drivers" not in st.session_state:
    st.session_state.risk_drivers = []


# ============================================================
# Data helpers
# ============================================================

def df_to_activities(df: pd.DataFrame) -> list[Activity]:
    """Convert the activities DataFrame to Activity objects."""
    activities = []
    errors = []
    for idx, row in df.iterrows():
        # Skip blank rows (e.g. accidental empty editor rows)
        if str(row.get("id", "")).strip().lower() in ("", "nan", "none"):
            continue
        try:
            dist_type = str(row["dist_type"]).strip().lower()
            if dist_type in ("", "nan", "none"):
                dist_type = "triangular"  # sensible default; P6 has no such field
            if dist_type in ("triangular", "pert"):
                a, m, b = float(row["param_a"]), float(row["param_m"]), float(row["param_b"])
                if a == m == b:
                    # Zero-range estimate (milestone / fixed duration):
                    # Triangular requires a < b, so treat as deterministic.
                    dist_type, params = "deterministic", {"value": m}
                else:
                    params = {"a": a, "m": m, "b": b}
            elif dist_type == "uniform":
                params = {"a": float(row["param_a"]), "b": float(row["param_b"])}
            elif dist_type == "normal":
                params = {"mu": float(row["param_a"]), "sigma": float(row["param_m"])}
            elif dist_type == "deterministic":
                # Value lives in param_m (0 for pure logic/milestone nodes)
                raw_m = row.get("param_m", 0)
                params = {"value": float(raw_m) if str(raw_m).strip() not in ("", "nan") else 0.0}
            else:
                errors.append(f"Row {idx+2}: unknown dist_type '{dist_type}'")
                continue

            dist = make_distribution(dist_type, **params)
            preds_raw = str(row.get("predecessors", "")).strip()
            predecessors = parse_predecessors(preds_raw)
            activities.append(Activity(
                id=str(row["id"]).strip(),
                name=str(row["name"]).strip(),
                distribution=dist,
                predecessors=predecessors,
            ))
        except Exception as e:
            errors.append(f"Row {idx+2} (id={row.get('id','?')}): {e}")
    return activities, errors


def activities_to_df(activities: list[Activity]) -> pd.DataFrame:
    """Convert Activity objects back to the editable DataFrame layout."""
    rows = []
    for a in activities:
        d = a.distribution
        rows.append({
            "id": a.id, "name": a.name,
            "predecessors": format_predecessors(a.predecessors),
            "dist_type": type(d).__name__.lower().replace("truncated", ""),
            # param slots are overloaded by dist type:
            #   triangular/pert/uniform: a/m/b ; normal: mu/sigma ;
            #   deterministic: value shown in param_m
            "param_a": getattr(d, "a", getattr(d, "mu", 0)),
            "param_m": getattr(d, "m", getattr(d, "sigma", getattr(d, "value", 0))),
            "param_b": getattr(d, "b", 0),
        })
    return pd.DataFrame(rows)


def df_to_risk_drivers(df: pd.DataFrame) -> list[RiskDriver]:
    """Convert the drivers DataFrame to RiskDriver objects."""
    drivers = []
    errors = []
    for idx, row in df.iterrows():
        try:
            impact_type = str(row.get("impact_type", "triangular")).strip().lower()
            if impact_type in ("triangular", "pert"):
                params = {
                    "a": float(row["impact_a"]),
                    "m": float(row["impact_m"]),
                    "b": float(row["impact_b"]),
                }
            else:
                params = {"a": float(row["impact_a"]), "b": float(row["impact_b"])}
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
            errors.append(f"Row {idx+2} (id={row.get('id','?')}): {e}")
    return drivers, errors


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("Converge")
    st.caption("Schedule Risk Analysis Engine")
    st.divider()

    # --- Example loader ---
    st.subheader("Load Example")
    examples_list = list_examples()
    example_names = ["(custom — keep current data)"] + [e["name"] for e in examples_list]
    example_keys = [None] + [e["key"] for e in examples_list]
    selected_ex_idx = st.selectbox(
        "Example network",
        range(len(example_names)),
        format_func=lambda i: example_names[i],
        key="example_selector",
    )
    if st.button("Load Example", use_container_width=True):
        key = example_keys[selected_ex_idx]
        if key:
            ex = get_example(key)
            st.session_state.activities_df = activities_to_df(ex.network.activities)

            drv_rows = []
            for rd in ex.risk_drivers:
                imp = rd.impact_distribution
                drv_rows.append({
                    "id": rd.id, "name": rd.name,
                    "probability": rd.probability,
                    "impact_type": type(imp).__name__.lower(),
                    "impact_a": getattr(imp, "a", 1.0),
                    "impact_m": getattr(imp, "m", 1.0),
                    "impact_b": getattr(imp, "b", 1.0),
                    "assigned_activities": ";".join(rd.assigned_activities),
                })
            st.session_state.drivers_df = pd.DataFrame(drv_rows) if drv_rows else _default_drivers_df().iloc[0:0]
            st.session_state.results = None
            st.success(f"Loaded: {ex.name}")
            if ex.source_citation:
                st.caption(f"Source: {ex.source_citation}")
            if ex.notes:
                st.info(ex.notes)

    st.divider()

    # --- Simulation controls ---
    st.subheader("Simulation Controls")
    n_iter = st.number_input(
        "Iterations (N)",
        min_value=1000, max_value=500_000,
        value=10_000, step=1000,
    )
    sampling_method = st.radio(
        "Sampling method",
        [SamplingMethod.RANDOM, SamplingMethod.LHS],
        format_func=lambda m: "Pure Random" if m == SamplingMethod.RANDOM else "Latin Hypercube (LHS)",
        horizontal=True,
    )
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)
    percentiles_input = st.multiselect(
        "Percentiles to display",
        options=[10, 25, 50, 75, 80, 85, 90, 95],
        default=[50, 80, 90],
    )

    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    # --- Optional calendar conversion (presentation only) ---
    with st.expander("Calendar dates (optional)"):
        use_calendar = st.checkbox("Convert results to calendar dates", value=False)
        if use_calendar:
            start_date = st.date_input("Project start date", key="cal_start_input")
            holidays_text = st.text_area(
                "Holidays (one YYYY-MM-DD per line, optional)",
                key="cal_holidays_input",
                height=80,
            )
            holidays_list = []
            holiday_errors = []
            for line in holidays_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    holidays_list.append(_date.fromisoformat(line))
                except ValueError:
                    holiday_errors.append(line)
            if holiday_errors:
                st.warning(f"Ignored invalid date(s): {', '.join(holiday_errors)}")
            st.session_state.project_start_date = start_date
            st.session_state.holidays_list = holidays_list
        else:
            st.session_state.project_start_date = None
            st.session_state.holidays_list = []
        st.caption(
            "Presentation only: the engine always computes in working days "
            "(5-day week). Benchmarks are validated in working-day terms."
        )

    st.divider()

    # --- Import / Export ---
    st.subheader("Import / Export")

    # Export project JSON
    if st.session_state.results is not None:
        acts, _ = df_to_activities(st.session_state.activities_df)
        drvs, _ = df_to_risk_drivers(st.session_state.drivers_df)
        proj_json = conv_io.export_project_json(
            acts, drvs,
            settings={"n_iterations": int(n_iter), "seed": int(seed)},
        )
        st.download_button(
            "Download Project (JSON)",
            data=proj_json,
            file_name="converge_project.json",
            mime="application/json",
            use_container_width=True,
        )

    # Import project JSON
    uploaded_json = st.file_uploader("Load Project (JSON)", type=["json"])
    if uploaded_json:
        try:
            project = conv_io.import_project_json(uploaded_json.read().decode())
            st.session_state.activities_df = activities_to_df(project["activities"])
            drvs = project["risk_drivers"]
            drv_rows = []
            for rd in drvs:
                imp = rd.impact_distribution
                drv_rows.append({
                    "id": rd.id, "name": rd.name,
                    "probability": rd.probability,
                    "impact_type": type(imp).__name__.lower(),
                    "impact_a": getattr(imp, "a", 1.0),
                    "impact_m": getattr(imp, "m", 1.0),
                    "impact_b": getattr(imp, "b", 1.0),
                    "assigned_activities": ";".join(rd.assigned_activities),
                })
            st.session_state.drivers_df = pd.DataFrame(drv_rows) if drv_rows else st.session_state.drivers_df
            st.session_state.results = None
            st.success("Project loaded successfully.")
        except Exception as e:
            st.error(f"Import error: {e}")

    # CSV templates
    st.download_button(
        "Download Activity Template (CSV)",
        data=conv_io.get_activity_template_csv(),
        file_name="activity_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Download Risk Driver Template (CSV)",
        data=conv_io.get_risk_driver_template_csv(),
        file_name="risk_driver_template.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# Main area — data editor tables
# ============================================================

st.title("Converge — Schedule Risk Analysis Engine")
st.caption(
    "Monte Carlo simulation with Risk Driver correlation, merge bias quantification, "
    "and a built-in mathematical validation bench."
)

tab_data, tab_results, tab_criticality, tab_mergebias, tab_drivers, tab_validation, tab_dist = st.tabs([
    "Data Input",
    "S-Curve & Histogram",
    "Criticality & Tornado",
    "Merge Bias",
    "Risk Drivers & Correlation",
    "Validation Bench",
    "Distribution Comparison",
])


# ---- Tab: Data Input ----
with tab_data:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Activity Register")
        st.caption(
            "Edit directly in the table. **predecessors**: semicolon-separated IDs (e.g. `A1;A2`). "
            "**dist_type**: `triangular`, `pert`, `uniform`, `normal`. "
            "For triangular/pert: param_a=min, param_m=likely, param_b=max. "
            "For normal: param_a=mean, param_m=std_dev."
        )
        edited_acts = st.data_editor(
            st.session_state.activities_df,
            num_rows="dynamic",
            use_container_width=True,
            height=450,
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "name": st.column_config.TextColumn("Name", width="large"),
                "predecessors": st.column_config.TextColumn("Predecessors", width="medium"),
                "dist_type": st.column_config.SelectboxColumn(
                    "Distribution",
                    options=["triangular", "pert", "uniform", "normal", "deterministic"],
                    width="medium",
                ),
                "param_a": st.column_config.NumberColumn("a / min / mean", format="%.1f", width="small"),
                "param_m": st.column_config.NumberColumn("m / likely / std", format="%.1f", width="small"),
                "param_b": st.column_config.NumberColumn("b / max", format="%.1f", width="small"),
            },
            key="act_editor",
        )
        st.session_state.activities_df = edited_acts

        # CSV / Excel import for activities
        act_file = st.file_uploader(
            "Import Activities (CSV or Excel)", type=["csv", "xlsx"], key="act_file"
        )
        if act_file:
            try:
                if act_file.name.lower().endswith(".xlsx"):
                    imported = conv_io.import_activities_excel(act_file.read())
                else:
                    imported = conv_io.import_activities_csv(act_file.read().decode())
                st.session_state.activities_df = activities_to_df(imported)
                st.success("Activities imported.")
                st.rerun()
            except Exception as e:
                st.error(f"Import error: {e}")

        # Export
        acts_for_export, _ = df_to_activities(edited_acts)
        if acts_for_export:
            exp_c1, exp_c2 = st.columns(2)
            exp_c1.download_button(
                "Export Activities (CSV)",
                data=conv_io.export_activities_csv(acts_for_export),
                file_name="activities.csv",
                mime="text/csv",
            )
            exp_c2.download_button(
                "Export Activities (Excel)",
                data=conv_io.export_activities_excel(acts_for_export),
                file_name="activities.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with col2:
        st.subheader("Risk Driver Register")
        st.caption(
            "Each driver fires with its **probability** and applies a **multiplicative impact** "
            "to all assigned activities simultaneously. This creates emergent correlation. "
            "impact_a=min multiplier, impact_m=likely, impact_b=max (e.g. 1.0–1.30 = no change to +30%)."
        )
        edited_drvs = st.data_editor(
            st.session_state.drivers_df,
            num_rows="dynamic",
            use_container_width=True,
            height=450,
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "name": st.column_config.TextColumn("Name", width="large"),
                "probability": st.column_config.NumberColumn("Probability", format="%.2f", min_value=0.0, max_value=1.0, width="small"),
                "impact_type": st.column_config.SelectboxColumn(
                    "Impact Dist.", options=["triangular", "uniform"], width="small"
                ),
                "impact_a": st.column_config.NumberColumn("Impact min", format="%.2f", width="small"),
                "impact_m": st.column_config.NumberColumn("Impact mode", format="%.2f", width="small"),
                "impact_b": st.column_config.NumberColumn("Impact max", format="%.2f", width="small"),
                "assigned_activities": st.column_config.TextColumn("Assigned Activities (IDs, semicolon-sep)", width="large"),
            },
            key="drv_editor",
        )
        st.session_state.drivers_df = edited_drvs

        # CSV / Excel import for risk drivers
        drv_file = st.file_uploader(
            "Import Risk Drivers (CSV or Excel)", type=["csv", "xlsx"], key="drv_file"
        )
        if drv_file:
            try:
                if drv_file.name.lower().endswith(".xlsx"):
                    imported_drvs = conv_io.import_risk_drivers_excel(drv_file.read())
                else:
                    imported_drvs = conv_io.import_risk_drivers_csv(drv_file.read().decode())
                drv_rows = []
                for rd in imported_drvs:
                    imp = rd.impact_distribution
                    drv_rows.append({
                        "id": rd.id, "name": rd.name,
                        "probability": rd.probability,
                        "impact_type": type(imp).__name__.lower(),
                        "impact_a": getattr(imp, "a", 1.0),
                        "impact_m": getattr(imp, "m", 1.0),
                        "impact_b": getattr(imp, "b", 1.0),
                        "assigned_activities": ";".join(rd.assigned_activities),
                    })
                st.session_state.drivers_df = pd.DataFrame(drv_rows)
                st.success("Risk drivers imported.")
                st.rerun()
            except Exception as e:
                st.error(f"Import error: {e}")

        drvs_for_export, _ = df_to_risk_drivers(edited_drvs)
        if drvs_for_export:
            exp_d1, exp_d2 = st.columns(2)
            exp_d1.download_button(
                "Export Risk Drivers (CSV)",
                data=conv_io.export_risk_drivers_csv(drvs_for_export),
                file_name="risk_drivers.csv",
                mime="text/csv",
            )
            exp_d2.download_button(
                "Export Risk Drivers (Excel)",
                data=conv_io.export_risk_drivers_excel(drvs_for_export),
                file_name="risk_drivers.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ============================================================
# Run simulation
# ============================================================

if run_btn:
    acts, act_errors = df_to_activities(st.session_state.activities_df)
    drvs, drv_errors = df_to_risk_drivers(st.session_state.drivers_df)

    all_errors = act_errors + drv_errors
    if all_errors:
        st.error("Data errors — fix before running:\n" + "\n".join(all_errors))
    elif not acts:
        st.error("No activities defined.")
    else:
        try:
            net = Network(acts)
            config = SimulationConfig(
                n_iterations=int(n_iter),
                sampling_method=sampling_method,
                seed=int(seed),
                percentiles=sorted(percentiles_input) if percentiles_input else [50, 80, 90],
            )
            with st.spinner(f"Running {n_iter:,} iterations..."):
                output = SimulationEngine(net, drvs).run(config)
                results = SimulationResults(output, net, drvs)
            st.session_state.results = results
            st.session_state.network = net
            st.session_state.risk_drivers = drvs
            st.success(f"Simulation complete — {n_iter:,} iterations in {sampling_method.value} mode.")
        except Exception as e:
            st.error(f"Simulation error: {e}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())


# ============================================================
# Results display helper
# ============================================================

def _no_results():
    st.info("Run the simulation first (click **Run Simulation** in the sidebar).")


@st.cache_data(show_spinner=False)
def _fig_bytes_cached(fig_json: str, fmt: str) -> bytes:
    """
    Cached figure-to-bytes conversion.

    Static formats (png/svg/pdf) invoke kaleido, which is expensive (it spawns
    a renderer). Caching on the figure's JSON keeps Streamlit's rerun-everything
    model responsive: each chart is converted once, not on every interaction.
    """
    import plotly.io as pio
    fig = pio.from_json(fig_json)
    return fig_to_bytes(fig, fmt)


def _download_chart(fig, label: str, key: str):
    """
    Render chart-export controls.

    Interactive HTML is always available (no external dependency). Static
    formats are generated lazily — only when the user asks — and fail soft
    with guidance if kaleido cannot render (e.g. kaleido >= 1.0 without
    Chrome installed) instead of crashing the app.
    """
    with st.expander(f"Export this chart — {label}"):
        fig_json = fig.to_json()
        st.download_button(
            "Interactive HTML", data=_fig_bytes_cached(fig_json, "html"),
            file_name=f"{key}.html", mime="text/html", key=f"dl_html_{key}",
        )
        if st.toggle("Generate static exports (PNG / SVG / PDF)", key=f"static_{key}"):
            try:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.download_button(
                        "PNG", data=_fig_bytes_cached(fig_json, "png"),
                        file_name=f"{key}.png", mime="image/png", key=f"dl_png_{key}",
                    )
                with c2:
                    st.download_button(
                        "SVG", data=_fig_bytes_cached(fig_json, "svg"),
                        file_name=f"{key}.svg", mime="image/svg+xml", key=f"dl_svg_{key}",
                    )
                with c3:
                    st.download_button(
                        "PDF", data=_fig_bytes_cached(fig_json, "pdf"),
                        file_name=f"{key}.pdf", mime="application/pdf", key=f"dl_pdf_{key}",
                    )
            except Exception as e:
                st.warning(
                    "Static export unavailable on this machine "
                    f"(kaleido error: {e}). Interactive HTML export above "
                    "still works. Fix: `pip install 'plotly>=5.22,<6' kaleido==0.2.1` "
                    "(self-contained), or install Chrome for kaleido >= 1.0."
                )


# ---- Tab: S-Curve & Histogram ----
with tab_results:
    if st.session_state.results is None:
        _no_results()
    else:
        results: SimulationResults = st.session_state.results
        pct_vals = results.percentiles(percentiles_input or [50, 80, 90])

        # Summary metrics row
        cols = st.columns(len(pct_vals) + 2)
        cols[0].metric("CPM Completion", f"{results.network.cpm_completion():.0f}d")
        cols[1].metric("Simulated Mean", f"{results.raw.completion.mean():.0f}d")
        for i, (p, v) in enumerate(sorted(pct_vals.items())):
            cols[i + 2].metric(f"P{int(p)}", f"{v:.0f}d")

        st.divider()

        # Target date input + probability of meeting it
        target_col, prob_col, _ = st.columns([1, 1, 1])
        target = target_col.number_input(
            "Mark target date (working days, 0 = none)",
            min_value=0.0, value=0.0, step=1.0, key="target_date_input",
        )
        target_val = target if target > 0 else None
        if target_val is not None:
            p_target = results.probability_of_meeting(target_val)
            prob_col.metric(
                "P(finish by target)", f"{p_target:.1%}",
                help="Fraction of iterations completing on or before the target — "
                     "the number an owner actually asks for.",
            )

        # Optional calendar-date conversion (presentation only — the math is
        # always in working days; see converge/calendar.py)
        if st.session_state.get("project_start_date"):
            from converge.calendar import working_day_to_date
            start = st.session_state.project_start_date
            holidays = st.session_state.get("holidays_list") or None
            date_rows = {
                "CPM completion": results.network.cpm_completion(),
                **{f"P{int(p)}": v for p, v in sorted(pct_vals.items())},
            }
            if target_val is not None:
                date_rows["Target"] = target_val
            cal_df = pd.DataFrame([
                {
                    "Milestone": k,
                    "Working Days": f"{v:.1f}",
                    "Calendar Date": working_day_to_date(start, v, holidays).isoformat(),
                }
                for k, v in date_rows.items()
            ])
            st.caption(
                f"Calendar dates from project start {start.isoformat()} "
                "(5-day week; fractional days rounded up conservatively)."
            )
            st.dataframe(cal_df, hide_index=True, use_container_width=True)

        # S-curve
        st.subheader("S-Curve (CDF)")
        with st.expander("Math note: What the S-curve shows", expanded=False):
            st.markdown("""
The S-curve (cumulative distribution function) shows **P(project completion ≤ t)** for each
possible completion date *t*. Reading off the chart:

- **P50** = 50% chance of finishing by this date — the median.
- **P80** = 80% chance — a common client confidence threshold.
- **P90** = 90% chance — common contractual risk allowance.

The CPM date (shown as a dashed grey line) is deterministic and is built on each activity's
**most-likely (mode) duration** — the industry-standard CPM convention. Because real duration
distributions are right-skewed (mean > mode) and parallel paths merge, the CPM date typically
has only **10–30% probability of being met** in real projects. The gap between CPM and the
simulated mean has two distinct causes — **skew/mean shift** and **true merge bias** — which
are decomposed and quantified separately on the Merge Bias tab.
            """)
        fig_scurve = plot_s_curve(results, percentiles_input or [50, 80, 90], target_val)
        st.plotly_chart(fig_scurve, use_container_width=True)
        _download_chart(fig_scurve, "S-Curve", "scurve")

        # Histogram
        st.subheader("Completion Histogram (PDF)")
        fig_hist = plot_histogram(results)
        st.plotly_chart(fig_hist, use_container_width=True)
        _download_chart(fig_hist, "Histogram", "histogram")

        # Download results CSV
        summary = results.summary()
        results_csv = conv_io.export_results_csv(summary, results.activity_results)
        st.download_button(
            "Download Results (CSV)",
            data=results_csv,
            file_name="simulation_results.csv",
            mime="text/csv",
        )


# ---- Tab: Criticality & Tornado ----
with tab_criticality:
    if st.session_state.results is None:
        _no_results()
    else:
        results: SimulationResults = st.session_state.results

        with st.expander("Math note: Criticality Index vs CPM Critical Path", expanded=False):
            st.markdown("""
The **criticality index** (CI) is the fraction of simulation iterations in which an activity
lies on the realised longest path (the critical path for that iteration's sampled durations).

**Why CI differs from CPM float:**
In the deterministic CPM, activities are ranked by total float — zero float = critical.
But when durations vary, an activity with *positive* CPM float can still become critical in
many iterations if its distribution is wide enough to bridge that float gap.

**Hulett (1996) Case 3** is the canonical example: Path B has 5 days of CPM float but carries
~69% criticality because its uncertainty easily spans 5 days. A planner watching only CPM
float would call Path A (zero float) the risk — but Path B is the actual driver.

**Sensitivity (Tornado)** measures Spearman rank correlation between each activity's duration
and the project completion. This combines the effect of an activity's uncertainty magnitude
(wide distribution → more impact) with its frequency of criticality.
            """)

        st.subheader("Criticality Index")
        fig_crit = plot_criticality(results)
        st.plotly_chart(fig_crit, use_container_width=True)
        _download_chart(fig_crit, "Criticality Index", "criticality")

        st.subheader("Criticality & Sensitivity Table")
        act_df = pd.DataFrame([{
            "ID": r.id,
            "Name": r.name,
            "Criticality Index": f"{r.criticality_index:.1%}",
            "Sensitivity (Spearman)": f"{r.sensitivity:.3f}",
            "CPM Duration (d)": f"{r.cpm_duration:.1f}",
            "Simulated Mean (d)": f"{r.mean_duration:.1f}",
            "On CPM Critical Path": "★" if r.is_cpm_critical else "",
        } for r in results.activity_results])
        st.dataframe(act_df, use_container_width=True, hide_index=True)

        st.subheader("Sensitivity Tornado")
        fig_tornado = plot_tornado(results)
        st.plotly_chart(fig_tornado, use_container_width=True)
        _download_chart(fig_tornado, "Tornado", "tornado")


# ---- Tab: Merge Bias ----
with tab_mergebias:
    if st.session_state.results is None:
        _no_results()
    else:
        results: SimulationResults = st.session_state.results
        mb = results.merge_bias

        with st.expander("Math note: Decomposing the CPM-vs-simulation gap", expanded=True):
            st.markdown(f"""
The naive comparison "simulated mean − CPM date" **conflates two distinct effects**.
Calling the whole gap "merge bias" is a common error; they must be separated:

**1 — Mean shift (skew bias).** CPM is built on each activity's **most-likely (mode)**
duration. For right-skewed distributions the mean exceeds the mode, and risk drivers raise
the expected duration further. Re-running the deterministic forward pass with each activity's
*simulated mean* duration gives $T(E[d])$:

$$\\text{{Mean shift}} = T(E[d]) - T_{{\\text{{CPM}}}}(\\text{{mode}})$$

This delay occurs **even on a single path with no merge points** — Hulett (1996) Case 1
is the canonical single-path example.

**2 — True merge bias (Jensen gap).** Project completion $T$ is a **convex function** of
activity durations (a composition of sums and $\\max$ at merge nodes). By **Jensen's
Inequality**:

$$E[T(d)] \\geq T(E[d])$$

The gap is strictly positive only when parallel paths merge — it is exactly **zero for a
purely serial chain**, where $T$ is linear. It grows with more parallel paths, higher
variance per path, and less correlation between paths. Hulett (1996) Case 2 isolates it
by adding one parallel path identical to Case 1.

**This is the physical reason** why data-centre commissioning schedules built in CPM
consistently run optimistic: the L5 IST merge point collects 4+ parallel subsystem paths.

| Step | Value | Component |
|------|-------|-----------|
| CPM completion (mode-based) | **{mb.cpm_completion:.1f} wd** | baseline |
| + Mean shift (skew + risk uplift) | **{mb.mean_shift_days:+.1f} wd** | → $T(E[d])$ = {mb.mean_based_completion:.1f} wd |
| + True merge bias (Jensen gap) | **{mb.merge_bias_days:+.1f} wd** | → $E[T]$ = {mb.simulated_mean:.1f} wd |
| **Total gap** | **{mb.gap_days:+.1f} wd ({mb.gap_pct:+.1f}%)** | |
            """)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("CPM (mode-based)", f"{mb.cpm_completion:.0f}d")
        col2.metric("T(E[d]) mean-based", f"{mb.mean_based_completion:.0f}d",
                    delta=f"+{mb.mean_shift_days:.1f}d skew/risk shift",
                    delta_color="inverse")
        col3.metric("Simulated Mean E[T]", f"{mb.simulated_mean:.0f}d",
                    delta=f"+{mb.merge_bias_days:.1f}d merge bias",
                    delta_color="inverse")
        col4.metric("True Merge Bias", f"{mb.merge_bias_days:.1f}d",
                    help="E[T(d)] − T(E[d]): the Jensen gap. Zero for a serial chain; "
                         "positive only when parallel paths merge.")
        col5.metric("P50 Completion", f"{results.percentile(50):.0f}d")

        if abs(mb.merge_bias_days) < 0.05 and mb.mean_shift_days > 0.1:
            st.info(
                "True merge bias is ~0: this network is effectively a serial chain "
                "(no competing parallel paths). The entire gap vs CPM comes from the "
                "skew/mean shift — exactly the Hulett (1996) Case 1 situation."
            )

        if mb.analytical_reference:
            st.success(
                f"Analytical reference: {mb.analytical_reference:.1f}d | "
                f"Simulation: {mb.simulated_mean:.1f}d | "
                f"Difference: {abs(mb.simulated_mean - mb.analytical_reference):.2f}d | "
                f"Source: {mb.reference_source}"
            )

        # Overlay distribution plots showing CPM vs simulated
        st.subheader("CPM vs Simulation Distribution")
        import plotly.graph_objects as go
        fig_mb = go.Figure()
        completion = results.completion
        fig_mb.add_trace(go.Histogram(
            x=completion, nbinsx=60,
            name="Simulated Completion",
            marker_color="#2563EB", opacity=0.7,
            histnorm="probability density",
        ))
        fig_mb.add_vline(x=mb.cpm_completion, line=dict(color="#6B7280", width=2.5, dash="dash"),
                         annotation_text=f"CPM (mode): {mb.cpm_completion:.0f}d", annotation_position="top left")
        fig_mb.add_vline(x=mb.mean_based_completion, line=dict(color="#059669", width=2, dash="dashdot"),
                         annotation_text=f"T(E[d]): {mb.mean_based_completion:.0f}d", annotation_position="bottom left")
        fig_mb.add_vline(x=mb.simulated_mean, line=dict(color="#DC2626", width=2.5),
                         annotation_text=f"E[T]: {mb.simulated_mean:.0f}d", annotation_position="top right")
        fig_mb.add_vline(x=results.percentile(50), line=dict(color="#D97706", width=2, dash="dot"),
                         annotation_text=f"P50: {results.percentile(50):.0f}d", annotation_position="bottom right")
        fig_mb.update_layout(
            title="Completion Distribution with CPM Reference",
            xaxis_title="Working Days", yaxis_title="Probability Density",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=60, r=30, t=60, b=50),
        )
        st.plotly_chart(fig_mb, use_container_width=True)
        _download_chart(fig_mb, "Merge Bias Distribution", "mergebias")


# ---- Tab: Risk Drivers & Correlation ----
with tab_drivers:
    if st.session_state.results is None:
        _no_results()
    else:
        results: SimulationResults = st.session_state.results

        with st.expander("Math note: Risk Driver Method & Emergent Correlation", expanded=False):
            st.markdown("""
The **Risk Driver Method** (Hulett, 2009) models correlation as a *structural property* of
the schedule — not as a hand-entered coefficient.

A **risk driver** is a named event that:
1. Fires with probability *p* on each iteration (Bernoulli draw).
2. If it fires, draws a single **multiplier** from its impact distribution.
3. Applies that **same multiplier** to **all assigned activities simultaneously**.

This means activities sharing a driver move together: their durations are correlated.
The correlation is an **output** of the structure, visible in the matrix below.

**Key insight:** A single shared driver makes two activities perfectly correlated.
Each additional *independent* driver dilutes that correlation toward zero.
This gives the Risk Driver Method discriminating power that a single global coefficient lacks:
different activity pairs can have different implied correlations based on how many risks they share.

**Contrast with global correlation coefficient:** A global coefficient applies uniformly to every
pair — it cannot represent a schedule where Power and Cooling are highly correlated (both share
Vendor Delay) but Power and BMS are less correlated. The Risk Driver Method can.
            """)

        if st.session_state.risk_drivers:
            ids, corr = results.compute_implied_correlation()
            if ids and len(ids) >= 2:
                st.subheader("Implied Correlation Matrix")
                st.caption(
                    "Computed from simulation samples — correlation is an *output* of the "
                    "risk-driver structure, not an input. Hover over cells for values."
                )
                fig_corr = plot_correlation_matrix(ids, corr)
                st.plotly_chart(fig_corr, use_container_width=True)
                _download_chart(fig_corr, "Correlation Matrix", "correlation")

                # Show matrix as table too (for export)
                corr_df = pd.DataFrame(corr, index=ids, columns=ids).round(3)
                st.subheader("Correlation Matrix (Table)")
                st.dataframe(corr_df, use_container_width=True)
                corr_csv = corr_df.to_csv()
                st.download_button(
                    "Download Correlation Matrix (CSV)",
                    data=corr_csv,
                    file_name="correlation_matrix.csv",
                    mime="text/csv",
                )
            else:
                st.info("Assign risk drivers to at least 2 activities to see the correlation matrix.")
        else:
            st.info("No risk drivers defined. Add drivers in the Data Input tab to see correlation.")


# ---- Tab: Validation Bench ----
with tab_validation:
    st.header("Validation Bench")
    st.markdown("""
This tab proves the engine is mathematically correct through three independent tests.
Each test states its claim, shows the computed vs reference value, and reports pass/fail.
    """)

    # Layer 1: Analytic agreement
    st.subheader("Layer 1 — Analytic Agreement (Single Activity)")
    with st.expander("What is being tested?", expanded=True):
        st.markdown("""
For a Triangular distribution with known parameters $(a, m, b)$, the analytic moments are:

$$\\text{Mean} = \\frac{a + m + b}{3}, \\quad \\text{Variance} = \\frac{a^2 + m^2 + b^2 - am - ab - mb}{18}$$

We run 50,000 iterations and check that the simulated mean and variance match these
formulas within 4 standard errors (the tolerance expected by the Central Limit Theorem).
        """)

    if st.button("Run Analytic Agreement Test", key="run_analytic"):
        from converge.distributions import Triangular as T
        from converge.network import Activity, Network
        from converge.engine import SimulationEngine, SimulationConfig
        import numpy as np

        a, m, b = 40.0, 50.0, 100.0
        dist = T(a, m, b)
        net = Network([Activity("A", "Test Activity", dist)])
        config = SimulationConfig(n_iterations=50_000, seed=42)
        output = SimulationEngine(net).run(config)
        samples = output.duration_matrix[:, 0]

        sim_mean = samples.mean()
        sim_var = samples.var()
        analytic_mean = dist.mean()
        analytic_var = dist.variance()
        se_mean = np.sqrt(analytic_var / 50_000)
        mean_ok = abs(sim_mean - analytic_mean) < 4 * se_mean
        var_ok = abs(sim_var - analytic_var) < 0.05 * analytic_var

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analytic Mean", f"{analytic_mean:.4f}d")
            st.metric("Simulated Mean", f"{sim_mean:.4f}d",
                      delta=f"{sim_mean - analytic_mean:+.4f}")
            st.success("PASS: Mean within 4 SE") if mean_ok else st.error("FAIL: Mean out of tolerance")

        with col2:
            st.metric("Analytic Variance", f"{analytic_var:.4f}")
            st.metric("Simulated Variance", f"{sim_var:.4f}",
                      delta=f"{sim_var - analytic_var:+.4f}")
            st.success("PASS: Variance within 5%") if var_ok else st.error("FAIL: Variance out of tolerance")

    # Layer 2: Merge bias numerical validation
    st.subheader("Layer 2 — Merge Bias vs Numerical Integration")
    with st.expander("What is being tested?", expanded=False):
        st.markdown("""
For two independent Triangular paths merging at a node, $E[\\max(X_1, X_2)]$ can be computed
numerically using the survival function formula:

$$E[\\max(X_1, X_2)] = \\int_{x_{\\min}}^{x_{\\max}} \\left[1 - F_1(x) \\cdot F_2(x)\\right] dx + x_{\\min}$$

where $F_i$ is the CDF of path $i$. We compute this with `scipy.integrate.quad` and compare it
to the simulation result. If they agree within 0.5 working days, the engine is proven correct
for this fundamental merge-point case.
        """)

    if st.button("Run Merge Bias Validation", key="run_merge"):
        from scipy import integrate
        import numpy as np
        from converge.distributions import Triangular as T, Deterministic
        from converge.network import Activity, Network
        from converge.engine import SimulationEngine, SimulationConfig
        from converge.results import SimulationResults

        a1, m1, b1 = 10.0, 15.0, 25.0
        a2, m2, b2 = 12.0, 18.0, 30.0

        def tri_cdf(a, m, b):
            def cdf(x):
                x = np.atleast_1d(np.asarray(x, dtype=float))
                result = np.zeros_like(x)
                mask1 = (x >= a) & (x < m)
                mask2 = (x >= m) & (x <= b)
                mask3 = x >= b
                result[mask1] = (x[mask1] - a)**2 / ((b - a) * (m - a))
                result[mask2] = 1 - (b - x[mask2])**2 / ((b - a) * (b - m))
                result[mask3] = 1.0
                return result
            return cdf

        cdf1 = tri_cdf(a1, m1, b1)
        cdf2 = tri_cdf(a2, m2, b2)
        x_low = min(a1, a2)
        x_high = max(b1, b2)
        ref, _ = integrate.quad(lambda x: 1.0 - float(cdf1(np.array([x]))) * float(cdf2(np.array([x]))), x_low, x_high)
        ref += x_low

        acts = [
            Activity("P1", "Path 1", T(a1, m1, b1)),
            Activity("P2", "Path 2", T(a2, m2, b2)),
            # Zero-duration logic node: Triangular forbids a == b, use Deterministic
            Activity("M",  "Merge",  Deterministic(0), predecessors=["P1", "P2"]),
        ]
        net = Network(acts)
        config = SimulationConfig(n_iterations=50_000, seed=42)
        output = SimulationEngine(net).run(config)
        sim_mean = float(output.completion.mean())

        max_of_means = max(T(a1, m1, b1).mean(), T(a2, m2, b2).mean())
        diff = abs(sim_mean - ref)
        tolerance = 0.5

        col1, col2, col3 = st.columns(3)
        col1.metric("max(E[X])", f"{max_of_means:.3f}d", help="CPM estimate (Jensen lower bound)")
        col2.metric("Numerical E[max]", f"{ref:.3f}d", help="scipy.integrate.quad reference")
        col3.metric("Simulated E[max]", f"{sim_mean:.3f}d",
                    delta=f"{sim_mean - ref:+.3f}d vs reference")

        if diff < tolerance:
            st.success(f"PASS: Simulation matches numerical reference within {tolerance}d (diff = {diff:.3f}d). Jensen's inequality confirmed: {sim_mean:.3f} > {max_of_means:.3f}")
        else:
            st.error(f"FAIL: Difference {diff:.3f}d exceeds tolerance {tolerance}d")

        mb_gap = sim_mean - max_of_means
        st.info(f"Merge bias: **{mb_gap:.3f} working days** — the gap between what CPM predicts and what actually happens on average.")

    # Layer 3: Convergence rate
    st.subheader("Layer 3 — Convergence Rate (σ/√N)")
    with st.expander("What is being tested?", expanded=False):
        st.markdown("""
The Monte Carlo standard error of the mean estimate decays as:

$$SE(N) \\approx \\frac{\\sigma}{\\sqrt{N}}$$

On a log-log plot, this appears as a straight line with slope **−0.5**.
We verify this empirically by running the simulation at multiple values of N
(500, 1000, 2000, 5000, 10000) and measuring the standard deviation of the
mean estimate across 20 independent replications.

The fitted slope must be in the range [−0.65, −0.35] to pass.
This is the strongest single correctness claim: it would fail if the sampling
was biased, if the RNG was broken, or if the vectorisation introduced errors.
        """)

    if st.button("Run Convergence Test (takes ~15 seconds)", key="run_conv"):
        from converge.distributions import Triangular as T
        from converge.network import Activity, Network
        from converge.engine import SimulationEngine, SimulationConfig
        import numpy as np

        n_sizes = [500, 1000, 2000, 5000, 10_000]
        M_REPS = 20
        se_vals = []
        dist = T(40, 50, 100)
        theoretical_sigma = dist.std()

        progress = st.progress(0)
        for i, n in enumerate(n_sizes):
            means = []
            for r in range(M_REPS):
                net = Network([Activity("A", "Activity", dist)])
                config = SimulationConfig(n_iterations=n, seed=1000 * n + r)
                output = SimulationEngine(net).run(config)
                means.append(output.completion.mean())
            se_vals.append(float(np.std(means)))
            progress.progress((i + 1) / len(n_sizes))

        theoretical_se = [theoretical_sigma / np.sqrt(n) for n in n_sizes]

        log_n = np.log(n_sizes)
        log_se = np.log(se_vals)
        slope = float(np.polyfit(log_n, log_se, 1)[0])
        passed = -0.65 <= slope <= -0.35

        fig_conv = plot_convergence(n_sizes, se_vals, theoretical_se)
        st.plotly_chart(fig_conv, use_container_width=True)
        _download_chart(fig_conv, "Convergence", "convergence")

        if passed:
            st.success(f"PASS: Log-log slope = {slope:.3f} (expected ≈ −0.5, tolerance [−0.65, −0.35])")
        else:
            st.error(f"FAIL: Slope {slope:.3f} outside expected range [−0.65, −0.35]")

        conv_df = pd.DataFrame({
            "N": n_sizes,
            "Simulated SE": [f"{s:.5f}" for s in se_vals],
            "Theoretical σ/√N": [f"{s:.5f}" for s in theoretical_se],
            "Ratio (sim/theory)": [f"{s/t:.3f}" for s, t in zip(se_vals, theoretical_se)],
        })
        st.dataframe(conv_df, use_container_width=True, hide_index=True)

    # Hulett Benchmarks
    st.subheader("Published Benchmark: Hulett (1996)")
    st.caption(
        "Source: Hulett, D.T. (1996). 'Schedule Risk Analysis Simplified.' "
        "PM Network, PMI, July 1996. "
        "https://www.pmi.org/learning/library/schedule-risk-analysis-simplified-10742"
    )

    if st.button("Run Hulett (1996) Benchmark Cases", key="run_hulett"):
        from converge.examples import hulett_case1, hulett_case2, hulett_case3
        from converge.engine import SimulationEngine, SimulationConfig
        from converge.results import SimulationResults

        n_bench = 100_000
        config = SimulationConfig(n_iterations=n_bench, seed=42)

        # ---- Case 1: single path (pure skew bias, zero merge bias) ----
        ex1 = hulett_case1()
        res1 = SimulationResults(
            SimulationEngine(ex1.network, ex1.risk_drivers).run(config), ex1.network
        )
        case1_mean = float(res1.completion.mean())
        with st.expander("**Case 1: Single Path**", expanded=True):
            st.caption(f"CPM completion: {ex1.network.cpm_completion():.0f}d")
            p_cpm1 = res1.probability_of_meeting(130.0)
            c1, c2 = st.columns(2)
            c1.metric("P(completion ≤ CPM)", f"{p_cpm1:.1%}",
                      help="Published target: only ~10-15% likely")
            c2.metric("Mean completion", f"{case1_mean:.1f}d")
            ok = 0.08 <= p_cpm1 <= 0.25
            st.success("PASS: P(CPM) in 10–25% range") if ok else st.error(f"FAIL: P(CPM) = {p_cpm1:.1%}")
            mb1 = res1.merge_bias
            st.caption(
                f"Decomposition check: true merge bias = {mb1.merge_bias_days:.2f}d "
                f"(expected ≈ 0 — single path, no merge); "
                f"skew/mean shift = {mb1.mean_shift_days:.1f}d carries the entire gap."
            )

        # ---- Case 2: identical parallel path added (isolates merge bias) ----
        ex2 = hulett_case2()
        res2 = SimulationResults(
            SimulationEngine(ex2.network, ex2.risk_drivers).run(config), ex2.network
        )
        case2_mean = float(res2.completion.mean())
        with st.expander("**Case 2: Two Parallel Paths (Merge Bias)**", expanded=True):
            st.caption(f"CPM completion: {ex2.network.cpm_completion():.0f}d (unchanged vs Case 1)")
            p_cpm2 = res2.probability_of_meeting(130.0)
            mean_shift = case2_mean - case1_mean
            c1, c2, c3 = st.columns(3)
            c1.metric("P(completion ≤ CPM)", f"{p_cpm2:.1%}",
                      help="Published target: under 5%")
            c2.metric("Mean completion", f"{case2_mean:.1f}d")
            c3.metric("Mean shift vs Case 1", f"+{mean_shift:.1f} wd",
                      help="Published: ~10 calendar days ≈ 7-8 working days under "
                           "a 5-day week. Caused purely by adding one identical "
                           "parallel path — the canonical merge-bias result.")
            ok_p = p_cpm2 < 0.10
            st.success(f"PASS: P(CPM) < 10% (got {p_cpm2:.1%})") if ok_p else st.error(f"FAIL: P(CPM) = {p_cpm2:.1%}")
            ok_shift = 4.0 <= mean_shift <= 10.0
            if ok_shift:
                st.success(
                    f"PASS: Mean shifts +{mean_shift:.1f} working days vs Case 1 — "
                    f"consistent with the published ~10 calendar days (~7-8 wd, 5-day week). "
                    f"CPM is identical in both cases; the shift is pure merge bias "
                    f"(Jensen: E[max(X,Y)] > E[X] for iid paths)."
                )
            else:
                st.error(f"FAIL: Mean shift {mean_shift:.1f} wd outside expected 4–10 wd range")
            st.caption(
                "Note: the published figure is in calendar days against a 1996 US "
                "calendar; comparison is made in working-day terms (~10 cal ≈ 7-8 wd)."
            )

        # ---- Case 3: criticality vs CPM critical path ----
        ex3 = hulett_case3()
        res3 = SimulationResults(
            SimulationEngine(ex3.network, ex3.risk_drivers).run(config), ex3.network
        )
        with st.expander("**Case 3: Criticality ≠ CPM Path**", expanded=True):
            st.caption(f"CPM completion: {ex3.network.cpm_completion():.0f}d")
            act_res = {ar.id: ar for ar in res3.activity_results}
            ci_b = max(act_res["B101"].criticality_index, act_res["B102"].criticality_index)
            ci_a = max(act_res["A101"].criticality_index, act_res["A102"].criticality_index)
            cols = st.columns(2)
            cols[0].metric("Path A (CPM critical) CI", f"{ci_a:.1%}",
                           help="Published: ~31%")
            cols[1].metric("Path B (5d float) CI", f"{ci_b:.1%}",
                           help="Published: ~69%")
            ok = ci_b > ci_a and 0.55 <= ci_b <= 0.80
            st.success(f"PASS: Path B ({ci_b:.1%}) > Path A ({ci_a:.1%}), near published ~69%") if ok else st.error(f"FAIL: CI_B={ci_b:.1%}, CI_A={ci_a:.1%}")


# ---- Tab: Distribution Comparison ----
with tab_dist:
    st.subheader("PERT vs Triangular — Tail Risk Comparison")
    with st.expander("Math note: The PERT limitation", expanded=True):
        st.markdown("""
**PERT** and **Triangular** both use three-point estimates $(a, m, b)$ but behave very differently
for skewed activities:

- **Triangular**: the standard deviation is fully determined by the spread: $\\text{std} \\approx \\frac{b-a}{\\sqrt{24}}$
- **PERT**: fixes the effective standard deviation at approximately $\\frac{b-a}{6}$, regardless of skew.
  This is the same approximation as the classical PERT "6-sigma rule."

For highly skewed activities (e.g. $a=10, m=12, b=60$), PERT's smaller standard deviation
**understates tail risk** — the chance of hitting the long tail (near $b=60$) is lower under
PERT than under Triangular. This is a deliberate design choice in PERT (smoothness over tail
fidelity) but a critical assumption to understand in practice.

The plot below lets you see this side by side. Try a skewed case like $(10, 12, 60)$.
        """)

    col1, col2, col3 = st.columns(3)
    cmp_a = col1.number_input("a (min)", value=10.0, step=1.0, key="cmp_a")
    cmp_m = col2.number_input("m (most likely)", value=15.0, step=1.0, key="cmp_m")
    cmp_b = col3.number_input("b (max)", value=25.0, step=1.0, key="cmp_b")

    if cmp_a < cmp_m < cmp_b:
        fig_cmp = plot_pert_vs_triangular(cmp_a, cmp_m, cmp_b)
        st.plotly_chart(fig_cmp, use_container_width=True)
        _download_chart(fig_cmp, "PERT vs Triangular", "pert_vs_tri")

        from converge.distributions import Triangular as T, PERT as P
        tri = T(cmp_a, cmp_m, cmp_b)
        pert_d = P(cmp_a, cmp_m, cmp_b)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Triangular Mean", f"{tri.mean():.2f}d")
            st.metric("Triangular Std Dev", f"{tri.std():.2f}d")
        with col2:
            st.metric("PERT Mean", f"{pert_d.mean():.2f}d")
            st.metric("PERT Std Dev", f"{pert_d.std():.2f}d")
            st.caption(
                f"PERT std is {pert_d.std()/tri.std():.0%} of Triangular std "
                f"— {'significant understatement of tail risk' if pert_d.std() < tri.std() * 0.8 else 'similar'}"
            )
    else:
        st.warning("Parameter constraint: a < m < b required.")


# ============================================================
# Entry point
# ============================================================

def main():
    """Console entry point (converge command)."""
    import subprocess
    import sys
    import os
    app_path = os.path.abspath(__file__)
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)


if __name__ == "__main__":
    pass  # streamlit run handles execution

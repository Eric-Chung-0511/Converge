"""
Plotly figure builders for Converge results.

All figures are fully interactive (zoom, pan, hover tooltips).
Each figure builder accepts a SimulationResults object and returns a
plotly.graph_objects.Figure ready for st.plotly_chart().

Export helpers
--------------
export_figure_png(fig, path)  — static PNG via kaleido
export_figure_svg(fig, path)  — static SVG via kaleido
export_figure_html(fig, path) — interactive standalone HTML
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from converge.results import SimulationResults, ActivityResult


# Colour palette
C_PRIMARY = "#2563EB"   # blue
C_SECONDARY = "#DC2626" # red
C_SUCCESS = "#16A34A"   # green
C_WARNING = "#D97706"   # amber
C_NEUTRAL = "#6B7280"   # grey
C_LIGHT = "#EFF6FF"     # light blue bg


def plot_s_curve(
    results: SimulationResults,
    percentiles: Optional[list[float]] = None,
    target_date: Optional[float] = None,
    title: str = "S-Curve: Probability of Completion by Date",
) -> go.Figure:
    """
    CDF (S-curve) of project completion.

    Marks P50/P80/P90 with vertical reference lines. Optionally marks
    a contractual target date to visualise the probability of meeting it.
    """
    if percentiles is None:
        percentiles = [50, 80, 90]

    completion = np.sort(results.completion)
    n = len(completion)
    cdf_y = np.arange(1, n + 1) / n * 100  # probability in percent

    pct_values = {p: results.percentile(p) for p in percentiles}
    colours = {50: C_PRIMARY, 80: C_WARNING, 90: C_SECONDARY}

    fig = go.Figure()

    # Main S-curve
    fig.add_trace(go.Scatter(
        x=completion,
        y=cdf_y,
        mode="lines",
        name="Completion CDF",
        line=dict(color=C_PRIMARY, width=2.5),
        hovertemplate="Day %{x:.0f}: %{y:.1f}%<extra></extra>",
    ))

    # Percentile markers
    for p, val in pct_values.items():
        color = colours.get(p, C_NEUTRAL)
        fig.add_vline(
            x=val,
            line=dict(color=color, width=1.5, dash="dash"),
            annotation_text=f"P{int(p)}: {val:.0f}d",
            annotation_position="top",
            annotation=dict(font=dict(color=color, size=11)),
        )

    # Target date marker
    if target_date is not None:
        p_target = results.probability_of_meeting(target_date) * 100
        fig.add_vline(
            x=target_date,
            line=dict(color=C_SUCCESS, width=2, dash="dot"),
            annotation_text=f"Target ({target_date:.0f}d): {p_target:.1f}% likely",
            annotation_position="bottom right",
            annotation=dict(font=dict(color=C_SUCCESS, size=11)),
        )

    # CPM date
    cpm = results.network.cpm_completion()
    p_cpm = results.probability_of_meeting(cpm) * 100
    fig.add_vline(
        x=cpm,
        line=dict(color=C_NEUTRAL, width=1.5, dash="longdash"),
        annotation_text=f"CPM: {cpm:.0f}d ({p_cpm:.1f}% likely)",
        annotation_position="bottom left",
        annotation=dict(font=dict(color=C_NEUTRAL, size=11)),
    )

    fig.update_layout(
        title=title,
        xaxis_title="Working Days from Project Start",
        yaxis_title="Probability of Completion (%)",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        legend=dict(x=0.02, y=0.95),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    _apply_grid_style(fig)
    return fig


def plot_histogram(
    results: SimulationResults,
    n_bins: int = 60,
    title: str = "Completion Distribution (PDF)",
) -> go.Figure:
    """Histogram of completion outcomes with percentile overlay."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=results.completion,
        nbinsx=n_bins,
        name="Completion frequency",
        marker_color=C_PRIMARY,
        opacity=0.75,
        hovertemplate="Day %{x:.0f}: %{y} iterations<extra></extra>",
    ))

    # Percentile markers
    for p, color in [(50, C_PRIMARY), (80, C_WARNING), (90, C_SECONDARY)]:
        val = results.percentile(p)
        fig.add_vline(
            x=val,
            line=dict(color=color, width=2, dash="dash"),
            annotation_text=f"P{p}: {val:.0f}d",
            annotation_position="top",
            annotation=dict(font=dict(color=color, size=11)),
        )

    # CPM line
    cpm = results.network.cpm_completion()
    fig.add_vline(
        x=cpm,
        line=dict(color=C_NEUTRAL, width=2, dash="longdash"),
        annotation_text=f"CPM: {cpm:.0f}d",
        annotation_position="bottom right",
        annotation=dict(font=dict(color=C_NEUTRAL, size=11)),
    )

    fig.update_layout(
        title=title,
        xaxis_title="Working Days from Project Start",
        yaxis_title="Number of Iterations",
        bargap=0.05,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    _apply_grid_style(fig)
    return fig


def plot_tornado(
    results: SimulationResults,
    top_n: int = 15,
    title: str = "Tornado: Activity Sensitivity to Completion",
) -> go.Figure:
    """
    Horizontal bar chart ranking activities by Spearman rank correlation
    with project completion (sensitivity / cruciality measure).
    """
    act_results = results.activity_results
    top = sorted(act_results, key=lambda r: abs(r.sensitivity), reverse=True)[:top_n]
    top = list(reversed(top))  # bottom-to-top for Plotly horizontal bars

    labels = [f"{r.id}: {r.name}" for r in top]
    values = [r.sensitivity for r in top]
    colors = [C_SECONDARY if v >= 0 else C_PRIMARY for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>Sensitivity: %{x:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Spearman Rank Correlation with Completion",
        yaxis_title="",
        xaxis=dict(range=[-1, 1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        margin=dict(l=220, r=30, t=60, b=50),
        height=max(350, 30 * len(top) + 100),
    )
    fig.add_vline(x=0, line=dict(color=C_NEUTRAL, width=1))
    _apply_grid_style(fig)
    return fig


def plot_criticality(
    results: SimulationResults,
    title: str = "Criticality Index vs CPM Critical Path",
) -> go.Figure:
    """
    Bar chart of criticality index per activity, coloured by whether the
    activity is on the deterministic CPM critical path.
    """
    act_results = sorted(results.activity_results,
                         key=lambda r: r.criticality_index, reverse=True)

    labels = [f"{r.id}: {r.name}" for r in act_results]
    ci_values = [r.criticality_index * 100 for r in act_results]
    colors = [C_SECONDARY if r.is_cpm_critical else C_PRIMARY for r in act_results]
    hover = [
        f"{'★ CPM Critical' if r.is_cpm_critical else '  Not CPM Critical'}<br>"
        f"Criticality Index: {r.criticality_index:.1%}<br>"
        f"CPM Duration: {r.cpm_duration:.1f}d<br>"
        f"Simulated Mean: {r.mean_duration:.1f}d"
        for r in act_results
    ]

    fig = go.Figure(go.Bar(
        x=labels,
        y=ci_values,
        marker_color=colors,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
    ))

    # Legend indicators
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color=C_SECONDARY,
        name="CPM Critical Path",
        showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color=C_PRIMARY,
        name="Not on CPM Critical Path",
        showlegend=True,
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Activity",
        yaxis_title="Criticality Index (%)",
        yaxis=dict(range=[0, 105], ticksuffix="%"),
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        xaxis_tickangle=-30,
        margin=dict(l=60, r=30, t=60, b=120),
        legend=dict(x=0.75, y=0.98),
    )
    _apply_grid_style(fig)
    return fig


def plot_correlation_matrix(
    activity_ids: list[str],
    corr_matrix: np.ndarray,
    title: str = "Implied Correlation Matrix (Risk-Driver Model)",
) -> go.Figure:
    """Heatmap of the implied Pearson correlation matrix."""
    if corr_matrix.size == 0 or len(activity_ids) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="No risk drivers assigned — correlation matrix not available.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    fig = go.Figure(go.Heatmap(
        z=corr_matrix,
        x=activity_ids,
        y=activity_ids,
        zmin=-1, zmax=1,
        colorscale="RdBu",
        reversescale=True,
        hovertemplate="%{y} ↔ %{x}: %{z:.3f}<extra></extra>",
        text=np.round(corr_matrix, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))

    fig.update_layout(
        title=title,
        xaxis_tickangle=-35,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        margin=dict(l=100, r=30, t=60, b=100),
        height=max(350, 40 * len(activity_ids) + 150),
    )
    return fig


def plot_convergence(
    n_sizes: list[int],
    se_values: list[float],
    theoretical_se: Optional[list[float]] = None,
    title: str = "Convergence: Monte Carlo Standard Error vs Iterations",
) -> go.Figure:
    """
    Log-log plot of SE vs N demonstrating sigma/sqrt(N) convergence.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=n_sizes, y=se_values,
        mode="lines+markers",
        name="Simulated SE",
        line=dict(color=C_PRIMARY, width=2.5),
        marker=dict(size=8),
        hovertemplate="N=%{x}: SE=%{y:.4f}<extra></extra>",
    ))

    if theoretical_se is not None:
        fig.add_trace(go.Scatter(
            x=n_sizes, y=theoretical_se,
            mode="lines",
            name="Theoretical σ/√N",
            line=dict(color=C_WARNING, width=2, dash="dash"),
            hovertemplate="N=%{x}: σ/√N=%{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Number of Iterations (N)",
        yaxis_title="Standard Error of Mean Estimate",
        xaxis_type="log",
        yaxis_type="log",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        margin=dict(l=70, r=30, t=60, b=50),
        legend=dict(x=0.98, y=0.98, xanchor="right"),
    )

    # Slope annotation
    if len(n_sizes) >= 2 and len(se_values) >= 2:
        log_n = np.log(n_sizes)
        log_se = np.log(se_values)
        slope = float(np.polyfit(log_n, log_se, 1)[0])
        fig.add_annotation(
            text=f"Fitted slope: {slope:.3f} (expected ≈ −0.5)",
            xref="paper", yref="paper",
            x=0.02, y=0.08,
            showarrow=False,
            font=dict(size=12, color=C_SECONDARY),
            bgcolor="rgba(255,255,255,0.8)",
        )
    _apply_grid_style(fig)
    return fig


def plot_pert_vs_triangular(
    a: float, m: float, b: float,
    n_samples: int = 10_000,
    title: str = "PERT vs Triangular — Tail Risk Comparison",
) -> go.Figure:
    """
    Overlay the PDF of Triangular(a, m, b) vs PERT(a, m, b) to show how
    PERT's reduced variance understates tail risk for skewed activities.
    """
    from converge.distributions import Triangular, PERT
    rng = np.random.default_rng(42)

    tri = Triangular(a, m, b)
    pert = PERT(a, m, b)

    tri_samples = tri.sample(n_samples, rng)
    pert_samples = pert.sample(n_samples, rng)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=tri_samples, nbinsx=60,
        name=f"Triangular (std={tri.std():.1f}d)",
        marker_color=C_PRIMARY,
        opacity=0.55,
        histnorm="probability density",
        hovertemplate="Day %{x:.0f}: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=pert_samples, nbinsx=60,
        name=f"PERT (std={pert.std():.1f}d)",
        marker_color=C_SECONDARY,
        opacity=0.55,
        histnorm="probability density",
        hovertemplate="Day %{x:.0f}: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Duration (working days)",
        yaxis_title="Probability Density",
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        legend=dict(x=0.75, y=0.95),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    _apply_grid_style(fig)
    return fig


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_figure_png(fig: go.Figure, path: str | Path) -> None:
    """Export figure to PNG (requires kaleido)."""
    fig.write_image(str(path), format="png", width=1200, height=700, scale=2)


def export_figure_svg(fig: go.Figure, path: str | Path) -> None:
    """Export figure to SVG."""
    fig.write_image(str(path), format="svg", width=1200, height=700)


def export_figure_pdf(fig: go.Figure, path: str | Path) -> None:
    """Export figure to PDF (vector, requires kaleido)."""
    fig.write_image(str(path), format="pdf", width=1200, height=700)


def export_figure_html(fig: go.Figure, path: str | Path) -> None:
    """Export figure to a standalone interactive HTML file."""
    fig.write_html(str(path), include_plotlyjs="cdn")


def fig_to_bytes(fig: go.Figure, fmt: str = "png") -> bytes:
    """Return figure as bytes (for st.download_button)."""
    if fmt == "html":
        return fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    return fig.to_image(format=fmt, width=1200, height=700, scale=2)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _apply_grid_style(fig: go.Figure) -> None:
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor="#E5E7EB",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor="#E5E7EB",
        zeroline=False,
    )

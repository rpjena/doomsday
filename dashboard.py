"""
EMN Doomsday Clock — Live Dashboard
=====================================
Dash application. Run with: python dashboard.py
Opens at http://127.0.0.1:8050
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from data.synthetic import generate_market_data
from components.component_a import normalize_component_a
from components.component_b import normalize_component_b
from components.component_c import normalize_component_c
from components.component_d import normalize_component_d
from aggregator import compute_clock, ClockReading


# ─────────────────────────────────────────────
# INSTALL BOOTSTRAP IF NEEDED
# ─────────────────────────────────────────────
try:
    import dash_bootstrap_components as dbc
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "dash-bootstrap-components", "--break-system-packages", "-q"])
    import dash_bootstrap_components as dbc


# ─────────────────────────────────────────────
# DATA & COMPUTATION
# ─────────────────────────────────────────────
DATA = generate_market_data(n_days=504)

def run_full_clock(data=DATA) -> ClockReading:
    res_a = normalize_component_a(data["factor_exposures"])
    res_b = normalize_component_b(
        data["long_returns"], data["short_returns"],
        data["portfolio_returns"], data["market_returns"]
    )
    res_c = normalize_component_c(
        crowding_score=float(data["crowding_scores"].iloc[-1]),
        liquidity_score=float(data["liquidity_scores"].iloc[-1]),
        vix_current=float(data["vix"].iloc[-1]),
        ig_spread_current=float(data["ig_spread"].iloc[-1]),
    )
    res_d = normalize_component_d(
        data["portfolio_returns"], data["market_returns"], data["vix"]
    )
    return compute_clock(
        res_a["normalized_score"], res_b["normalized_score"],
        res_c["normalized_score"], res_d["normalized_score"],
        res_a, res_b, res_c, res_d,
    )

READING = run_full_clock()


# ─────────────────────────────────────────────
# CLOCK FACE FIGURE
# ─────────────────────────────────────────────
ZONE_COLORS = {
    "SAFE":     "#00C853",
    "NORMAL":   "#FFD600",
    "ELEVATED": "#FF6D00",
    "CRITICAL": "#D50000",
}

def make_clock_figure(reading: ClockReading) -> go.Figure:
    mins = reading.minutes_to_midnight
    angle_deg = 90 - (mins / 23) * (360 * 23 / 60)   # fraction of clock face

    zone_color = ZONE_COLORS[reading.zone]

    fig = go.Figure()

    # Danger arc background sectors
    sector_config = [
        (0,  5,  "#D50000", 0.18),   # Critical  0–5 min
        (5,  10, "#FF6D00", 0.13),   # Elevated  5–10 min
        (10, 20, "#FFD600", 0.08),   # Normal    10–20 min
        (20, 23, "#00C853", 0.05),   # Safe      20–23 min
    ]
    theta_vals = np.linspace(0, 2 * np.pi, 500)
    for lo, hi, color, alpha in sector_config:
        t0 = np.pi / 2 - (hi / 23) * 2 * np.pi
        t1 = np.pi / 2 - (lo / 23) * 2 * np.pi
        ts = np.linspace(t0, t1, 80)
        xs = [0] + list(0.85 * np.cos(ts)) + [0]
        ys = [0] + list(0.85 * np.sin(ts)) + [0]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself",
            fillcolor=color, opacity=alpha,
            line=dict(width=0), hoverinfo="skip", showlegend=False
        ))

    # Clock ring
    t = np.linspace(0, 2 * np.pi, 300)
    fig.add_trace(go.Scatter(
        x=np.cos(t), y=np.sin(t),
        mode="lines", line=dict(color="#cccccc", width=2),
        hoverinfo="skip", showlegend=False
    ))

    # Hour tick marks
    for i in range(12):
        angle = np.pi / 2 - i * np.pi / 6
        x0, y0 = 0.88 * np.cos(angle), 0.88 * np.sin(angle)
        x1, y1 = 1.00 * np.cos(angle), 1.00 * np.sin(angle)
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color="#999999", width=1.5),
            hoverinfo="skip", showlegend=False
        ))

    # Minute hand (pointing to current risk time)
    # Minutes mapped: 23 min = 12 o'clock, 0 min = 12 o'clock + full revolution
    # We only show the last 23 minutes before midnight (11:37 PM → midnight)
    # Angle: 12 o'clock = 90°, clockwise. mins=23 → exactly 12 o'clock
    hand_angle = np.pi / 2 - (1 - mins / 23) * 2 * np.pi  # 0 min = full circle from 12
    hx = 0.72 * np.cos(hand_angle)
    hy = 0.72 * np.sin(hand_angle)
    fig.add_trace(go.Scatter(
        x=[0, hx], y=[0, hy], mode="lines",
        line=dict(color=zone_color, width=6),
        hoverinfo="skip", showlegend=False
    ))
    # Hand tip dot
    fig.add_trace(go.Scatter(
        x=[hx], y=[hy], mode="markers",
        marker=dict(color=zone_color, size=14, symbol="circle"),
        hoverinfo="skip", showlegend=False
    ))
    # Center pin
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(color="#ffffff", size=10, symbol="circle",
                    line=dict(color="#444444", width=2)),
        hoverinfo="skip", showlegend=False
    ))

    # "MIDNIGHT" label at top
    fig.add_annotation(x=0, y=1.15, text="MIDNIGHT", showarrow=False,
                       font=dict(size=11, color="#888888", family="monospace"))

    # Central readout
    fig.add_annotation(
        x=0, y=-0.30,
        text=f"<b>{mins:.1f}</b>",
        showarrow=False,
        font=dict(size=40, color=zone_color, family="monospace"),
    )
    fig.add_annotation(
        x=0, y=-0.50,
        text="minutes to midnight",
        showarrow=False,
        font=dict(size=12, color="#aaaaaa", family="monospace"),
    )
    fig.add_annotation(
        x=0, y=-0.65,
        text=f"{reading.zone_emoji}  {reading.zone}  |  score {reading.raw_score:.2f}/10",
        showarrow=False,
        font=dict(size=13, color=zone_color, family="monospace"),
    )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(range=[-1.3, 1.3], visible=False, scaleanchor="y"),
        yaxis=dict(range=[-1.3, 1.3], visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        height=400,
    )
    return fig


def make_gauge_figure(score: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": label, "font": {"size": 11, "color": "#cccccc", "family": "monospace"}},
        number={"font": {"size": 22, "color": color, "family": "monospace"}},
        gauge={
            "axis": {"range": [0, 10], "tickfont": {"size": 8, "color": "#888888"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1a1a1a",
            "bordercolor": "#333333",
            "steps": [
                {"range": [0, 4],  "color": "#1a2a1a"},
                {"range": [4, 6],  "color": "#2a2a10"},
                {"range": [6, 8],  "color": "#2a1a10"},
                {"range": [8, 10], "color": "#2a1010"},
            ],
            "threshold": {
                "line": {"color": "#ff4444", "width": 2},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=15, r=15, t=30, b=5),
        paper_bgcolor="#111111",
        font=dict(color="#cccccc"),
    )
    return fig


def make_timeseries_figure(data=DATA) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=["Portfolio Cumulative Returns", "VIX Level", "L/S Rolling Correlation (60d)"],
        vertical_spacing=0.10,
    )

    cum_ret = (1 + data["portfolio_returns"]).cumprod()
    fig.add_trace(go.Scatter(
        x=data["dates"], y=cum_ret, name="Portfolio",
        line=dict(color="#00bcd4", width=1.5)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data["dates"], y=data["vix"], name="VIX",
        line=dict(color="#ff9800", width=1.5)), row=2, col=1)

    corr = data["long_returns"].rolling(60).corr(data["short_returns"])
    fig.add_trace(go.Scatter(
        x=data["dates"], y=corr, name="L/S Corr",
        line=dict(color="#e91e63", width=1.5)), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#555555", row=3, col=1)

    fig.update_layout(
        showlegend=True,
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font=dict(color="#cccccc", family="monospace", size=10),
        height=350,
        margin=dict(l=40, r=10, t=40, b=10),
        legend=dict(bgcolor="#1a1a1a", font=dict(size=9)),
    )
    fig.update_xaxes(gridcolor="#222222", linecolor="#333333")
    fig.update_yaxes(gridcolor="#222222", linecolor="#333333")
    return fig


# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="EMN Doomsday Clock",
)

def score_color(s):
    if s >= 8: return "#D50000"
    if s >= 6: return "#FF6D00"
    if s >= 4: return "#FFD600"
    return "#00C853"

r = READING

app.layout = html.Div(
    style={"backgroundColor": "#0d0d0d", "minHeight": "100vh", "fontFamily": "monospace", "color": "#cccccc"},
    children=[
        # Header
        html.Div([
            html.H2("⚛ EMN DOOMSDAY CLOCK", style={"color": "#ff4444", "letterSpacing": "4px", "margin": 0}),
            html.P("Equity Market Neutral — Real-Time Risk Monitor",
                   style={"color": "#666666", "fontSize": "12px", "margin": "4px 0 0 0", "letterSpacing": "2px"}),
        ], style={"padding": "20px 30px 10px", "borderBottom": "1px solid #222"}),

        # Main body
        html.Div([
            # Left: Clock
            html.Div([
                dcc.Graph(
                    id="clock-fig",
                    figure=make_clock_figure(r),
                    config={"displayModeBar": False},
                )
            ], style={"width": "38%", "display": "inline-block", "verticalAlign": "top"}),

            # Right: 4 component gauges
            html.Div([
                html.Div("COMPONENT SCORES", style={"color": "#555", "fontSize": "10px",
                          "letterSpacing": "3px", "padding": "10px 0 5px 0"}),
                html.Div([
                    html.Div([
                        dcc.Graph(figure=make_gauge_figure(
                            r.component_scores["A"], "A · FACTOR EXPOSURE",
                            score_color(r.component_scores["A"])),
                            config={"displayModeBar": False})
                    ], style={"width": "50%", "display": "inline-block"}),
                    html.Div([
                        dcc.Graph(figure=make_gauge_figure(
                            r.component_scores["B"], "B · CORRELATION BREAKDOWN",
                            score_color(r.component_scores["B"])),
                            config={"displayModeBar": False})
                    ], style={"width": "50%", "display": "inline-block"}),
                    html.Div([
                        dcc.Graph(figure=make_gauge_figure(
                            r.component_scores["C"], "C · CROWDING & LIQUIDITY",
                            score_color(r.component_scores["C"])),
                            config={"displayModeBar": False})
                    ], style={"width": "50%", "display": "inline-block"}),
                    html.Div([
                        dcc.Graph(figure=make_gauge_figure(
                            r.component_scores["D"], "D · MACRO SHOCK",
                            score_color(r.component_scores["D"])),
                            config={"displayModeBar": False})
                    ], style={"width": "50%", "display": "inline-block"}),
                ]),
            ], style={"width": "58%", "display": "inline-block", "verticalAlign": "top", "padding": "0 0 0 15px"}),
        ], style={"padding": "10px 20px"}),

        # Risk breakdown table
        html.Div([
            html.Div("WEIGHTED CONTRIBUTIONS", style={"color": "#555", "fontSize": "10px",
                      "letterSpacing": "3px", "padding": "15px 0 8px 0"}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Component"), html.Th("Raw Score"), html.Th("Weight"),
                    html.Th("Contribution"), html.Th("Key Signal")
                ], style={"color": "#777", "fontSize": "11px"})),
                html.Tbody([
                    html.Tr([
                        html.Td("A · Factor Exposure"),
                        html.Td(f"{r.component_scores['A']:.2f}",
                                style={"color": score_color(r.component_scores['A'])}),
                        html.Td("35%"),
                        html.Td(f"{r.weighted_contributions['A']:.3f}",
                                style={"color": score_color(r.weighted_contributions['A']*10/3.5)}),
                        html.Td(f"Max factor: {r.sub_details['A']['max_exposure_factor']} "
                                f"({r.sub_details['A']['max_exposure_value']:.3f})"),
                    ]),
                    html.Tr([
                        html.Td("B · Corr Breakdown"),
                        html.Td(f"{r.component_scores['B']:.2f}",
                                style={"color": score_color(r.component_scores['B'])}),
                        html.Td("30%"),
                        html.Td(f"{r.weighted_contributions['B']:.3f}"),
                        html.Td(f"L/S corr: {r.sub_details['B']['current_long_short_corr']:.3f}  "
                                f"β: {r.sub_details['B']['current_beta']:.3f}"),
                    ]),
                    html.Tr([
                        html.Td("C · Crowding & Liq"),
                        html.Td(f"{r.component_scores['C']:.2f}",
                                style={"color": score_color(r.component_scores['C'])}),
                        html.Td("25%"),
                        html.Td(f"{r.weighted_contributions['C']:.3f}"),
                        html.Td(f"VIX: {r.sub_details['C']['vix_current']:.1f}  "
                                f"IG: {r.sub_details['C']['ig_spread_current']:.0f}bps"),
                    ]),
                    html.Tr([
                        html.Td("D · Macro Shock"),
                        html.Td(f"{r.component_scores['D']:.2f}",
                                style={"color": score_color(r.component_scores['D'])}),
                        html.Td("10%"),
                        html.Td(f"{r.weighted_contributions['D']:.3f}"),
                        html.Td(f"Regime: {r.sub_details['D']['regime']}  "
                                f"VIX pct: {r.sub_details['D']['vix_percentile']:.0%}"),
                    ]),
                    html.Tr([
                        html.Td(html.B("TOTAL")),
                        html.Td(""), html.Td(""),
                        html.Td(html.B(f"{r.raw_score:.3f}"),
                                style={"color": score_color(r.raw_score)}),
                        html.Td(html.B(f"→ {r.minutes_to_midnight:.1f} min  {r.zone_emoji} {r.zone}"),
                                style={"color": ZONE_COLORS[r.zone]}),
                    ]),
                ], style={"fontSize": "12px", "lineHeight": "2.0"}),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
        ], style={"padding": "0 30px 20px"}),

        # Time series
        html.Div([
            html.Div("MARKET DIAGNOSTICS", style={"color": "#555", "fontSize": "10px",
                      "letterSpacing": "3px", "padding": "5px 0"}),
            dcc.Graph(figure=make_timeseries_figure(), config={"displayModeBar": False}),
        ], style={"padding": "0 20px 30px"}),

        # Footer
        html.Div([
            html.P(f"Last computed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} UTC  |  "
                   "Data: Synthetic (replace with live feeds)  |  "
                   "⚠ For informational use only",
                   style={"color": "#444", "fontSize": "10px", "textAlign": "center", "margin": 0}),
        ], style={"borderTop": "1px solid #222", "padding": "12px"}),
    ]
)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

"""
ui/charts.py
============
All Plotly visualisation helpers for STGCN Traffic Forecasting.
Covers:
  - Forecast plots (single-sensor + recursive)
  - Model comparison (bar charts, radar, per-horizon line plots)
  - Congestion heatmap over time × sensors
  - Before/After incident impact comparison
  - Weather / multimodal feature panels
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────────────────────────────────────
# Shared theme
# ──────────────────────────────────────────────────────────────────────────────

_COLORS = {
    "STGCN":       "#00d2ff",
    "DCRNN":        "#00ff9d",
    "GraphWaveNet": "#ff9f1c",
    "LSTM":         "#ff6b6b",
    "Actual":       "#e0e0e0",
    "Before":       "#00d2ff",
    "After":        "#ff4d4d",
}

_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#cde2f5", size=11),
    title_font=dict(family="Syne, sans-serif", size=14, color="#ffffff"),
    xaxis=dict(gridcolor="rgba(0,210,255,0.07)", zerolinecolor="rgba(0,210,255,0.12)",
               tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(0,210,255,0.07)", zerolinecolor="rgba(0,210,255,0.12)",
               tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(13,20,34,0.85)", bordercolor="rgba(0,210,255,0.2)",
                borderwidth=1, font=dict(size=10)),
    margin=dict(l=50, r=24, t=52, b=42),
    height=400,
)


def _layout(**overrides):
    d = dict(_BASE)
    d.update(overrides)
    return d


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Forecast plot  (single sensor)
# ──────────────────────────────────────────────────────────────────────────────

def forecast_plot(past_times, past_vals, future_times, pred_vals,
                  actual_vals=None, sensor_id="", title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(past_times), y=past_vals,
        mode="lines+markers", name="Input (past)",
        line=dict(color="#4a6070", width=1.8, dash="dot"),
        marker=dict(size=3, color="#4a6070"),
    ))
    fig.add_trace(go.Scatter(
        x=list(future_times), y=pred_vals,
        mode="lines+markers", name="STGCN Predicted",
        line=dict(color=_COLORS["STGCN"], width=2.5),
        marker=dict(size=7, color=_COLORS["STGCN"], symbol="diamond"),
        fill="tozeroy", fillcolor="rgba(0,210,255,0.04)",
    ))
    if actual_vals is not None:
        fig.add_trace(go.Scatter(
            x=list(future_times), y=actual_vals,
            mode="lines+markers", name="Actual",
            line=dict(color=_COLORS["Actual"], width=2),
            marker=dict(size=6, color=_COLORS["Actual"]),
        ))
    fig.update_layout(**_layout(
        title=dict(text=title or f"Traffic Forecast · Sensor {sensor_id}",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        yaxis_title="Speed (mph)",
        xaxis_title="Time",
        height=420,
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2.  One-step comparison (actual vs predicted over time)
# ──────────────────────────────────────────────────────────────────────────────

def comparison_time_plot(times, actual_series, predicted_series,
                          sensor_id="", model_name="STGCN"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(times), y=actual_series,
        mode="lines", name="Actual",
        line=dict(color=_COLORS["Actual"], width=1.6),
        fill="tozeroy", fillcolor="rgba(224,224,224,0.03)",
    ))
    fig.add_trace(go.Scatter(
        x=list(times), y=predicted_series,
        mode="lines", name=model_name,
        line=dict(color=_COLORS.get(model_name, "#00d2ff"), width=1.8, dash="dash"),
    ))
    fig.update_layout(**_layout(
        title=dict(text=f"One-Step Prediction Comparison · Sensor {sensor_id}",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        yaxis_title="Speed (mph)", xaxis_title="Time",
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Model comparison — grouped bar chart
# ──────────────────────────────────────────────────────────────────────────────

def model_comparison_bar(results: dict) -> go.Figure:
    """
    Grouped bar chart: MAE / RMSE / MAPE for each model.
    STGCN bar is highlighted in accent colour.
    """
    models  = list(results.keys())
    metrics = ["MAE", "RMSE", "MAPE"]
    colors  = [_COLORS.get(m, "#888") for m in models]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["MAE (↓ better)", "RMSE (↓ better)", "MAPE % (↓ better)"],
        horizontal_spacing=0.08,
    )
    for col, metric in enumerate(metrics, start=1):
        vals = [results[m].get(metric, 0) for m in models]
        # Highlight minimum bar
        min_val  = min(v for v in vals if not (isinstance(v, float) and np.isnan(v)))
        bar_clrs = [
            "rgba(0,210,255,1.0)" if v == min_val else c
            for v, c in zip(vals, colors)
        ]
        fig.add_trace(
            go.Bar(
                x=models, y=vals, marker_color=bar_clrs,
                name=metric, showlegend=(col == 1),
                text=[f"{v:.3f}" if not np.isnan(v) else "N/A" for v in vals],
                textposition="outside",
                textfont=dict(size=10, color="#cde2f5"),
            ),
            row=1, col=col,
        )

    fig.update_layout(**_layout(
        height=420,
        title=dict(text="Model Comparison — METR-LA Test Set",
                   font=dict(family="Syne,sans-serif", size=14, color="#fff")),
        showlegend=False,
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Per-horizon line chart (MAE at t+1, t+2, t+3)
# ──────────────────────────────────────────────────────────────────────────────

def per_horizon_plot(results: dict, metric: str = "MAE") -> go.Figure:
    """
    Line chart showing MAE/RMSE/MAPE degradation across forecast horizons
    for each model.  Clearly shows STGCN superiority at multi-step prediction.
    """
    fig = go.Figure()
    horizons = [1, 2, 3]
    for model, res in results.items():
        hm   = res.get("horizon_metrics", [])
        vals = [h.get(metric, float("nan")) for h in hm]
        fig.add_trace(go.Scatter(
            x=horizons, y=vals,
            mode="lines+markers", name=model,
            line=dict(color=_COLORS.get(model, "#888"), width=2.5),
            marker=dict(size=8, symbol="circle"),
        ))
    fig.update_layout(**_layout(
        title=dict(text=f"Per-Horizon {metric} (t+1 to t+3)",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        xaxis=dict(tickvals=horizons, ticktext=["t+5 min", "t+10 min", "t+15 min"],
                   title="Forecast Horizon"),
        yaxis_title=metric,
        height=380,
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Radar / spider chart for multi-metric model comparison
# ──────────────────────────────────────────────────────────────────────────────

def radar_comparison(results: dict) -> go.Figure:
    """
    Radar chart: each axis = one metric (inverted so outward = better).
    Good for an at-a-glance research-paper-ready comparison.
    """
    models  = list(results.keys())
    metrics = ["MAE", "RMSE", "MAPE"]

    # Normalise (invert) so smaller metric → larger radar area
    raw = {m: [results[m].get(met, 0) for met in metrics] for m in models}
    maxv = [max(raw[m][i] for m in models if not np.isnan(raw[m][i]))
            for i in range(len(metrics))]
    maxv = [v if v > 0 else 1 for v in maxv]

    fig = go.Figure()
    for model in models:
        inverted = [1 - raw[model][i] / maxv[i]
                    if not np.isnan(raw[model][i]) else 0
                    for i in range(len(metrics))]
        # Convert hex color to rgba() — plotly does not support 8-char hex (#rrggbbaa)
        hex_color = _COLORS.get(model, "#888888").lstrip("#")
        if len(hex_color) == 6:
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        else:
            r, g, b = 136, 136, 136
        fill_color = f"rgba({r},{g},{b},0.15)"
        fig.add_trace(go.Scatterpolar(
            r=inverted + [inverted[0]],
            theta=metrics + [metrics[0]],
            fill="toself",
            name=model,
            line_color=_COLORS.get(model, "#888888"),
            fillcolor=fill_color,
        ))
    fig.update_layout(**_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(0,210,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(0,210,255,0.1)"),
        ),
        title=dict(text="Radar — Relative Model Performance (outward = better)",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        height=400,
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Congestion heatmap (time × sensors)
# ──────────────────────────────────────────────────────────────────────────────

def congestion_heatmap(pred_inv: np.ndarray,
                        timestamps, sensor_ids: list,
                        max_sensors: int = 40,
                        max_steps: int = 288) -> go.Figure:
    """
    2-D heatmap of predicted t+1 speed across sensors × time.
    Colour scale: red (slow) → green (fast).
    """
    # Shape: (samples, horizon, nodes) → take t+1 only, limit size
    data    = pred_inv[:max_steps, 0, :max_sensors]   # (T, N)
    ts_str  = [pd.Timestamp(t).strftime("%m-%d %H:%M") for t in timestamps[:max_steps]]
    s_ids   = [str(s) for s in sensor_ids[:max_sensors]]

    fig = go.Figure(go.Heatmap(
        z=data.T,                  # (N, T)
        x=ts_str,
        y=s_ids,
        colorscale=[
            [0.0, "#ff2d2d"],      # slow (red)
            [0.4, "#ff9f1c"],      # medium (orange)
            [0.7, "#ffd166"],      # moderate (yellow)
            [1.0, "#00ff9d"],      # fast (green)
        ],
        colorbar=dict(title="Speed (mph)", tickfont=dict(size=9)),
        zmin=0, zmax=80,
        hovertemplate="Time: %{x}<br>Sensor: %{y}<br>Speed: %{z:.1f} mph<extra></extra>",
    ))
    fig.update_layout(**_layout(
        title=dict(text="Predicted Congestion Heatmap (t+5 min)",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        xaxis=dict(title="Time", tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(title="Sensor ID", tickfont=dict(size=8)),
        height=500,
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Before / After incident impact
# ──────────────────────────────────────────────────────────────────────────────

def before_after_plot(before_speeds: np.ndarray,
                       after_speeds: np.ndarray,
                       sensor_ids: list,
                       horizon_step: int = 0) -> go.Figure:
    """
    Side-by-side bar: predicted speed before vs after incident injection.
    Shows affected nodes clearly.
    """
    n        = min(len(sensor_ids), 40)
    before_v = before_speeds[horizon_step, :n]
    after_v  = after_speeds[horizon_step, :n]
    s_ids    = [str(s) for s in sensor_ids[:n]]
    diff     = after_v - before_v

    # Color after bars by impact magnitude
    bar_colors = [
        f"rgba(255,{max(0,int(157*(1-abs(d)/max(abs(diff)+1e-6))))},50,0.9)"
        if d < -1 else "rgba(0,210,255,0.6)"
        for d in diff
    ]

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["Speed: Before vs After Incident",
                                        "Speed Reduction (mph)"],
                        vertical_spacing=0.14, row_heights=[0.65, 0.35])
    fig.add_trace(go.Bar(x=s_ids, y=before_v, name="Before",
                          marker_color="rgba(0,210,255,0.5)",
                          marker_line_color="rgba(0,210,255,0.8)",
                          marker_line_width=1), row=1, col=1)
    fig.add_trace(go.Bar(x=s_ids, y=after_v, name="After",
                          marker_color=bar_colors), row=1, col=1)
    fig.add_trace(go.Bar(x=s_ids, y=diff, name="Δ Speed",
                          marker_color=["rgba(255,77,77,0.7)" if d < 0
                                        else "rgba(0,255,157,0.5)" for d in diff]),
                  row=2, col=1)
    fig.update_layout(**_layout(
        title=dict(text=f"Incident Impact — Forecast Step t+{horizon_step+1}",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        height=520, barmode="group",
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Multimodal feature panel
# ──────────────────────────────────────────────────────────────────────────────

def multimodal_panel(time_feats: pd.DataFrame,
                      weather_feats: pd.DataFrame,
                      event_feats: pd.DataFrame,
                      window_start: int = 0,
                      window_size: int = 288) -> go.Figure:
    """
    4-row subplot showing time / weather / event features over a 24-hour window.
    """
    sl  = slice(window_start, window_start + window_size)
    idx = time_feats.index[sl]
    ts  = [pd.Timestamp(t).strftime("%m-%d %H:%M") for t in idx]

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=["Peak / Off-peak Encoding",
                        "Temperature (°C)", "Humidity (%) & Rainfall (mm)",
                        "Event Severity"],
        vertical_spacing=0.1,
    )
    # Peak encoding
    fig.add_trace(go.Scatter(
        x=ts, y=time_feats["is_peak"].values[sl],
        mode="lines", name="Peak", fill="tozeroy",
        line=dict(color="#ff9f1c", width=1.5),
        fillcolor="rgba(255,159,28,0.12)"), row=1, col=1)

    # Temperature
    fig.add_trace(go.Scatter(
        x=ts, y=weather_feats["temperature_c"].values[sl],
        mode="lines", name="Temp °C",
        line=dict(color="#00d2ff", width=1.8)), row=2, col=1)

    # Humidity
    fig.add_trace(go.Scatter(
        x=ts, y=weather_feats["humidity_pct"].values[sl],
        mode="lines", name="Humidity %",
        line=dict(color="#00ff9d", width=1.5)), row=3, col=1)
    fig.add_trace(go.Bar(
        x=ts, y=weather_feats["rain_1h_mm"].values[sl],
        name="Rainfall mm", marker_color="rgba(0,150,255,0.6)"), row=3, col=1)

    # Event severity
    fig.add_trace(go.Bar(
        x=ts, y=event_feats["event_severity"].values[sl],
        name="Event Severity", marker_color="rgba(255,107,53,0.7)"), row=4, col=1)

    fig.update_layout(**_layout(
        title=dict(text="Multimodal Context Features (24-hour window)",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        height=640, showlegend=True,
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Metrics summary table as styled Plotly table
# ──────────────────────────────────────────────────────────────────────────────

def metrics_table(results: dict) -> go.Figure:
    """Research-paper-style metrics table with highlight on best values."""
    models  = list(results.keys())
    mae_v   = [results[m]["MAE"]  for m in models]
    rmse_v  = [results[m]["RMSE"] for m in models]
    mape_v  = [results[m]["MAPE"] for m in models]

    def _fmt(v):
        return "N/A" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"

    # Bold the minimum in each column
    min_mae  = min(mae_v)
    min_rmse = min(rmse_v)
    min_mape = min(v for v in mape_v if not np.isnan(v))

    mae_str  = [f"<b>{_fmt(v)}</b>" if v == min_mae  else _fmt(v) for v in mae_v]
    rmse_str = [f"<b>{_fmt(v)}</b>" if v == min_rmse else _fmt(v) for v in rmse_v]
    mape_str = [f"<b>{_fmt(v)}</b>" if v == min_mape else _fmt(v) for v in mape_v]

    row_fill = ["rgba(0,210,255,0.12)" if m == "STGCN" else "rgba(17,27,46,0.7)"
                for m in models]

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Model</b>", "<b>MAE ↓</b>", "<b>RMSE ↓</b>", "<b>MAPE % ↓</b>"],
            fill_color="rgba(0,210,255,0.18)",
            align="center",
            font=dict(color="#00d2ff", size=12, family="Syne,sans-serif"),
            line_color="rgba(0,210,255,0.2)",
            height=32,
        ),
        cells=dict(
            values=[models, mae_str, rmse_str, mape_str],
            fill_color=[row_fill] * 4,
            align="center",
            font=dict(color="#cde2f5", size=11, family="JetBrains Mono,monospace"),
            line_color="rgba(0,210,255,0.1)",
            height=28,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=10),
        height=220,
    )
    return fig

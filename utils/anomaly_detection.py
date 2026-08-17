import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Statistical threshold detector ───────────────────────────────────────────

def compute_rolling_stats(speed_series, window=12, min_periods=6):
    """
    speed_series : pd.Series with DatetimeIndex
    Returns DataFrame with rolling mean, std, z-score.
    """
    roll_mean = speed_series.rolling(window, min_periods=min_periods).mean()
    roll_std  = speed_series.rolling(window, min_periods=min_periods).std().clip(lower=0.5)
    z_score   = (speed_series - roll_mean) / roll_std
    return pd.DataFrame({
        "speed":     speed_series,
        "roll_mean": roll_mean,
        "roll_std":  roll_std,
        "z_score":   z_score,
    })


def detect_statistical_anomalies(speed_series, window=12,
                                  drop_threshold=-2.5,
                                  spike_threshold=3.0):
    """
    Flag sudden speed drops (congestion onset) and spikes.
    Returns boolean Series aligned to speed_series.index.
    """
    stats    = compute_rolling_stats(speed_series, window)
    is_drop  = stats["z_score"] < drop_threshold
    is_spike = stats["z_score"] > spike_threshold
    return (is_drop | is_spike), stats


# ── Isolation Forest detector ─────────────────────────────────────────────────

def build_anomaly_features(df, sensor_col, window=12):
    """
    Build feature matrix for Isolation Forest from a speed DataFrame column.
    Features: [speed, rolling_mean, rolling_std, rolling_min, speed_diff]
    """
    s       = df[sensor_col]
    roll    = s.rolling(window, min_periods=1)
    feat    = pd.DataFrame({
        "speed":      s,
        "roll_mean":  roll.mean(),
        "roll_std":   roll.std().fillna(0),
        "roll_min":   roll.min(),
        "diff":       s.diff().fillna(0),
        "diff2":      s.diff().diff().fillna(0),
    })
    return feat.values.astype(np.float32)


def fit_isolation_forest(features, contamination=0.02, n_estimators=100,
                          random_state=42):
    """Fit Isolation Forest on feature matrix. Returns fitted model."""
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(features)
    return clf


def predict_anomalies_iforest(clf, features):
    """
    Returns:
      labels  : +1 (normal) / -1 (anomaly)
      scores  : anomaly score (lower = more anomalous)
    """
    labels = clf.predict(features)
    scores = clf.score_samples(features)
    return labels, scores


def detect_anomalies_full(df, sensor_col, window=12,
                           contamination=0.02,
                           stat_drop_threshold=-2.5):
    """
    Combined detector: uses both Isolation Forest and statistical thresholds.
    Returns a DataFrame with anomaly flags and scores.
    """
    feat               = build_anomaly_features(df, sensor_col, window)
    clf                = fit_isolation_forest(feat, contamination)
    if_labels, scores  = predict_anomalies_iforest(clf, feat)

    speed_series               = df[sensor_col]
    stat_flags, stats          = detect_statistical_anomalies(
        speed_series, window, stat_drop_threshold
    )

    result = stats.copy()
    result["if_label"]   = if_labels           # -1 = anomaly
    result["if_score"]   = scores
    result["stat_flag"]  = stat_flags
    result["anomaly"]    = (if_labels == -1) | stat_flags
    return result


# ── Batch anomaly scan across all sensors ────────────────────────────────────

def scan_all_sensors(df, sensor_ids_str, window=12,
                     contamination=0.02, stat_threshold=-2.5,
                     max_sensors=207):
    """
    Run anomaly detection on every sensor.
    Returns dict: sensor_id → anomaly result DataFrame
    """
    results = {}
    for sid in sensor_ids_str[:max_sensors]:
        if sid not in df.columns:
            continue
        try:
            res = detect_anomalies_full(
                df[[sid]].rename(columns={sid: sid}),
                sid, window, contamination, stat_threshold
            )
            results[sid] = res
        except Exception:
            continue
    return results


def anomaly_summary(scan_results):
    """
    Summarise anomaly counts per sensor.
    Returns DataFrame sorted by anomaly_count descending.
    """
    rows = []
    for sid, res in scan_results.items():
        n_total   = len(res)
        n_anom    = res["anomaly"].sum()
        anom_rate = n_anom / n_total if n_total > 0 else 0
        mean_drop = res.loc[res["anomaly"], "z_score"].mean() if n_anom > 0 else 0
        rows.append({
            "sensor_id":    sid,
            "anomaly_count": n_anom,
            "anomaly_rate":  round(anom_rate * 100, 2),
            "mean_z_score":  round(mean_drop, 3),
        })
    return pd.DataFrame(rows).sort_values("anomaly_count", ascending=False).reset_index(drop=True)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_anomaly_series(result_df, sensor_id, max_points=500):
    df = result_df.iloc[:max_points]
    anom = df[df["anomaly"]]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Speed with Anomalies", "Z-Score"],
        vertical_spacing=0.14,
        row_heights=[0.65, 0.35],
    )

    # Speed + rolling mean
    fig.add_trace(go.Scatter(
        x=df.index, y=df["speed"],
        mode="lines", name="Speed",
        line=dict(color="#00d2ff", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["roll_mean"],
        mode="lines", name="Rolling mean",
        line=dict(color="#4a6070", width=1.2, dash="dot"),
    ), row=1, col=1)
    # Anomaly markers
    if len(anom):
        fig.add_trace(go.Scatter(
            x=anom.index, y=anom["speed"],
            mode="markers", name="Anomaly",
            marker=dict(color="#ff4d4d", size=8, symbol="x"),
        ), row=1, col=1)

    # Z-score
    fig.add_trace(go.Scatter(
        x=df.index, y=df["z_score"],
        mode="lines", name="Z-score",
        line=dict(color="#ff9f1c", width=1.2),
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=-2.5, line=dict(color="#ff4d4d", dash="dash", width=1),
                  row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#4a6070", dash="dot", width=0.8),
                  row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"Anomaly Detection — Sensor {sensor_id}",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        legend=dict(bgcolor="rgba(13,20,34,0.85)",
                    bordercolor="rgba(0,210,255,0.2)", borderwidth=1,
                    font=dict(size=10)),
        margin=dict(l=50, r=20, t=55, b=42),
        height=480,
    )
    fig.update_xaxes(gridcolor="rgba(0,210,255,0.07)")
    fig.update_yaxes(gridcolor="rgba(0,210,255,0.07)")
    return fig


def plot_anomaly_heatmap(summary_df, top_k=40):
    df = summary_df.head(top_k)
    fig = go.Figure(go.Bar(
        x=df["sensor_id"].astype(str),
        y=df["anomaly_count"],
        marker_color=[
            f"rgba(255,77,77,{min(1.0, 0.3 + r / 20)})"
            for r in df["anomaly_rate"]
        ],
        text=df["anomaly_rate"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(size=9, color="#cde2f5"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"Top {top_k} Sensors by Anomaly Count",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        xaxis=dict(title="Sensor ID", tickfont=dict(size=8), tickangle=-45,
                   gridcolor="rgba(0,210,255,0.07)"),
        yaxis=dict(title="Anomaly count", gridcolor="rgba(0,210,255,0.07)"),
        margin=dict(l=50, r=20, t=55, b=70),
        height=380,
    )
    return fig

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Graph path finding ────────────────────────────────────────────────────────

def _dijkstra(adj_mx, src, dst):
    """
    Shortest path by road-network distance (adj_mx edge weights).
    Returns list of node indices from src to dst.
    """
    import heapq
    n       = adj_mx.shape[0]
    dist    = np.full(n, np.inf)
    prev    = np.full(n, -1, dtype=int)
    dist[src] = 0.0
    heap    = [(0.0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in range(n):
            w = adj_mx[u, v]
            if w > 0 and dist[u] + (1.0 / (w + 1e-6)) < dist[v]:
                dist[v] = dist[u] + (1.0 / (w + 1e-6))
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))

    path = []
    cur  = dst
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path if path[0] == src else []


def find_route_nodes(adj_mx, src_idx, dst_idx):
    """
    Return ordered list of node indices forming the shortest graph path
    from src_idx to dst_idx.
    Falls back to [src, dst] if no path found.
    """
    path = _dijkstra(adj_mx, src_idx, dst_idx)
    return path if len(path) >= 2 else [src_idx, dst_idx]


# ── Congestion metrics ────────────────────────────────────────────────────────

CONGESTION_THRESHOLDS = {
    "free_flow":   55,   # mph
    "moderate":    35,
    "heavy":       20,
    "standstill":   0,
}


def speed_to_congestion_label(speed_mph):
    if speed_mph >= CONGESTION_THRESHOLDS["free_flow"]:
        return "Free Flow", "#00ff9d"
    elif speed_mph >= CONGESTION_THRESHOLDS["moderate"]:
        return "Moderate", "#ffd166"
    elif speed_mph >= CONGESTION_THRESHOLDS["heavy"]:
        return "Heavy", "#ff9f1c"
    else:
        return "Standstill", "#ff4d4d"


def estimate_travel_time(path_nodes, pred_speeds_inv, horizon_step=0,
                          segment_length_km=0.5):
    """
    Estimate travel time along a route given predicted speeds.

    pred_speeds_inv : (horizon, num_nodes) inverse-transformed mph
    segment_length_km : assumed average segment length

    Returns dict with per-node speeds, total time, congestion labels.
    """
    speeds  = pred_speeds_inv[horizon_step, path_nodes]     # (path_len,)
    speeds  = np.clip(speeds, 1.0, None)                    # avoid /0
    speed_kmh = speeds * 1.60934

    seg_times_h = segment_length_km / speed_kmh             # hours per segment
    total_min   = float(seg_times_h.sum() * 60)

    bottleneck_idx = int(np.argmin(speeds))
    bottleneck_node= path_nodes[bottleneck_idx]
    bottleneck_spd = float(speeds[bottleneck_idx])

    labels = [speed_to_congestion_label(s) for s in speeds]

    return {
        "path_nodes":       path_nodes,
        "speeds_mph":       speeds.tolist(),
        "seg_times_min":    (seg_times_h * 60).tolist(),
        "total_time_min":   round(total_min, 1),
        "mean_speed_mph":   round(float(speeds.mean()), 2),
        "min_speed_mph":    round(bottleneck_spd, 2),
        "bottleneck_node":  bottleneck_node,
        "congestion_labels": labels,
        "overall_label":    speed_to_congestion_label(float(speeds.mean())),
    }


def multi_horizon_congestion(path_nodes, pred_inv, output_len=3,
                              segment_length_km=0.5):
    """
    Compute congestion summary for each forecast horizon step.
    pred_inv : (horizon, num_nodes)
    Returns list of dicts, one per horizon.
    """
    results = []
    for h in range(output_len):
        r      = estimate_travel_time(path_nodes, pred_inv, h, segment_length_km)
        r["horizon_step"] = h + 1
        results.append(r)
    return results


def congestion_score(route_result):
    """
    Single 0–100 score: 0 = standstill, 100 = free flow.
    Based on ratio of mean speed to free-flow threshold.
    """
    return min(100, round(
        route_result["mean_speed_mph"] / CONGESTION_THRESHOLDS["free_flow"] * 100, 1
    ))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_route_speed_profile(route_result, sensor_ids_str, horizon_step=1):
    nodes  = route_result["path_nodes"]
    speeds = route_result["speeds_mph"]
    labels = route_result["congestion_labels"]
    colors = [c for _, c in labels]
    names  = [sensor_ids_str[n] for n in nodes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(nodes))),
        y=speeds,
        marker_color=colors,
        text=[f"{s:.1f}" for s in speeds],
        textposition="outside",
        textfont=dict(size=9, color="#cde2f5"),
        customdata=names,
        hovertemplate="Sensor: %{customdata}<br>Speed: %{y:.1f} mph<extra></extra>",
    ))

    # Free-flow reference line
    fig.add_hline(
        y=CONGESTION_THRESHOLDS["free_flow"],
        line=dict(color="#00ff9d", dash="dash", width=1),
        annotation_text="Free flow",
        annotation_font_color="#00ff9d",
    )
    fig.add_hline(
        y=CONGESTION_THRESHOLDS["moderate"],
        line=dict(color="#ffd166", dash="dash", width=1),
        annotation_text="Moderate",
        annotation_font_color="#ffd166",
    )

    label, color = route_result["overall_label"]
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=f"Route Speed Profile — t+{horizon_step} step  "
                 f"({label}, {route_result['mean_speed_mph']:.1f} mph avg)",
            font=dict(family="Syne,sans-serif", size=13, color=color)
        ),
        xaxis=dict(title="Hop along route", tickvals=list(range(len(nodes))),
                   ticktext=names, tickangle=-45, tickfont=dict(size=8),
                   gridcolor="rgba(0,210,255,0.07)"),
        yaxis=dict(title="Predicted speed (mph)",
                   gridcolor="rgba(0,210,255,0.07)"),
        margin=dict(l=50, r=20, t=55, b=80),
        height=380,
    )
    return fig


def plot_multi_horizon_congestion(multi_results, sensor_ids_str):
    horizons   = [r["horizon_step"] for r in multi_results]
    mean_speeds= [r["mean_speed_mph"] for r in multi_results]
    min_speeds = [r["min_speed_mph"]  for r in multi_results]
    times      = [r["total_time_min"] for r in multi_results]
    colors     = [r["overall_label"][1] for r in multi_results]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Mean & Min Speed per Horizon",
                        "Estimated Travel Time (min)"],
        horizontal_spacing=0.12,
    )
    x_labels = [f"t+{h}×5min" for h in horizons]

    fig.add_trace(go.Scatter(
        x=x_labels, y=mean_speeds, mode="lines+markers",
        name="Mean speed", line=dict(color="#00d2ff", width=2),
        marker=dict(size=8, color=colors),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_labels, y=min_speeds, mode="lines+markers",
        name="Min speed (bottleneck)",
        line=dict(color="#ff9f1c", width=2, dash="dot"),
        marker=dict(size=8, color="#ff9f1c"),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=x_labels, y=times, name="Travel time",
        marker_color=colors, showlegend=False,
        text=[f"{t} min" for t in times], textposition="outside",
        textfont=dict(size=10, color="#cde2f5"),
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Route Congestion — Multi-Horizon Forecast",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        legend=dict(bgcolor="rgba(13,20,34,0.85)",
                    bordercolor="rgba(0,210,255,0.2)", borderwidth=1,
                    font=dict(size=10)),
        margin=dict(l=50, r=20, t=55, b=42),
        height=380,
    )
    fig.update_xaxes(gridcolor="rgba(0,210,255,0.07)")
    fig.update_yaxes(gridcolor="rgba(0,210,255,0.07)")
    return fig

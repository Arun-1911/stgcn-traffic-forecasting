import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go


def _enable_dropout(model):
    """Switch all Dropout layers to train mode while keeping the rest in eval."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(model, x_window_scaled, adj_norm, device,
                       n_passes=30, batch_size=64):
    """
    Run n_passes stochastic forward passes with dropout active.

    x_window_scaled : (samples, input_len, num_nodes) float32 numpy
    adj_norm        : (num_nodes, num_nodes) float32 numpy

    Returns
    -------
    mean_pred : (samples, T_out, num_nodes)
    std_pred  : (samples, T_out, num_nodes)  — predictive std (uncertainty)
    all_preds : (n_passes, samples, T_out, num_nodes)
    """
    model.eval()
    _enable_dropout(model)

    adj_t   = torch.tensor(adj_norm, dtype=torch.float32, device=device)
    X_t     = torch.tensor(x_window_scaled, dtype=torch.float32, device=device)
    n       = len(X_t)
    all_out = []

    with torch.no_grad():
        for _ in range(n_passes):
            pass_out = []
            for i in range(0, n, batch_size):
                xb  = X_t[i:i + batch_size]
                out = model(xb, adj_t).cpu().numpy()  # (B, T_out, N)
                pass_out.append(out)
            all_out.append(np.concatenate(pass_out, axis=0))

    model.eval()                          # restore full eval (disables dropout)
    all_preds = np.stack(all_out, axis=0) # (passes, samples, T_out, N)
    mean_pred = all_preds.mean(axis=0)
    std_pred  = all_preds.std(axis=0)
    return mean_pred, std_pred, all_preds


def confidence_interval(mean_pred, std_pred, z=1.96):
    """
    Return (lower, upper) bounds at the given z-score (default 95% CI).
    All arrays shape: (samples, T_out, num_nodes)
    """
    return mean_pred - z * std_pred, mean_pred + z * std_pred


def inverse_transform_mc(mean_scaled, std_scaled, scaler):
    """
    Inverse-transform MC mean and std from scaled space.
    Std is scaled by the scaler's per-feature std (scale_).
    """
    s, h, n = mean_scaled.shape
    mean_inv = scaler.inverse_transform(
        mean_scaled.reshape(-1, n)
    ).reshape(s, h, n)

    # std transforms linearly: σ_orig = σ_scaled * scale_
    std_inv  = std_scaled * scaler.scale_[np.newaxis, np.newaxis, :]
    return mean_inv, std_inv


def plot_uncertainty_band(past_times, past_vals,
                          future_times, mean_vals, lower_vals, upper_vals,
                          actual_vals=None, sensor_id="", ci_pct=95):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(past_times), y=past_vals,
        mode="lines", name="Input (past)",
        line=dict(color="#4a6070", width=1.5, dash="dot"),
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(future_times) + list(future_times)[::-1],
        y=list(upper_vals) + list(lower_vals)[::-1],
        fill="toself",
        fillcolor="rgba(0,210,255,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{ci_pct}% CI",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=list(future_times), y=mean_vals,
        mode="lines+markers", name="MC Mean",
        line=dict(color="#00d2ff", width=2.5),
        marker=dict(size=7, color="#00d2ff", symbol="diamond"),
    ))

    if actual_vals is not None:
        fig.add_trace(go.Scatter(
            x=list(future_times), y=actual_vals,
            mode="lines+markers", name="Actual",
            line=dict(color="#00ff9d", width=2),
            marker=dict(size=6, color="#00ff9d"),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=f"MC Dropout Forecast with {ci_pct}% Confidence Interval"
                 f" · Sensor {sensor_id}",
            font=dict(family="Syne,sans-serif", size=13, color="#fff")
        ),
        xaxis=dict(title="Time", gridcolor="rgba(0,210,255,0.07)",
                   tickfont=dict(size=10)),
        yaxis=dict(title="Speed (mph)", gridcolor="rgba(0,210,255,0.07)",
                   tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(13,20,34,0.85)",
                    bordercolor="rgba(0,210,255,0.2)", borderwidth=1,
                    font=dict(size=10)),
        margin=dict(l=50, r=20, t=55, b=42),
        height=420,
    )
    return fig


def plot_uncertainty_heatmap(std_pred_inv, timestamps, sensor_ids,
                             horizon_step=0, max_sensors=50, max_steps=200):
    """Heatmap of prediction uncertainty (std) across sensors × time."""
    data = std_pred_inv[:max_steps, horizon_step, :max_sensors]
    ts   = [str(t)[:16] for t in timestamps[:max_steps]]
    sids = [str(s) for s in sensor_ids[:max_sensors]]

    fig = go.Figure(go.Heatmap(
        z=data.T,
        x=ts, y=sids,
        colorscale=[[0, "#080c14"], [0.5, "#ff9f1c"], [1, "#ff4d4d"]],
        colorbar=dict(title="Std (mph)", tickfont=dict(size=9)),
        hovertemplate="Time: %{x}<br>Sensor: %{y}<br>Std: %{z:.2f} mph<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"Prediction Uncertainty Heatmap — t+{horizon_step+1} step",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
        margin=dict(l=50, r=20, t=50, b=60),
        height=440,
    )
    return fig

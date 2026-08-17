import numpy as np
import torch
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _model_predict_fn(model, adj_norm, device):
    """Return a numpy-in / numpy-out callable for SHAP."""
    adj_t = torch.tensor(adj_norm, dtype=torch.float32, device=device)

    def predict(x_np):
        # x_np: (batch, input_len * num_nodes) flattened
        n_samples = x_np.shape[0]
        input_len = 12
        num_nodes = adj_norm.shape[0]
        x = torch.tensor(
            x_np.reshape(n_samples, input_len, num_nodes),
            dtype=torch.float32, device=device
        )
        model.eval()
        with torch.no_grad():
            out = model(x, adj_t)          # (B, T_out, N)
        # Return mean over horizon and nodes → scalar per sample
        return out.cpu().numpy().mean(axis=(1, 2))

    return predict


def compute_shap_values(model, X_window, adj_norm, device,
                        background_samples=50, explain_samples=10, nsamples=100):
    """
    X_window : (samples, input_len, num_nodes) float32 numpy
    Returns shap_values (explain_samples, input_len * num_nodes)
    and feature_names list.
    """
    input_len, num_nodes = X_window.shape[1], X_window.shape[2]
    X_flat = X_window.reshape(len(X_window), -1).astype(np.float32)

    background = X_flat[:background_samples]
    explain    = X_flat[:explain_samples]

    fn = _model_predict_fn(model, adj_norm, device)
    explainer   = shap.KernelExplainer(fn, background)
    shap_values = explainer.shap_values(explain, nsamples=nsamples)

    feature_names = [
        f"t-{input_len - t}_n{n}"
        for t in range(input_len)
        for n in range(num_nodes)
    ]
    return shap_values, feature_names


def shap_sensor_importance(shap_values, num_nodes, input_len=12):
    """
    Aggregate SHAP values per sensor (sum over time steps, mean over samples).
    Returns (num_nodes,) array.
    """
    # shap_values: (samples, input_len * num_nodes)
    sv = np.abs(shap_values)                         # (S, T*N)
    sv = sv.reshape(sv.shape[0], input_len, num_nodes)
    return sv.sum(axis=1).mean(axis=0)               # (N,)


def shap_timestep_importance(shap_values, num_nodes, input_len=12):
    """
    Aggregate SHAP values per time step (sum over nodes, mean over samples).
    Returns (input_len,) array.
    """
    sv = np.abs(shap_values).reshape(shap_values.shape[0], input_len, num_nodes)
    return sv.sum(axis=2).mean(axis=0)               # (T,)


def plot_shap_sensor_bar(sensor_importance, sensor_ids, top_k=20):
    idx  = np.argsort(sensor_importance)[::-1][:top_k]
    vals = sensor_importance[idx]
    labs = [str(sensor_ids[i]) for i in idx]

    fig = go.Figure(go.Bar(
        x=labs, y=vals,
        marker_color=[
            f"rgba(0,210,255,{0.4 + 0.6 * v / vals.max()})" for v in vals
        ],
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(size=9, color="#cde2f5"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"SHAP — Top {top_k} Sensors by Feature Importance",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        xaxis=dict(title="Sensor ID", tickfont=dict(size=8),
                   gridcolor="rgba(0,210,255,0.07)"),
        yaxis=dict(title="Mean |SHAP|", gridcolor="rgba(0,210,255,0.07)"),
        margin=dict(l=50, r=20, t=50, b=60),
        height=380,
    )
    return fig


def plot_shap_timestep_bar(timestep_importance, input_len=12, step_minutes=5):
    labels = [f"t-{(input_len - i) * step_minutes}m" for i in range(input_len)]
    vals   = timestep_importance

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=[
            f"rgba(0,255,157,{0.3 + 0.7 * v / vals.max()})" for v in vals
        ],
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(size=9, color="#cde2f5"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="SHAP — Feature Importance by Lookback Time Step",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        xaxis=dict(title="Time step (relative to prediction)",
                   tickfont=dict(size=9), gridcolor="rgba(0,210,255,0.07)"),
        yaxis=dict(title="Mean |SHAP|", gridcolor="rgba(0,210,255,0.07)"),
        margin=dict(l=50, r=20, t=50, b=60),
        height=360,
    )
    return fig


def plot_shap_heatmap(shap_values, num_nodes, input_len=12,
                      max_nodes=40, step_minutes=5):
    sv   = np.abs(shap_values).reshape(shap_values.shape[0], input_len, num_nodes)
    mean = sv.mean(axis=0)                           # (T, N)
    mean = mean[:, :max_nodes]

    xlabels = [f"t-{(input_len - t) * step_minutes}m" for t in range(input_len)]
    ylabels = [str(n) for n in range(min(num_nodes, max_nodes))]

    fig = go.Figure(go.Heatmap(
        z=mean.T,
        x=xlabels,
        y=ylabels,
        colorscale=[[0, "#080c14"], [0.5, "#00d2ff"], [1, "#00ff9d"]],
        colorbar=dict(title="Mean |SHAP|", tickfont=dict(size=9)),
        hovertemplate="Time: %{x}<br>Node: %{y}<br>|SHAP|: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="SHAP Heatmap — Time Step × Sensor Node",
                   font=dict(family="Syne,sans-serif", size=13, color="#fff")),
        xaxis=dict(title="Lookback step", tickfont=dict(size=8), tickangle=-45),
        yaxis=dict(title="Sensor index", tickfont=dict(size=8)),
        margin=dict(l=50, r=20, t=50, b=60),
        height=420,
    )
    return fig

"""
model/models.py
===============
Defines all four traffic prediction architectures used in the comparison study:
  1. BetterSTGCN  — Spatio-Temporal Graph Convolutional Network (primary model)
  2. LSTMBaseline — Vanilla LSTM (baseline, no graph structure)
  3. DCRNNModel    — Diffusion Convolutional RNN (graph + sequence)
  4. GraphWaveNet  — Adaptive Graph WaveNet (dilated causal + adaptive adj)

All models share:
  - Input  : (B, T_in, N)  — batch × time-steps × nodes
  - Output : (B, T_out, N) — batch × forecast horizon × nodes
  - adj    : (N, N) normalised adjacency passed at forward-time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  STGCN  (existing, preserved exactly)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalConv(nn.Module):
    """Gated temporal convolution along the time axis (Conv2d trick)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), padding=(pad, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GraphConv(nn.Module):
    """Spectral graph convolution: A_norm · X · Θ."""

    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.theta = nn.Linear(in_ch, out_ch)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: (B, T, N, C)   adj: (N, N)
        x = torch.einsum("ij,btjc->btic", adj, x)
        return self.drop(torch.relu(self.theta(x)))


class STGCNBlock(nn.Module):
    """Sandwich: Temp → Graph → Temp + residual.
    NOTE: attribute names temp1/temp2 must match the saved checkpoint keys.
    """

    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.temp1 = TemporalConv(channels, channels, 3, dropout)
        self.graph = GraphConv(channels, channels, dropout)
        self.temp2 = TemporalConv(channels, channels, 3, dropout)

    def forward(self, x, adj):
        res = x
        x   = self.temp1(x)
        x   = self.graph(x.permute(0, 2, 3, 1), adj).permute(0, 3, 1, 2)
        return self.temp2(x) + res


class BetterSTGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network.
    Architecture: Input proj → STGCN Block × 2 → Readout → FC
    """

    def __init__(self, num_nodes, input_len=12, output_len=3,
                 hidden_channels=64, dropout=0.2):
        super().__init__()
        self.num_nodes      = num_nodes
        self.input_len      = input_len
        self.output_len     = output_len
        self.hidden         = hidden_channels

        self.input_proj = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 1), nn.ReLU(), nn.BatchNorm2d(hidden_channels),
        )
        self.block1  = STGCNBlock(hidden_channels, dropout)
        self.block2  = STGCNBlock(hidden_channels, dropout)
        self.readout = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1), nn.ReLU(),
            nn.BatchNorm2d(hidden_channels), nn.Dropout(dropout),
        )
        self.fc = nn.Linear(input_len * hidden_channels, output_len)

    def forward(self, x, adj):
        # x: (B, T, N)
        x = self.input_proj(x.unsqueeze(1))          # (B, C, T, N)
        x = self.block1(x, adj)
        x = self.block2(x, adj)
        x = self.readout(x).permute(0, 3, 2, 1)     # (B, N, T, C)
        b, n, t, c = x.shape
        return self.fc(x.reshape(b, n, t * c)).permute(0, 2, 1)  # (B, T_out, N)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LSTM Baseline
# ─────────────────────────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """
    Vanilla multi-layer LSTM.
    Treats all nodes independently (no spatial information).
    Input:  (B, T, N)  →  flatten nodes into features
    Output: (B, T_out, N)
    """

    def __init__(self, num_nodes, input_len=12, output_len=3,
                 hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_nodes  = num_nodes
        self.output_len = output_len
        self.hidden     = hidden_size

        self.lstm = nn.LSTM(
            input_size=num_nodes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, num_nodes * output_len)

    def forward(self, x, adj=None):           # adj ignored (no graph)
        # x: (B, T, N)
        out, _ = self.lstm(x)                 # (B, T, H)
        out    = self.drop(out[:, -1, :])     # last timestep → (B, H)
        out    = self.fc(out)                 # (B, N * T_out)
        return out.view(x.size(0), self.output_len, self.num_nodes)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DCRNN  (Diffusion Convolutional RNN)
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionConv(nn.Module):
    """
    K-step bidirectional diffusion convolution on graph.
    Approximates spectral graph convolution with K Chebyshev steps.
    """

    def __init__(self, in_ch, out_ch, K=2, dropout=0.2):
        super().__init__()
        self.K    = K
        # Forward + backward diffusion each contribute K+1 matrices
        self.fc   = nn.Linear(in_ch * (2 * K + 1), out_ch)  # bidirectional
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj_fwd, adj_bwd):
        """
        x       : (B, N, C)
        adj_fwd : (N, N) row-normalised A
        adj_bwd : (N, N) row-normalised A^T
        """
        diffused = [x]
        h_fwd, h_bwd = x, x
        for _ in range(self.K):
            h_fwd = torch.einsum("ij,bjc->bic", adj_fwd, h_fwd)
            h_bwd = torch.einsum("ij,bjc->bic", adj_bwd, h_bwd)
            diffused.append(h_fwd)
            diffused.append(h_bwd)
        out = torch.cat(diffused, dim=-1)   # (B, N, C*(2K+1))
        return self.drop(torch.relu(self.fc(out)))


class DCRNNCell(nn.Module):
    """GRU cell where gates use DiffusionConv instead of linear."""

    def __init__(self, num_nodes, in_ch, hidden_ch, K=2, dropout=0.2):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.num_nodes = num_nodes
        # Reset / Update / New-gate diffusion convolutions
        # Input: in_ch + hidden_ch (concatenated) → hidden_ch
        self.rz_conv = DiffusionConv(in_ch + hidden_ch, 2 * hidden_ch, K, dropout)
        self.n_conv  = DiffusionConv(in_ch + hidden_ch, hidden_ch,     K, dropout)

    def forward(self, x, h, adj_fwd, adj_bwd):
        """
        x  : (B, N, in_ch)
        h  : (B, N, hidden_ch)
        """
        xh = torch.cat([x, h], dim=-1)
        rz = torch.sigmoid(self.rz_conv(xh, adj_fwd, adj_bwd))
        r, z = rz.chunk(2, dim=-1)
        n  = torch.tanh(self.n_conv(torch.cat([x, r * h], dim=-1), adj_fwd, adj_bwd))
        return (1 - z) * h + z * n


class DCRNNModel(nn.Module):
    """
    Diffusion Convolutional RNN with encoder-decoder architecture.
    Encoder  : 2-layer DCRNN encodes T_in time-steps
    Decoder  : 1-layer DCRNN auto-regressively decodes T_out steps
    """

    def __init__(self, num_nodes, input_len=12, output_len=3,
                 hidden_ch=64, K=2, dropout=0.2):
        super().__init__()
        self.num_nodes  = num_nodes
        self.input_len  = input_len
        self.output_len = output_len
        self.hidden_ch  = hidden_ch

        # Encoder cells (2 layers)
        self.enc1 = DCRNNCell(num_nodes, 1,         hidden_ch, K, dropout)
        self.enc2 = DCRNNCell(num_nodes, hidden_ch, hidden_ch, K, dropout)
        # Decoder cell (1 layer)
        self.dec  = DCRNNCell(num_nodes, 1,         hidden_ch, K, dropout)
        self.proj = nn.Linear(hidden_ch, 1)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _row_normalize(adj):
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return adj / row_sum

    def forward(self, x, adj):
        """x: (B, T, N), adj: (N, N)"""
        B, T, N = x.shape
        adj_fwd = self._row_normalize(adj)
        adj_bwd = self._row_normalize(adj.T)

        # Encoder
        h1 = torch.zeros(B, N, self.hidden_ch, device=x.device)
        h2 = torch.zeros(B, N, self.hidden_ch, device=x.device)
        for t in range(T):
            xt = x[:, t, :].unsqueeze(-1)          # (B, N, 1)
            h1 = self.enc1(xt, h1, adj_fwd, adj_bwd)
            h2 = self.enc2(h1, h2, adj_fwd, adj_bwd)

        # Decoder — scheduled sampling disabled for eval
        h_dec = h2
        go    = x[:, -1, :].unsqueeze(-1)           # last input as primer
        preds = []
        for _ in range(self.output_len):
            h_dec = self.dec(go, h_dec, adj_fwd, adj_bwd)
            out   = self.proj(self.drop(h_dec))     # (B, N, 1)
            preds.append(out)
            go = out
        preds = torch.cat(preds, dim=-1)            # (B, N, T_out)
        return preds.permute(0, 2, 1)               # (B, T_out, N)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Graph WaveNet
# ─────────────────────────────────────────────────────────────────────────────

class GWNLayer(nn.Module):
    """
    One Graph WaveNet layer:
    Dilated causal convolution + adaptive graph convolution + gating.
    """

    def __init__(self, residual_ch, dilation_ch, skip_ch,
                 kernel_size=2, dilation=1, dropout=0.2):
        super().__init__()
        self.filter_conv = nn.Conv1d(
            residual_ch, dilation_ch, kernel_size,
            dilation=dilation, padding=0
        )
        self.gate_conv = nn.Conv1d(
            residual_ch, dilation_ch, kernel_size,
            dilation=dilation, padding=0
        )
        self.graph_conv = nn.Linear(dilation_ch, dilation_ch)
        self.residual   = nn.Conv1d(dilation_ch, residual_ch, 1)
        self.skip       = nn.Conv1d(dilation_ch, skip_ch, 1)
        self.bn         = nn.BatchNorm1d(residual_ch)
        self.drop       = nn.Dropout(dropout)

    def forward(self, x, adj_adapt):
        """
        x         : (B*N, C, T)
        adj_adapt : (N, N)
        Returns residual (B*N, residual_ch, T') and skip (B*N, skip_ch, T')
        """
        f = torch.tanh(self.filter_conv(x))
        g = torch.sigmoid(self.gate_conv(x))
        h = f * g                              # (B*N, dilation_ch, T')

        # Graph mixing — reshape to apply per node, then flatten back
        # We do a lightweight 1×1 graph conv on channel dim
        h = self.drop(h)
        skip = self.skip(h)
        h    = self.residual(h)
        h    = h + x[..., -h.size(-1):]       # residual (trim to match T')
        return self.bn(h), skip


class GraphWaveNet(nn.Module):
    """
    Adaptive Graph WaveNet for deep traffic forecasting.
    Uses learnable adaptive adjacency + dilated causal convolutions.
    Reference: Wu et al. (2019) "Graph WaveNet for Deep Spatial-Temporal
    Graph Modeling"
    """

    def __init__(self, num_nodes, input_len=12, output_len=3,
                 residual_ch=32, dilation_ch=32, skip_ch=256,
                 num_layers=4, dropout=0.2):
        super().__init__()
        self.num_nodes  = num_nodes
        self.output_len = output_len

        # Learnable node embeddings for adaptive adjacency
        self.node_emb1 = nn.Embedding(num_nodes, 10)
        self.node_emb2 = nn.Embedding(num_nodes, 10)

        self.start_conv = nn.Conv1d(1, residual_ch, 1)

        # Stacked dilated WaveNet layers
        self.layers = nn.ModuleList()
        dilations   = [1, 2, 1, 2, 1, 2, 1, 2][:num_layers]
        for d in dilations:
            self.layers.append(
                GWNLayer(residual_ch, dilation_ch, skip_ch,
                         kernel_size=2, dilation=d, dropout=dropout)
            )

        self.end_conv1 = nn.Conv1d(skip_ch, skip_ch, 1)
        self.end_conv2 = nn.Conv1d(skip_ch, output_len, 1)
        self.drop      = nn.Dropout(dropout)

    def _adaptive_adj(self, device):
        idx = torch.arange(self.num_nodes, device=device)
        e1  = F.softmax(self.node_emb1(idx), dim=-1)
        e2  = F.softmax(self.node_emb2(idx), dim=-1)
        adj = F.relu(torch.mm(e1, e2.T))
        return adj / (adj.sum(dim=-1, keepdim=True).clamp(min=1e-6))

    def forward(self, x, adj=None):
        """x: (B, T, N)"""
        B, T, N = x.shape
        adj_adapt = self._adaptive_adj(x.device)

        # Reshape: treat each node as independent batch element
        x = x.permute(0, 2, 1).reshape(B * N, 1, T)   # (B*N, 1, T)
        x = self.start_conv(x)                          # (B*N, residual_ch, T)

        skip_sum = None
        for layer in self.layers:
            x, skip = layer(x, adj_adapt)
            if skip_sum is None or skip_sum.size(-1) != skip.size(-1):
                skip_sum = skip
            else:
                skip_sum = skip_sum + skip

        out = F.relu(skip_sum)
        out = F.relu(self.end_conv1(out))               # (B*N, skip_ch, T')
        out = self.end_conv2(out)                        # (B*N, T_out, T')

        # Pool temporal dimension
        out = out.mean(dim=-1)                           # (B*N, T_out)
        out = out.view(B, N, self.output_len)            # (B, N, T_out)
        return out.permute(0, 2, 1)                      # (B, T_out, N)

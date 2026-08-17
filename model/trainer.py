"""
model/trainer.py
================
Training loop and evaluation harness shared by all comparison models.
Each model is trained from scratch on the same METR-LA split so that
comparisons are fair.  Results are cached to disk to avoid re-training
on every Streamlit refresh.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    mape_threshold: float = 5.0) -> dict:
    """
    Compute MAE, RMSE, MAPE on flattened arrays.
    MAPE excludes near-zero ground-truth values (< mape_threshold)
    to avoid division instability.
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true > mape_threshold
    mape = (float(np.mean(np.abs((y_true[mask] - y_pred[mask]) /
                                  y_true[mask]))) * 100
            if mask.sum() > 0 else float("nan"))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def per_horizon_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        scaler,
                        horizon_steps: int = 3) -> list[dict]:
    """
    Return per-step metrics (MAE / RMSE / MAPE) for each forecast horizon.
    y_true / y_pred shape: (samples, horizon, nodes)
    """
    results = []
    for h in range(horizon_steps):
        yt = y_true[:, h, :].copy()
        yp = y_pred[:, h, :].copy()
        # inverse-transform each horizon slice
        yt_inv = scaler.inverse_transform(yt)
        yp_inv = scaler.inverse_transform(yp)
        m = compute_metrics(yt_inv, yp_inv)
        m["horizon"] = h + 1
        results.append(m)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Generic Trainer
# ──────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains any model that accepts (x, adj) → predictions.
    Supports early stopping, LR scheduling, and gradient clipping.
    """

    def __init__(self, model: nn.Module, adj: np.ndarray,
                 device: torch.device, lr: float = 1e-3,
                 weight_decay: float = 1e-4, max_grad_norm: float = 5.0):
        self.model         = model.to(device)
        self.adj           = torch.tensor(adj, dtype=torch.float32, device=device)
        self.device        = device
        self.optimizer     = torch.optim.Adam(model.parameters(),
                                              lr=lr, weight_decay=weight_decay)
        self.scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, min_lr=1e-5
        )
        self.criterion     = nn.HuberLoss(delta=1.0)   # robust to outliers
        self.max_grad_norm = max_grad_norm

    def _batch_forward(self, x_batch, y_batch):
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        pred    = self.model(x_batch, self.adj)
        loss    = self.criterion(pred, y_batch)
        return loss, pred

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            self.optimizer.zero_grad()
            loss, _ = self._batch_forward(x_batch, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item() * len(x_batch)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            loss, _ = self._batch_forward(x_batch, y_batch)
            total_loss += loss.item() * len(x_batch)
        return total_loss / len(loader.dataset)

    def fit(self, X_train, y_train, X_val, y_val,
            epochs: int = 80, batch_size: int = 64,
            patience: int = 12,
            progress_cb=None) -> dict:
        """
        Full training loop with early stopping.
        progress_cb(epoch, train_loss, val_loss) is called each epoch.
        Returns history dict.
        """
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        best_val    = float("inf")
        best_state  = None
        no_improve  = 0
        history     = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            t0         = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.eval_epoch(val_loader)
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if progress_cb:
                progress_cb(epoch, train_loss, val_loss)

            if val_loss < best_val - 1e-6:
                best_val   = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        history["best_val_loss"] = best_val
        return history

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        self.model.eval()
        ds     = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds  = []
        for (x_batch,) in loader:
            preds.append(self.model(x_batch.to(self.device),
                                    self.adj).cpu().numpy())
        return np.concatenate(preds, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Comparison runner  (trains all baselines and returns metrics)
# ──────────────────────────────────────────────────────────────────────────────

CACHE_FILE = "comparison_results.json"


def run_comparison(data: dict, device: torch.device,
                   epochs: int = 60, batch_size: int = 64,
                   force_retrain: bool = False,
                   progress_cb=None) -> dict:
    """
    Train LSTM, DCRNN, GraphWaveNet on the same split as STGCN.
    Returns a dict {model_name: {MAE, RMSE, MAPE, horizon_metrics}}.

    STGCN results are passed in via data['stgcn_metrics'] (already computed
    from the pre-trained model checkpoint).

    Results are cached to CACHE_FILE to avoid re-training unnecessarily.
    """
    if not force_retrain and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                cached = json.load(f)
            # Verify cache is compatible
            if set(cached.keys()) >= {"STGCN", "LSTM", "DCRNN", "GraphWaveNet"}:
                return cached
        except Exception:
            pass

    from model.models import LSTMBaseline, DCRNNModel, GraphWaveNet

    scaler    = data["scaler"]
    adj_norm  = data["adj_norm"]
    X_train   = data["X_train"]
    y_train   = data["y_train"]
    X_val     = data["X_val"]
    y_val     = data["y_val"]
    X_test    = data["X_test"]
    y_test    = data["y_test"]
    num_nodes = data["num_nodes"]

    def _inv(arr):
        """Inverse transform (samples, horizon, nodes)."""
        s, h, n = arr.shape
        return scaler.inverse_transform(arr.reshape(-1, n)).reshape(s, h, n)

    results = {}

    # ── STGCN (pre-trained checkpoint) ──────────────────────────────────────
    stgcn_pred_inv = data["stgcn_pred_inv"]
    stgcn_true_inv = _inv(y_test)
    results["STGCN"] = {
        **compute_metrics(stgcn_true_inv, stgcn_pred_inv),
        "horizon_metrics": per_horizon_metrics(y_test, data["stgcn_pred_scaled"],
                                               scaler),
    }

    # ── Helper: train + eval one model ──────────────────────────────────────
    def _run(name, model_cls, kwargs):
        if progress_cb:
            progress_cb(f"Training {name}…", 0)
        model   = model_cls(num_nodes=num_nodes, **kwargs)
        trainer = ModelTrainer(model, adj_norm, device)
        trainer.fit(X_train, y_train, X_val, y_val,
                    epochs=epochs, batch_size=batch_size,
                    progress_cb=None)
        pred_scaled = trainer.predict(X_test, batch_size=batch_size)
        pred_inv    = _inv(pred_scaled)
        true_inv    = _inv(y_test)
        metrics     = compute_metrics(true_inv, pred_inv)
        metrics["horizon_metrics"] = per_horizon_metrics(y_test, pred_scaled, scaler)
        if progress_cb:
            progress_cb(f"{name} done", 1)
        return metrics

    # ── LSTM ─────────────────────────────────────────────────────────────────
    results["LSTM"] = _run("LSTM", LSTMBaseline, {
        "input_len": 12, "output_len": 3,
        "hidden_size": 128, "num_layers": 2, "dropout": 0.2,
    })

    # ── DCRNN ────────────────────────────────────────────────────────────────
    results["DCRNN"] = _run("DCRNN", DCRNNModel, {
        "input_len": 12, "output_len": 3,
        "hidden_ch": 64, "K": 2, "dropout": 0.2,
    })

    # ── Graph WaveNet ─────────────────────────────────────────────────────────
    results["GraphWaveNet"] = _run("GraphWaveNet", GraphWaveNet, {
        "input_len": 12, "output_len": 3,
        "residual_ch": 32, "dilation_ch": 32, "skip_ch": 128,
        "num_layers": 4, "dropout": 0.2,
    })

    # Cache to disk
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass

    return results

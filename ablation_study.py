"""
ablation_study.py
=================
Run from the stgcn-traffic-forecasting/ directory:

    python ablation_study.py [--epochs 40] [--output results/ablation.json]

Compares STGCN and baselines trained:
  - WITH    weather + event + time features  (full multimodal)
  - WITHOUT those features                   (speed only)

Results saved to JSON and printed as a table.
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_utils import (
    load_graph, load_speed_dataframe, normalize_adjacency,
    make_sequences, inv3d,
    build_time_features, build_weather_features_for_index,
    build_event_features_for_index,
    INPUT_LEN, OUTPUT_LEN, TRAIN_RATIO, VAL_RATIO,
)
from model.models   import BetterSTGCN, LSTMBaseline
from model.trainer  import ModelTrainer, compute_metrics
from sklearn.preprocessing import StandardScaler


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",      default="METR-LA.h5")
    p.add_argument("--pkl",     default="adj_METR-LA.pkl")
    p.add_argument("--epochs",  type=int, default=40)
    p.add_argument("--batch",   type=int, default=64)
    p.add_argument("--output",  default="ablation_results.json")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_base_data(h5_path, pkl_path):
    import h5py, pickle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(pkl_path, "rb") as f:
            sensor_ids, _, adj_mx = pickle.load(f, encoding="latin1")

    sensor_ids_str = list(map(str, sensor_ids))
    adj_mx = np.array(adj_mx, np.float32)

    with h5py.File(h5_path, "r") as f:
        cols   = [c.decode("utf-8") if isinstance(c, bytes) else str(c)
                  for c in f["df/axis0"][:]]
        ts_ns  = f["df/axis1"][:]
        values = f["df/block0_values"][:]

    import pandas as pd
    idx = pd.to_datetime(ts_ns)
    df  = pd.DataFrame(values, index=idx, columns=cols)
    df  = df[[c for c in sensor_ids_str if c in df.columns]]

    return df, adj_mx, sensor_ids_str


def split_and_scale(df):
    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
    scaler    = StandardScaler()
    tr_s      = scaler.fit_transform(df.values[:train_end]).astype(np.float32)
    va_s      = scaler.transform(df.values[train_end:val_end]).astype(np.float32)
    te_s      = scaler.transform(df.values[val_end:]).astype(np.float32)
    return tr_s, va_s, te_s, scaler, df.iloc[val_end:]


def build_multimodal_sequences(df, tr_s, va_s, te_s, include_features=True):
    """
    If include_features=True, append time/weather/event features to speed.
    The extra features are broadcast across all nodes for each timestep,
    increasing the channel dimension from N to N + num_extra_features.
    
    For fair comparison the model input_size must match — here we keep the
    same architecture and simply concatenate extra features as additional
    pseudo-nodes. This is a common ablation approach.
    """
    if not include_features:
        X_tr, y_tr = make_sequences(tr_s, INPUT_LEN, OUTPUT_LEN)
        X_va, y_va = make_sequences(va_s, INPUT_LEN, OUTPUT_LEN)
        X_te, y_te = make_sequences(te_s, INPUT_LEN, OUTPUT_LEN)
        return X_tr, y_tr, X_va, y_va, X_te, y_te

    # Build extra feature columns (T × F)
    tf = build_time_features(df.index)
    wf = build_weather_features_for_index(df.index)
    ef = build_event_features_for_index(df.index)

    extra = np.concatenate([
        tf.values,    # 6 cols
        wf.values,    # 3 cols
        ef.values,    # 1 col
    ], axis=1).astype(np.float32)           # (T, 10)

    # Scale extra features
    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    extra_scaler = StandardScaler()
    ex_tr = extra_scaler.fit_transform(extra[:train_end])
    ex_va = extra_scaler.transform(extra[train_end:val_end])
    ex_te = extra_scaler.transform(extra[val_end:])

    # Concatenate to speed along node axis (N → N + 10)
    tr_full = np.concatenate([tr_s, ex_tr.astype(np.float32)], axis=1)
    va_full = np.concatenate([va_s, ex_va.astype(np.float32)], axis=1)
    te_full = np.concatenate([te_s, ex_te.astype(np.float32)], axis=1)

    X_tr, y_tr = make_sequences(tr_full, INPUT_LEN, OUTPUT_LEN)
    X_va, y_va = make_sequences(va_full, INPUT_LEN, OUTPUT_LEN)
    X_te, y_te = make_sequences(te_full, INPUT_LEN, OUTPUT_LEN)

    return X_tr, y_tr, X_va, y_va, X_te, y_te


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(model_cls, model_kwargs, adj_norm, device,
               X_tr, y_tr, X_va, y_va, X_te, y_te, scaler,
               epochs, batch_size, label):
    print(f"  Training {label}…", flush=True)

    # y uses only speed nodes (first N cols) for metric computation
    num_speed_nodes = scaler.mean_.shape[0]

    model   = model_cls(**model_kwargs)
    trainer = ModelTrainer(model, adj_norm, device,
                           lr=1e-3, weight_decay=1e-4)
    trainer.fit(X_tr, y_tr, X_va, y_va,
                epochs=epochs, batch_size=batch_size, patience=10)

    pred_sc = trainer.predict(X_te, batch_size=batch_size)
    # Only score over the speed nodes
    p_sc  = pred_sc[:, :, :num_speed_nodes]
    y_sc  = y_te[:, :, :num_speed_nodes]

    p_inv = inv3d(p_sc, scaler)
    y_inv = inv3d(y_sc, scaler)
    m     = compute_metrics(y_inv, p_inv)
    print(f"    MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MAPE={m['MAPE']:.2f}%")
    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading data…")
    df, adj_mx, sensor_ids_str = load_base_data(args.h5, args.pkl)
    tr_s, va_s, te_s, scaler, test_df = split_and_scale(df)

    adj_norm  = normalize_adjacency(adj_mx)
    num_nodes = adj_norm.shape[0]

    # Expand adjacency for multimodal (extra feature pseudo-nodes)
    num_extra = 10
    adj_mm    = np.zeros((num_nodes + num_extra, num_nodes + num_extra),
                         dtype=np.float32)
    adj_mm[:num_nodes, :num_nodes] = adj_norm

    configs = {
        "STGCN_speed_only": {
            "model_cls":    BetterSTGCN,
            "model_kwargs": dict(num_nodes=num_nodes, input_len=INPUT_LEN,
                                 output_len=OUTPUT_LEN, hidden_channels=64),
            "adj":          adj_norm,
            "with_features": False,
        },
        "STGCN_multimodal": {
            "model_cls":    BetterSTGCN,
            "model_kwargs": dict(num_nodes=num_nodes + num_extra,
                                 input_len=INPUT_LEN,
                                 output_len=OUTPUT_LEN, hidden_channels=64),
            "adj":          adj_mm,
            "with_features": True,
        },
        "LSTM_speed_only": {
            "model_cls":    LSTMBaseline,
            "model_kwargs": dict(num_nodes=num_nodes, input_len=INPUT_LEN,
                                 output_len=OUTPUT_LEN, hidden_size=128),
            "adj":          adj_norm,
            "with_features": False,
        },
        "LSTM_multimodal": {
            "model_cls":    LSTMBaseline,
            "model_kwargs": dict(num_nodes=num_nodes + num_extra,
                                 input_len=INPUT_LEN,
                                 output_len=OUTPUT_LEN, hidden_size=128),
            "adj":          adj_mm,
            "with_features": True,
        },
    }

    results = {}
    for label, cfg in configs.items():
        print(f"\n[{label}]")
        X_tr, y_tr, X_va, y_va, X_te, y_te = build_multimodal_sequences(
            df, tr_s, va_s, te_s,
            include_features=cfg["with_features"],
        )
        m = run_single(
            model_cls=cfg["model_cls"],
            model_kwargs=cfg["model_kwargs"],
            adj_norm=cfg["adj"],
            device=device,
            X_tr=X_tr, y_tr=y_tr,
            X_va=X_va, y_va=y_va,
            X_te=X_te, y_te=y_te,
            scaler=scaler,
            epochs=args.epochs,
            batch_size=args.batch,
            label=label,
        )
        results[label] = m

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'MAPE %':>8}")
    print("-" * 62)
    for label, m in results.items():
        mape = f"{m['MAPE']:.2f}" if not np.isnan(m["MAPE"]) else "N/A"
        print(f"{label:<25} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {mape:>8}")

    # ── Delta analysis ────────────────────────────────────────────────────────
    print("\n── Feature contribution (multimodal vs speed-only) ──")
    for arch in ["STGCN", "LSTM"]:
        k_so = f"{arch}_speed_only"
        k_mm = f"{arch}_multimodal"
        if k_so in results and k_mm in results:
            delta_mae  = results[k_so]["MAE"]  - results[k_mm]["MAE"]
            delta_rmse = results[k_so]["RMSE"] - results[k_mm]["RMSE"]
            print(f"  {arch}: ΔMAE={delta_mae:+.4f}  ΔRMSE={delta_rmse:+.4f}"
                  f"  ({'multimodal better' if delta_mae > 0 else 'no gain'})")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()

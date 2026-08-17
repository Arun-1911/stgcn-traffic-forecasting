# STGCN Traffic Forecasting

**A Spatio-Temporal Graph Convolutional Network for traffic speed forecasting on the METR-LA sensor network, benchmarked against LSTM, DCRNN, and Graph WaveNet baselines.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-STGCN-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-anomaly%20detection-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A pre-trained STGCN checkpoint forecasts 15-minute-ahead traffic speed across **207 real LA freeway sensors**, evaluated against three baselines trained from scratch inside the app on the same split. On the full test set the checkpoint measures **MAE 3.26 mph, RMSE 7.69, MAPE 7.17%** — close to the architecture's published benchmark on MAE/MAPE, with a heavier RMSE tail than the original paper, reported here rather than smoothed over.

---

## Results

STGCN test-set performance, measured directly from the included checkpoint (full test set, 15-min / 3-step horizon):

| Metric | Value |
|---|---|
| MAE | **3.26 mph** |
| RMSE | **7.69** |
| MAPE | **7.17%** |

<details>
<summary><b>Comparison against published benchmark values</b></summary>

<br>

On METR-LA (5-min intervals, 15-min horizon), values reported in the original papers:

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| LSTM (baseline) | ~4.1–5.2 | ~7.5–9.0 | ~9–12% |
| DCRNN | ~3.60 | ~6.99 | ~7.59% |
| Graph WaveNet | ~3.19 | ~6.37 | ~7.45% |
| STGCN | ~3.15 | ~6.45 | ~7.01% |

The measured checkpoint's MAE and MAPE land inside this published range; RMSE is higher, meaning it has a somewhat heavier tail of large errors than the original STGCN paper's reported result. LSTM, DCRNN, and Graph WaveNet in this repo are trained from scratch inside the app's **Model Comparison** tab on request — their numbers will vary run to run depending on epochs and initialization, so they're not included in the table above as fixed figures.

</details>

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The pre-trained STGCN checkpoint and adjacency matrix are included in the repo, so inference works immediately. The METR-LA speed dataset (57 MB) is not committed — see [Dataset](#dataset) below for where to get it.

Optional, for live map routing and live weather:

```bash
export GOOGLE_MAPS_API_KEY="your_key_here"
export OPENWEATHER_API_KEY="your_key_here"
```

---

## Dashboard

| Tab | What it shows |
|---|---|
| **Overview** | Sensor/timestep counts, STGCN test-set metrics, predicted congestion heatmap |
| **Forecast** | Per-sensor forecast vs. actual, recursive multi-step prediction, one-step comparison over time |
| **Model Comparison** | STGCN vs LSTM vs DCRNN vs Graph WaveNet, trained on the same split — bar, radar, per-horizon charts |
| **Live Map** | Google Maps routing + a graph-shortest-path route forecast using real STGCN-predicted speeds |
| **Incidents** | Graph-propagated Gaussian-decay incident impact, before/after view |
| **Multimodal Features** | Weather, calendar events, and time-of-day features aligned to the dataset |
| **Model Insights** | SHAP feature attribution (which sensors/time-steps drove a forecast) and MC-Dropout confidence intervals |
| **Anomaly Detection** | Rolling z-score + Isolation Forest anomaly flags, per-sensor and network-wide |

---

## How it works

### Model architectures

**STGCN** (pre-trained checkpoint, primary model)
- 2 ST-Conv blocks: `TemporalConv → GraphConv → TemporalConv` + residual
- Input projection `Conv2d(1, 64, 1)`, readout FC `input_len × hidden → output_len`

**LSTM** — 2-layer, hidden=128, treats every sensor independently (no graph structure) — the no-spatial-information baseline.

**DCRNN** — bidirectional diffusion convolution (K=2) inside a GRU encoder-decoder, captures directed flow on the road graph.

**Graph WaveNet** — learnable adaptive adjacency (node embeddings) + 4 dilated causal convolution layers (dilations 1, 2, 1, 2).

### Dataset

[METR-LA](https://arxiv.org/abs/1707.01926) (Li et al., 2018) — 207 loop-detector sensors on Los Angeles freeways, 5-minute intervals, 2012-03-01 to 2012-06-28. Split 70% train / 15% val / 15% test, chronological (no shuffling, so the split respects time order).

### Multimodal features

| Category | Features | Alignment |
|---|---|---|
| Weather | temperature, humidity, rainfall | Live OpenWeatherMap for the current window; climatological pattern for the 2012 historical period |
| Events | event_severity (0–1) | Hand-curated calendar of known 2012 LA events + live incidents |
| Time | hour/day-of-week sin-cos, is_peak, is_weekend | Computed directly from the timestamp index |

### Incident propagation

Speed reduction for sensor node `n`, from an incident of given severity and radius:

```
impact[n] = severity_reduction × exp(-0.5 × (dist_km / radius_km)²)
propagated = impact + Σ_{k=1}^{2} 0.3^k × (adj_norm^k @ impact)
```

`severity_reduction`: low = 10%, medium = 25%, high = 45%. The second line propagates the impact 1–2 hops further through the sensor graph at decaying weight, so a blocked road also degrades nearby connected sensors, not just the ones inside the incident's radius.

### Explainability

**SHAP** (`utils/xai_shap.py`) — `KernelExplainer` treats the STGCN forward pass as a black-box function and attributes each forecast back to the input sensors and lookback time steps that drove it. Gated behind a button in the UI since KernelExplainer's cost scales with samples × feature count (2,484 input features here).

**MC-Dropout** (`utils/uncertainty.py`) — runs the model with dropout active across 30 stochastic forward passes to produce a 95% confidence interval around each forecast, rather than a single point estimate.

**Anomaly detection** (`utils/anomaly_detection.py`) — flags a timestep if either a rolling z-score threshold or an Isolation Forest (trained on speed + rolling stats + first/second differences) fires.

---

## What's real vs. approximated

| | Status |
|---|---|
| Sensor speed readings, timestamps | Real — METR-LA loop-detector data |
| Adjacency / road-network graph | Real — standard METR-LA distance-based adjacency matrix |
| STGCN, LSTM, DCRNN, Graph WaveNet predictions | Real — actual trained model output, not mocked |
| STGCN test-set metrics reported above | Real — measured directly from the included checkpoint |
| SHAP attributions, MC-Dropout intervals, anomaly flags | Real — computed from actual model runs, not illustrative |
| Sensor lat/lng used for incident distance & route matching | **Approximated** — linearly interpolated across METR-LA's bounding box; the pipeline doesn't include each sensor's true GPS coordinate |
| Incident speed-reduction model | **Heuristic** — a Gaussian-decay formula, not learned from real incident-response data |
| Historical (2012) weather | **Approximated** — climatological pattern, since no retroactive weather API exists for that period; only the *current* window can use live OpenWeatherMap data |
| Event calendar | **Partial** — a small hand-curated list of known 2012 LA events, not a comprehensive events database |
| "Current" conditions for route congestion | **Proxy** — uses the most recent test-window forecast, since there's no live sensor feed behind this dataset |

---

## Project structure

```
app.py                  Streamlit dashboard (all 8 tabs)
model/
  models.py             BetterSTGCN, LSTMBaseline, DCRNNModel, GraphWaveNet
  trainer.py             training loop, run_comparison(), compute_metrics()
utils/
  data_utils.py          data loading, preprocessing, multimodal features
  incident_utils.py      incident management, graph propagation, map HTML
  xai_shap.py             SHAP explainability
  uncertainty.py          MC-Dropout confidence intervals
  anomaly_detection.py    rolling z-score + Isolation Forest
  route_congestion.py     graph shortest-path + congestion scoring
ui/
  charts.py               Plotly chart helpers
ablation_study.py        standalone script: multimodal feature ablation
requirements.txt
```

## Dataset

`METR-LA.h5` (57 MB) isn't committed. Download it from the official data-preparation sections of either paper's repo and place it in the project root as `METR-LA.h5`:

- [liyaguang/DCRNN — Data Preparation](https://github.com/liyaguang/DCRNN#data-preparation)
- [nnzhan/Graph-WaveNet — Data Preparation](https://github.com/nnzhan/Graph-WaveNet#data-preparation)

`stgcn_tuned_real_graph_model.pt` (pre-trained checkpoint) and `adj_METR-LA.pkl` (adjacency matrix) are already in the repo.

## Citation

```
Yu, B., Yin, H., & Zhu, Z. (2018). Spatio-temporal graph convolutional networks:
A deep learning framework for traffic forecasting. IJCAI 2018.

Li, Y., et al. (2018). Diffusion Convolutional Recurrent Neural Network:
Data-Driven Traffic Forecasting. ICLR 2018.

Wu, Z., et al. (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling. IJCAI 2019.
```

## License

MIT

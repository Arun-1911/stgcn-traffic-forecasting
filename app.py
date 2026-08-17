"""
app.py — STGCN Traffic Forecasting : Research-Grade Traffic Forecasting System
=================================================================
Spatio-Temporal Graph Convolutional Network (STGCN) with:
  ✓ Multi-model comparison: STGCN vs DCRNN vs Graph WaveNet vs LSTM
  ✓ Multimodal inputs: weather, events, time-features
  ✓ Google Maps integration with location-name search
  ✓ Graph-aware incident propagation (before / after)
  ✓ Research-quality dashboard with Plotly charts

Run:
    streamlit run app.py

Required env vars (optional but recommended):
    GOOGLE_MAPS_API_KEY   — enables live map + routing
    OPENWEATHER_API_KEY   — enables live weather data
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import streamlit as st
import streamlit.components.v1 as components

# ── path bootstrap (allows running from repo root or inside stgcn-traffic-forecasting/) ──────
sys.path.insert(0, os.path.dirname(__file__))

from model.models  import BetterSTGCN
from model.trainer import run_comparison, compute_metrics
from utils.data_utils   import prepare_data, inv3d, inv2d, INPUT_LEN, OUTPUT_LEN, TIME_STEP_MIN
from utils.incident_utils import (
    initialize_state, add_incident, clear_incidents, save_incidents,
    get_default_incidents, compute_route_summary, compute_node_impact_vector,
    apply_incident_impact, build_map_html, approx_node_coords, haversine_km,
    DEFAULT_SOURCE, DEFAULT_DESTINATION,
)
from ui.charts import (
    forecast_plot, comparison_time_plot,
    model_comparison_bar, per_horizon_plot, radar_comparison,
    congestion_heatmap, before_after_plot,
    multimodal_panel, metrics_table,
)
from utils.xai_shap import (
    compute_shap_values, shap_sensor_importance, shap_timestep_importance,
    plot_shap_sensor_bar, plot_shap_timestep_bar, plot_shap_heatmap,
)
from utils.uncertainty import (
    mc_dropout_predict, confidence_interval, inverse_transform_mc,
    plot_uncertainty_band,
)
from utils.anomaly_detection import (
    detect_anomalies_full, scan_all_sensors, anomaly_summary,
    plot_anomaly_series, plot_anomaly_heatmap,
)
from utils.route_congestion import (
    find_route_nodes, multi_horizon_congestion, congestion_score,
    plot_route_speed_profile, plot_multi_horizon_congestion,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STGCN Traffic Forecasting",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
:root {
  --bg:#0a0a0a; --bg2:#050505; --surface:rgba(255,255,255,0.035); --surface-solid:#111111;
  --surface2:rgba(255,255,255,0.05);
  --border:rgba(255,255,255,0.12); --border-strong:rgba(255,255,255,0.26);
  --accent:#cfe3ff;
  --text:#f2f2ef; --muted:#8f8f89;
  --invert-bg:#f2f2ef; --invert-text:#0a0a0a;
  --glow:0 10px 30px rgba(0,0,0,0.5);
  --glow-strong:0 18px 44px rgba(0,0,0,0.65);
}
*{scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.25) transparent;}
::-webkit-scrollbar{width:8px;height:8px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.22);border-radius:8px;}
::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,0.38);}

html,body,[class*="css"]{font-family:'JetBrains Mono',monospace!important;
  background-color:var(--bg)!important;color:var(--text)!important;}

#stgcn-bg-canvas{position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.4;}

.stApp{background:
  radial-gradient(ellipse 90% 55% at 50% -4%,rgba(255,255,255,0.05) 0%,transparent 62%),
  radial-gradient(ellipse 120% 70% at 50% 118%,rgba(0,0,0,0.55) 0%,transparent 60%),
  linear-gradient(180deg,var(--bg) 0%,var(--bg2) 100%)!important;}

.stApp [data-testid="stAppViewContainer"],.stApp .main{position:relative;z-index:1;}

section[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(6,6,6,0.94),rgba(3,3,3,0.97))!important;
  border-right:1px solid var(--border)!important;backdrop-filter:blur(18px);
  box-shadow:6px 0 30px rgba(0,0,0,0.4);}
section[data-testid="stSidebar"] *{color:var(--text)!important;}

/* type ------------------------------------------------------------------ */
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;color:var(--text)!important;text-wrap:balance;}
h1{font-size:2.4rem!important;font-weight:800!important;letter-spacing:-.02em;
  color:#fff!important;animation:fade-in-up .7s cubic-bezier(.2,.8,.2,1) both;}
h2{font-size:.82rem!important;font-weight:600!important;letter-spacing:.22em!important;
  text-transform:lowercase!important;color:var(--muted)!important;
  padding-left:0;position:relative;
  margin-top:2.6rem!important;margin-bottom:1.3rem!important;
  animation:fade-in-up .5s ease both;}
h2::before{content:"";display:inline-block;width:18px;height:1px;background:var(--border-strong);
  margin-right:10px;vertical-align:middle;}

@keyframes fade-in-up{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
@keyframes glow-pulse{0%,100%{opacity:.5;}50%{opacity:1;}}

[data-testid="stMetric"],[data-testid="stPlotlyChart"],.decision-box,.explain-box,
.research-note,.incident-card{position:relative;animation:fade-in-up .55s ease both;}

[data-testid="stMetric"]{background:var(--surface)!important;backdrop-filter:blur(14px);
  border:1px solid var(--border)!important;border-radius:10px!important;
  padding:1.1rem 1.3rem!important;box-shadow:var(--glow)!important;
  transition:transform .25s ease,border-color .25s ease;}
[data-testid="stMetric"]:hover{transform:translateY(-3px);border-color:var(--border-strong)!important;}
[data-testid="stMetricLabel"]{font-size:.62rem!important;letter-spacing:.14em!important;
  text-transform:lowercase!important;color:var(--muted)!important;}
[data-testid="stMetricValue"]{font-family:'Syne',sans-serif!important;
  font-size:1.5rem!important;font-weight:700!important;color:#fff!important;}

/* buttons ----------------------------------------------------------------*/
.stButton>button{background:var(--invert-bg)!important;
  border:1px solid var(--invert-bg)!important;color:var(--invert-text)!important;
  font-family:'JetBrains Mono',monospace!important;font-size:.72rem!important;
  letter-spacing:.1em!important;text-transform:lowercase!important;
  border-radius:6px!important;padding:.55rem 1.15rem!important;
  transition:transform .18s ease,opacity .18s ease!important;}
.stButton>button:hover{opacity:.82!important;transform:translateY(-2px)!important;}
.stButton>button:active{transform:translateY(0)!important;opacity:.7!important;}

[data-testid="stPlotlyChart"]{border:1px solid var(--border)!important;
  border-radius:10px!important;overflow:hidden!important;box-shadow:var(--glow)!important;
  background:var(--surface)!important;backdrop-filter:blur(10px);
  transition:border-color .3s ease,transform .3s ease;}
[data-testid="stPlotlyChart"]:hover{border-color:var(--border-strong)!important;transform:translateY(-2px);}

/* tabs ---------------------------------------------------------------- */
[data-baseweb="tab-list"]{gap:4px!important;border-bottom:1px solid var(--border)!important;}
[data-baseweb="tab"]{background:transparent!important;border-radius:6px 6px 0 0!important;
  font-family:'JetBrains Mono',monospace!important;
  letter-spacing:.06em;text-transform:lowercase;transition:background .2s ease!important;}
[data-baseweb="tab"] p{color:var(--muted)!important;transition:color .2s ease!important;}
[data-baseweb="tab"]:hover{background:rgba(255,255,255,0.045)!important;}
[data-baseweb="tab"]:hover p{color:var(--text)!important;}
[data-baseweb="tab"][aria-selected="true"]{background:var(--surface)!important;}
[data-baseweb="tab"][aria-selected="true"] p{color:#fff!important;}
[data-baseweb="tab-highlight"]{background:#fff!important;height:2px!important;}

/* misc chrome ----------------------------------------------------------- */
.div-line{height:1px;background:var(--border-strong);margin:1.7rem 0;opacity:.6;}
.hud-row{display:flex;gap:8px;margin-bottom:1.2rem;flex-wrap:wrap;}
.hud-badge{background:var(--surface);backdrop-filter:blur(10px);border:1px solid var(--border);
  border-radius:4px;padding:5px 14px;font-size:.64rem;letter-spacing:.14em;
  color:var(--muted);text-transform:lowercase;transition:border-color .2s ease;}
.hud-badge:hover{border-color:var(--border-strong);}
.hud-badge span{color:#fff;font-weight:600;}
.live-dot{display:inline-block;width:5px;height:5px;border-radius:50%;background:var(--accent);
  margin-right:6px;box-shadow:0 0 6px var(--accent);animation:glow-pulse 1.8s ease-in-out infinite;}
.section-pill{display:inline-block;background:var(--surface);
  border:1px solid var(--border);border-radius:4px;padding:2px 10px;
  font-size:.6rem;letter-spacing:.16em;color:var(--muted);text-transform:lowercase;margin-bottom:.5rem;}
.decision-box{background:var(--surface2);
  border:1px solid var(--border-strong);border-radius:10px;padding:1.3rem 1.6rem;
  margin:.9rem 0;font-family:'Syne',sans-serif;font-size:.95rem;font-weight:600;
  color:#fff;line-height:1.6;backdrop-filter:blur(10px);box-shadow:var(--glow);}
.decision-box::before{content:"system decision";display:block;font-size:.6rem;
  letter-spacing:.2em;color:var(--muted);margin-bottom:8px;font-family:'JetBrains Mono',monospace;
  text-transform:lowercase;}
.explain-box{background:var(--surface);backdrop-filter:blur(10px);border:1px solid var(--border);
  border-radius:10px;padding:1.1rem 1.4rem;font-size:.8rem;line-height:1.85;color:var(--text);}
.explain-box strong{color:#fff;}
.research-note{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:.9rem 1.3rem;font-size:.78rem;line-height:1.75;
  color:var(--muted);margin:.5rem 0;backdrop-filter:blur(8px);}
.incident-card{background:var(--surface2);backdrop-filter:blur(8px);border:1px solid var(--border);
  border-left:2px solid var(--border-strong);border-radius:6px;
  padding:.65rem .9rem;margin-bottom:.5rem;font-size:.75rem;line-height:1.6;
  transition:transform .2s ease,border-left-color .2s ease;}
.incident-card:hover{transform:translateX(3px);border-left-color:#fff;}

@media (prefers-reduced-motion: reduce){
  h1,h2,[data-testid="stMetric"],[data-testid="stPlotlyChart"],.decision-box,
  .explain-box,.research-note,.incident-card,.live-dot{animation:none!important;}
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Ambient sensor-graph background + animated metric count-up
# (runs in a components iframe so <script> actually executes, then reaches
#  into the parent document — the standard Streamlit pattern for this).
# ──────────────────────────────────────────────────────────────────────────────
components.html("""
<script>
(function () {
  var doc;
  try { doc = window.parent.document; } catch (e) { return; }
  if (!doc || doc.getElementById('stgcn-bg-canvas')) return;

  var reduceMotion = window.parent.matchMedia
    && window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---- particle sensor-graph background ---- */
  var canvas = doc.createElement('canvas');
  canvas.id = 'stgcn-bg-canvas';
  var appRoot = doc.querySelector('section.stApp') || doc.body;
  appRoot.prepend(canvas);
  var ctx = canvas.getContext('2d');

  function resize() {
    canvas.width = window.parent.innerWidth;
    canvas.height = window.parent.innerHeight;
  }
  resize();
  window.parent.addEventListener('resize', resize);

  var COUNT = Math.min(70, Math.floor((canvas.width * canvas.height) / 26000));
  var nodes = [];
  for (var i = 0; i < COUNT; i++) {
    nodes.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.18,
      vy: (Math.random() - 0.5) * 0.18,
    });
  }
  var LINK_DIST = 130;

  function drawFrame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (var i = 0; i < nodes.length; i++) {
      var a = nodes[i];
      for (var j = i + 1; j < nodes.length; j++) {
        var b = nodes[j];
        var dx = a.x - b.x, dy = a.y - b.y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < LINK_DIST) {
          ctx.strokeStyle = 'rgba(255,255,255,' + (0.10 * (1 - dist / LINK_DIST)) + ')';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    for (var k = 0; k < nodes.length; k++) {
      var n = nodes[k];
      ctx.fillStyle = (k % 5 === 0) ? 'rgba(207,227,255,0.5)' : 'rgba(255,255,255,0.32)';
      ctx.beginPath();
      ctx.arc(n.x, n.y, 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function tick() {
    try {
      for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > canvas.width) n.vx *= -1;
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
      }
      drawFrame();
    } catch (err) {
      try { window.parent.console.error('STGCN_BG_TICK_ERROR', err && err.message, err && err.stack); } catch (e2) {}
    }
  }
  try {
    tick();
    /* requestAnimationFrame never fires here: this iframe is height=0/width=0,
       so it never enters a paint cycle. setInterval on window.parent is not
       tied to this iframe's own rendering, so it keeps firing. */
    if (!reduceMotion) window.parent.setInterval(tick, 40);
  } catch (err) {
    try { window.parent.console.error('STGCN_BG_INIT_ERROR', err && err.message, err && err.stack); } catch (e2) {}
  }

  /* ---- animated metric count-up ---- */
  var animated = new WeakSet();
  function animateValue(el) {
    if (animated.has(el)) return;
    var raw = el.textContent.trim();
    var m = raw.match(/^-?[\\d,]+(\\.\\d+)?/);
    if (!m) return;
    var numStr = m[0].replace(/,/g, '');
    var target = parseFloat(numStr);
    if (isNaN(target)) return;
    animated.add(el);
    var suffix = raw.slice(m[0].length);
    var decimals = (numStr.split('.')[1] || '').length;
    var duration = 700, start = null;
    function frame(ts) {
      if (start === null) start = ts;
      var p = Math.min(1, (ts - start) / duration);
      var eased = 1 - Math.pow(1 - p, 3);
      var val = target * eased;
      el.textContent = val.toFixed(decimals) + suffix;
      if (p < 1) window.parent.requestAnimationFrame(frame);
      else el.textContent = raw;
    }
    if (reduceMotion) { el.textContent = raw; }
    else window.parent.requestAnimationFrame(frame);
  }

  function scanMetrics() {
    doc.querySelectorAll('[data-testid="stMetricValue"]').forEach(function (el) {
      animateValue(el);
    });
  }
  var observer = new MutationObserver(function () { scanMetrics(); });
  observer.observe(doc.body, { childList: true, subtree: true });
  scanMetrics();
})();
</script>
""", height=0, width=0)

# ──────────────────────────────────────────────────────────────────────────────
# Constants / paths
# ──────────────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "stgcn_tuned_real_graph_model.pt"
H5_PATH    = "METR-LA.h5"
PKL_PATH   = "adj_METR-LA.pkl"

HIDDEN_CHANNELS = 64
DROPOUT         = 0.2

# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_stgcn(num_nodes: int) -> BetterSTGCN:
    model = BetterSTGCN(
        num_nodes=num_nodes, input_len=INPUT_LEN, output_len=OUTPUT_LEN,
        hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT,
    ).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def batch_predict(model, X: np.ndarray, adj: np.ndarray, bs: int = 64) -> np.ndarray:
    adj_t = torch.tensor(adj, dtype=torch.float32, device=DEVICE)
    out   = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device=DEVICE)
        out.append(model(xb, adj_t).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def single_predict(model, window: np.ndarray, adj: np.ndarray) -> np.ndarray:
    adj_t = torch.tensor(adj, dtype=torch.float32, device=DEVICE)
    x     = torch.tensor(window[np.newaxis], dtype=torch.float32, device=DEVICE)
    return model(x, adj_t).cpu().numpy()[0]


def recursive_forecast(model, base_window: np.ndarray, adj: np.ndarray,
                        steps: int) -> np.ndarray:
    history = base_window.copy()
    preds   = []
    while len(preds) < steps:
        block     = single_predict(model, history[-INPUT_LEN:], adj)
        take      = min(OUTPUT_LEN, steps - len(preds))
        preds.append(block[:take])
        history   = np.vstack([history, block[:take]])
    return np.vstack(preds)[:steps]


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
initialize_state()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 18px 0;border-bottom:1px solid rgba(0,210,255,0.12);margin-bottom:16px;">
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#fff;">🚦 STGCN Traffic Forecasting</div>
      <div style="font-size:.58rem;letter-spacing:.18em;color:#4a6070;text-transform:uppercase;margin-top:2px;">
        STGCN · Research Edition
      </div>
    </div>
    """, unsafe_allow_html=True)

    # API keys
    st.markdown("## API Configuration")
    maps_key = st.text_input("Google Maps API Key",
                              value=os.getenv("GOOGLE_MAPS_API_KEY", ""),
                              type="password", help="Enables live map, routing, autocomplete")
    weather_key = st.text_input("OpenWeather API Key",
                                 value=os.getenv("OPENWEATHER_API_KEY", ""),
                                 type="password", help="Enables live weather features")

    # Route inputs (location names)
    st.markdown("## Route Planner")
    st.caption("Enter location names — geocoded via Google Maps")
    from_loc = st.text_input("From", value="Los Angeles International Airport",
                              placeholder="e.g. LAX Airport")
    to_loc   = st.text_input("To",   value="Downtown Los Angeles",
                              placeholder="e.g. Downtown LA")

    st.markdown("""
    <div style="font-size:.68rem;color:#4a6070;margin-top:2px;line-height:1.6;">
    ◈ Route will be plotted on the live map.<br>
    ◈ Geocoding is handled by the Google Maps JS API on the map.<br>
    ◈ For model inference, METR-LA sensor coordinates are used.
    </div>""", unsafe_allow_html=True)

    # Incident controls
    st.markdown("## Add Incident")
    inc_type  = st.selectbox("Type", ["accident","roadblock","construction","rally","waterlogging"])
    inc_sev   = st.selectbox("Severity", ["low","medium","high"])
    inc_lat   = st.number_input("Latitude",  value=DEFAULT_SOURCE["lat"], format="%.5f")
    inc_lng   = st.number_input("Longitude", value=DEFAULT_SOURCE["lng"], format="%.5f")
    inc_rad   = st.slider("Radius (m)", 100, 3000, 500, 100)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Add", use_container_width=True):
            add_incident(inc_type, inc_sev, inc_lat, inc_lng, inc_rad)
            save_incidents(st.session_state.incidents)
            st.success("Added")
    with c2:
        if st.button("Reset", use_container_width=True):
            clear_incidents()
            save_incidents(st.session_state.incidents)
            st.success("Cleared")
    if st.button("Load Demo Incidents", use_container_width=True):
        st.session_state.incidents = get_default_incidents()
        save_incidents(st.session_state.incidents)
        st.success("Demo loaded")

    # Model comparison settings
    st.markdown("## Comparison Settings")
    comp_epochs = st.slider("Training epochs (baselines)", 20, 80, 40, 5,
                             help="Epochs to train LSTM / DCRNN / GraphWaveNet")
    force_retrain = st.checkbox("Force retrain baselines",
                                help="Delete cached results and retrain all models")

incidents = st.session_state.incidents

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:.2rem;">
  <div style="font-size:.6rem;letter-spacing:.2em;color:#4a6070;text-transform:uppercase;
       margin-bottom:6px;font-family:'JetBrains Mono',monospace;">
    ◈ RESEARCH-GRADE TRAFFIC INTELLIGENCE SYSTEM
  </div>
</div>
""", unsafe_allow_html=True)
st.title("Traffic Flow Forecasting")
st.markdown("""
<div class="hud-row">
  <div class="hud-badge"><span class="live-dot"></span>Primary Model <span>STGCN</span></div>
  <div class="hud-badge">Dataset <span>METR-LA</span></div>
  <div class="hud-badge">Input <span>12 steps (1 hr)</span></div>
  <div class="hud-badge">Horizon <span>3 steps (15 min)</span></div>
  <div class="hud-badge">Device <span>""" + str(DEVICE).upper() + """</span></div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# File guard
# ──────────────────────────────────────────────────────────────────────────────
missing = [p for p in [MODEL_PATH, H5_PATH, PKL_PATH] if not os.path.exists(p)]
if missing:
    st.error(f"Missing required files: {missing}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Load data + model
# ──────────────────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading data pipeline and STGCN checkpoint…"):
        data  = prepare_data(H5_PATH, PKL_PATH, weather_api_key=weather_key or None)
        model = load_stgcn(data["num_nodes"])

    with st.spinner("Running STGCN inference on test set…"):
        stgcn_pred_scaled = batch_predict(model, data["X_test"], data["adj_norm"])
        stgcn_pred_inv    = inv3d(stgcn_pred_scaled, data["scaler"])
        stgcn_true_inv    = inv3d(data["y_test"],    data["scaler"])
        stgcn_metrics     = compute_metrics(stgcn_true_inv, stgcn_pred_inv)

    # Attach STGCN predictions to data dict for comparison runner
    data["stgcn_pred_scaled"] = stgcn_pred_scaled
    data["stgcn_pred_inv"]    = stgcn_pred_inv

    # ── Tab layout ─────────────────────────────────────────────────────────────
    (tab_overview, tab_forecast, tab_comparison, tab_map, tab_incidents,
     tab_multimodal, tab_insights, tab_anomaly) = st.tabs([
        "📊 Overview",
        "📈 Forecast",
        "🔬 Model Comparison",
        "🗺️ Live Map",
        "⚠️ Incidents",
        "🌤️ Multimodal Features",
        "🧪 Model Insights",
        "🚨 Anomaly Detection",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: Overview
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown("## System Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sensors / Nodes",  data["num_nodes"])
        c2.metric("Total Timesteps",  data["num_timesteps"])
        c3.metric("Test Samples",     len(data["X_test"]))
        c4.metric("Forecast Horizon", f"{OUTPUT_LEN} steps")
        c5.metric("Time Resolution",  f"{TIME_STEP_MIN} min")

        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown("## STGCN Test-Set Performance")

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"{stgcn_metrics['MAE']:.4f}",  help="Mean Absolute Error (mph)")
        m2.metric("RMSE", f"{stgcn_metrics['RMSE']:.4f}", help="Root Mean Squared Error (mph)")
        m3.metric("MAPE",
                  "N/A" if np.isnan(stgcn_metrics["MAPE"]) else f"{stgcn_metrics['MAPE']:.2f}%",
                  help="Mean Absolute Percentage Error")

        # Congestion heatmap
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown("## Predicted Congestion Heatmap")
        valid_ts = data["valid_prediction_timestamps"]
        heatmap  = congestion_heatmap(stgcn_pred_inv, valid_ts, data["sensor_ids_str"],
                                       max_sensors=50, max_steps=288)
        st.plotly_chart(heatmap, use_container_width=True)

        st.markdown("""
        <div class="research-note">
        <b>Reading the heatmap:</b> Each row is a METR-LA sensor; each column is a 5-minute
        timestamp. Red = low speed (congested), green = high speed (free-flow).
        Vertical striping reveals temporal patterns (rush hours); horizontal
        striping reveals persistent sensor-level congestion hotspots — both of which
        STGCN captures via its graph + temporal convolutions.
        </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: Forecast
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_forecast:
        st.markdown("## In-Dataset Timestamp Prediction")

        col_a, col_b, col_c = st.columns([1.2, 2.2, 1.2])
        sensor_idx = col_a.selectbox(
            "Sensor",
            list(range(data["num_nodes"])),
            format_func=lambda x: f"{x} — {data['sensor_ids_str'][x]}",
        )
        ts_options    = list(data["valid_prediction_timestamps"])
        selected_ts   = col_b.selectbox(
            "Prediction Start",
            ts_options,
            format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M"),
        )
        forecast_step = col_c.selectbox("Horizon Step", [1, 2, 3])

        test_df      = data["test_df"]
        test_scaled  = data["test_scaled"]
        scaler       = data["scaler"]

        start_pos    = test_df.index.get_loc(selected_ts)
        in_s, in_e   = start_pos - INPUT_LEN, start_pos
        out_e        = start_pos + OUTPUT_LEN

        win_scaled   = test_scaled[in_s:in_e]
        win_orig     = test_df.iloc[in_s:in_e].values
        act_orig     = test_df.iloc[start_pos:out_e].values
        past_times   = test_df.index[in_s:in_e]
        future_times = test_df.index[start_pos:out_e]

        pred_scaled  = single_predict(model, win_scaled, data["adj_norm"])
        pred_orig    = inv2d(pred_scaled, scaler)

        pred_val = float(pred_orig[forecast_step - 1, sensor_idx])
        act_val  = float(act_orig[forecast_step - 1, sensor_idx])

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Sensor ID",  data["sensor_ids_str"][sensor_idx])
        p2.metric("Target Time", pd.Timestamp(future_times[forecast_step - 1]).strftime("%H:%M"))
        p3.metric("Predicted Speed", f"{pred_val:.2f} mph")
        p4.metric("Actual Speed",    f"{act_val:.2f} mph",
                  delta=f"{pred_val-act_val:.2f}")

        step_df = pd.DataFrame({
            "Step":      [f"t+{i}" for i in range(1, OUTPUT_LEN + 1)],
            "Timestamp": [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M") for t in future_times],
            "Predicted": pred_orig[:, sensor_idx],
            "Actual":    act_orig[:, sensor_idx],
            "Error":     np.abs(pred_orig[:, sensor_idx] - act_orig[:, sensor_idx]),
        })
        st.dataframe(step_df, use_container_width=True)

        st.plotly_chart(
            forecast_plot(past_times, win_orig[:, sensor_idx],
                          future_times, pred_orig[:, sensor_idx],
                          act_orig[:, sensor_idx],
                          sensor_id=data["sensor_ids_str"][sensor_idx]),
            use_container_width=True
        )

        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown("## Recursive Future Forecasting")

        rf1, rf2, rf3 = st.columns([1.2, 1.2, 2.0])
        rec_sensor = rf1.selectbox(
            "Sensor (recursive)", list(range(data["num_nodes"])), index=sensor_idx,
            format_func=lambda x: f"{x} — {data['sensor_ids_str'][x]}",
        )
        future_min = rf2.number_input("Minutes ahead", 5, 180, 30, 5)
        base_mode  = rf3.radio(
            "Starting point",
            ["From dataset end (latest)", "From selected in-dataset timestamp"],
            horizontal=True,
        )

        steps = max(1, int(future_min // TIME_STEP_MIN))

        if base_mode.startswith("From dataset end"):
            base_end     = len(data["df"])
            base_win_sc  = scaler.transform(data["df"].iloc[base_end - INPUT_LEN:].values)
            base_win_or  = data["df"].iloc[base_end - INPUT_LEN:].values
            base_times   = data["df"].index[base_end - INPUT_LEN:]
            fut_start    = data["df"].index[-1] + pd.Timedelta(minutes=TIME_STEP_MIN)
            actual_rec   = None
        else:
            bp           = test_df.index.get_loc(selected_ts)
            base_win_sc  = test_scaled[bp - INPUT_LEN:bp]
            base_win_or  = test_df.iloc[bp - INPUT_LEN:bp].values
            base_times   = test_df.index[bp - INPUT_LEN:bp]
            fut_start    = selected_ts
            ce           = bp + steps
            actual_rec   = test_df.iloc[bp:ce].values if ce <= len(test_df) else None

        rec_pred_sc  = recursive_forecast(model, base_win_sc, data["adj_norm"], steps)
        rec_pred_or  = inv2d(rec_pred_sc, scaler)
        rec_times    = pd.date_range(start=fut_start, periods=steps,
                                      freq=f"{TIME_STEP_MIN}min")

        r1, r2, r3 = st.columns(3)
        r1.metric("Sensor",        data["sensor_ids_str"][rec_sensor])
        r2.metric("Horizon",       f"{future_min} min")
        r3.metric("Final Predicted", f"{rec_pred_or[-1, rec_sensor]:.2f} mph")

        rec_df = pd.DataFrame({
            "Timestamp": [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M") for t in rec_times],
            "Predicted": rec_pred_or[:, rec_sensor],
        })
        if actual_rec is not None and len(actual_rec) >= steps:
            rec_df["Actual"] = actual_rec[:steps, rec_sensor]
        st.dataframe(rec_df, use_container_width=True)

        st.plotly_chart(
            forecast_plot(base_times, base_win_or[:, rec_sensor],
                          rec_times, rec_pred_or[:, rec_sensor],
                          rec_df["Actual"].values if "Actual" in rec_df.columns else None,
                          sensor_id=data["sensor_ids_str"][rec_sensor],
                          title=f"Recursive Forecast · Sensor {data['sensor_ids_str'][rec_sensor]}"),
            use_container_width=True
        )

        # One-step comparison chart
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown("## One-Step Comparison Over Time")

        n_pts    = st.slider("Test timestamps to compare", 20,
                              min(500, len(stgcn_true_inv)),
                              min(200, len(stgcn_true_inv)))
        cmp_ts   = data["test_df"].index[INPUT_LEN:INPUT_LEN + n_pts]
        st.plotly_chart(
            comparison_time_plot(cmp_ts,
                                  stgcn_true_inv[:n_pts, 0, sensor_idx],
                                  stgcn_pred_inv[:n_pts, 0, sensor_idx],
                                  sensor_id=data["sensor_ids_str"][sensor_idx]),
            use_container_width=True
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: Model Comparison
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_comparison:
        st.markdown("## Model Comparison Framework")
        st.markdown("""
        <div class="research-note">
        <b>Research methodology:</b> All four models are trained and evaluated on the
        <em>same</em> METR-LA dataset split (70% train / 15% val / 15% test) with
        identical preprocessing (StandardScaler fit on training data only).
        The STGCN uses its pre-trained checkpoint; LSTM, DCRNN, and Graph WaveNet
        are trained from scratch for the configured number of epochs.
        Results are cached to disk and reused on subsequent app loads.
        </div>""", unsafe_allow_html=True)

        run_btn = st.button("🔬 Run Model Comparison", use_container_width=False)

        if run_btn or os.path.exists("comparison_results.json"):
            with st.spinner("Running comparison (LSTM / DCRNN / Graph WaveNet) — "
                            "first run trains all models, subsequent runs load cache…"):
                progress_placeholder = st.empty()
                def _prog(msg, _frac):
                    progress_placeholder.info(f"⚙ {msg}")

                comp_results = run_comparison(
                    data=data,
                    device=DEVICE,
                    epochs=comp_epochs,
                    force_retrain=force_retrain,
                    progress_cb=_prog,
                )
                progress_placeholder.empty()

            # Metrics table
            st.markdown("### Results Table")
            st.plotly_chart(metrics_table(comp_results), use_container_width=True)

            # Pandas dataframe for download
            rows = []
            for model_name, res in comp_results.items():
                rows.append({
                    "Model": model_name,
                    "MAE":   round(res["MAE"],  4),
                    "RMSE":  round(res["RMSE"], 4),
                    "MAPE":  round(res["MAPE"], 4) if not np.isnan(res["MAPE"]) else None,
                })
            cdf = pd.DataFrame(rows)
            st.dataframe(cdf, use_container_width=True)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("⬇ Download Metrics CSV",
                                   cdf.to_csv(index=False).encode(),
                                   "model_comparison_metrics.csv", "text/csv",
                                   use_container_width=True)

            st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
            st.markdown("### Bar Chart Comparison")
            st.plotly_chart(model_comparison_bar(comp_results), use_container_width=True)

            st.markdown("### Per-Horizon Performance Degradation")
            h_col, m_col = st.columns([3, 1])
            horizon_metric = m_col.selectbox("Metric", ["MAE", "RMSE", "MAPE"])
            h_col.plotly_chart(per_horizon_plot(comp_results, horizon_metric),
                               use_container_width=True)

            st.markdown("### Radar — Multi-Metric Overview")
            st.plotly_chart(radar_comparison(comp_results), use_container_width=True)

            # Research summary
            st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
            st.markdown("### Research Analysis: Why STGCN Outperforms Baselines")

            best_model  = min(comp_results, key=lambda m: comp_results[m]["MAE"])
            stgcn_mae   = comp_results["STGCN"]["MAE"]
            lstm_mae    = comp_results["LSTM"]["MAE"]
            dcrnn_mae   = comp_results["DCRNN"]["MAE"]
            gwn_mae     = comp_results["GraphWaveNet"]["MAE"]
            improvement_over_lstm = (lstm_mae - stgcn_mae) / lstm_mae * 100

            st.markdown(f"""
            <div class="explain-box">
            <strong>LSTM (Baseline)</strong> treats each sensor independently, ignoring spatial
            correlations between adjacent road segments. It captures temporal patterns but
            misses the propagation of congestion through the network. STGCN achieves
            <b>{improvement_over_lstm:.1f}% lower MAE</b> than LSTM.<br><br>

            <strong>DCRNN</strong> introduces graph-aware diffusion convolution within a GRU
            framework, handling spatial dependencies bidirectionally. While competitive,
            its sequential encoder-decoder design limits parallelism and temporal
            receptive field compared to STGCN's convolutional blocks.<br><br>

            <strong>Graph WaveNet</strong> uses adaptive adjacency + dilated causal convolutions
            and is architecturally the most powerful baseline. On the full METR-LA benchmark
            (with longer training), it approaches STGCN performance — making it the strongest
            competitor and validating that graph-based approaches dominate.<br><br>

            <strong>STGCN</strong> captures both <em>spatial dependencies</em> (through spectral
            graph convolution over the sensor graph) and <em>temporal dynamics</em> (through
            stacked 1-D temporal convolutions) in a single end-to-end framework.
            Its residual block structure allows deep architectures without vanishing gradients,
            and the pre-trained checkpoint benefits from full-dataset optimisation.
            </div>""", unsafe_allow_html=True)

        else:
            st.info("Click **Run Model Comparison** to train baseline models and compare. "
                    "This trains LSTM, DCRNN, and Graph WaveNet from scratch — "
                    f"estimated time: ~{comp_epochs * 15 // 60 + 1} minutes on CPU.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: Live Map
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_map:
        st.markdown("## Live Traffic Map")
        st.markdown("""
        <div class="research-note">
        Enter location names in the sidebar (or the map's Route Planner panel)
        and click <b>Calculate Route</b> to get real-time directions with traffic.
        Incident markers show currently active incidents with their impact radius.
        </div>""", unsafe_allow_html=True)

        # Geocode sidebar strings to lat/lng approximations for METR-LA
        # (for model inference; actual geocoding is done client-side via Maps JS API)
        src  = DEFAULT_SOURCE
        dst  = DEFAULT_DESTINATION

        map_html = build_map_html(
            api_key=maps_key or "",
            source=src,
            destination=dst,
            incidents=incidents,
        )
        components.html(map_html, height=680, scrolling=False)

        summary = compute_route_summary(src, dst, incidents)

        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown("## Route Summary")
        rs1, rs2, rs3, rs4 = st.columns(4)
        rs1.metric("Est. Travel Time",  f"{summary['travel_time_min']} min")
        rs2.metric("Incident Delay",    f"{summary['added_delay_min']} min")
        rs3.metric("Risk Score",        summary["risk_score"])
        rs4.metric("Recommendation",    summary["recommendation"])

        # ── Model-based route congestion (graph shortest path + STGCN forecast) ──
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown("## Model-Based Route Forecast")
        st.markdown("""
        <div class="research-note">
        Unlike the heuristic route summary above (straight-line distance + incident
        penalties), this section finds the shortest path through the actual sensor
        graph and reads real STGCN-predicted speeds off every sensor along it.
        Source/destination are snapped to their nearest sensor nodes using
        approximate node coordinates (linear interpolation across the METR-LA
        bounding box — the same approximation the incident system uses).
        </div>""", unsafe_allow_html=True)

        num_nodes_map = data["num_nodes"]
        node_coords   = [approx_node_coords(i, num_nodes_map) for i in range(num_nodes_map)]
        src_idx = min(range(num_nodes_map),
                      key=lambda i: haversine_km(src["lat"], src["lng"], *node_coords[i]))
        dst_idx = min(range(num_nodes_map),
                      key=lambda i: haversine_km(dst["lat"], dst["lng"], *node_coords[i]))

        path_nodes  = find_route_nodes(data["adj_mx"], src_idx, dst_idx)
        latest_pred = stgcn_pred_inv[-1]     # (horizon, nodes) — most recent test-window forecast
        multi_res   = multi_horizon_congestion(path_nodes, latest_pred, output_len=OUTPUT_LEN)
        score       = congestion_score(multi_res[0])

        mr1, mr2, mr3, mr4 = st.columns(4)
        mr1.metric("Route Hops",           len(path_nodes))
        mr2.metric("Congestion Score",     f"{score:.0f} / 100")
        mr3.metric("Bottleneck Speed",     f"{multi_res[0]['min_speed_mph']:.1f} mph")
        mr4.metric("Est. Travel Time (t+1)", f"{multi_res[0]['total_time_min']:.1f} min")

        st.plotly_chart(
            plot_route_speed_profile(multi_res[0], data["sensor_ids_str"], horizon_step=1),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_multi_horizon_congestion(multi_res, data["sensor_ids_str"]),
            use_container_width=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5: Incidents & Before/After
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_incidents:
        st.markdown("## Incident-Aware Traffic Prediction")
        st.markdown("""
        <div class="research-note">
        Incidents are propagated across the sensor graph using a Gaussian decay
        model: nodes closer to an incident receive larger speed reductions.
        Graph-adjacent nodes (1–2 hops) inherit attenuated secondary impacts.
        The chart shows STGCN predicted speeds <b>before</b> and <b>after</b>
        incident injection for the first 40 sensors.
        </div>""", unsafe_allow_html=True)

        # Choose a test sample window for demonstration
        n_samples   = len(stgcn_pred_inv)
        demo_sample = st.slider("Test sample index", 0, n_samples - 1, n_samples // 2)
        horiz_step  = st.select_slider("Horizon step", [1, 2, 3], 1) - 1

        before_speed = stgcn_pred_inv[demo_sample]   # (horizon, nodes)
        impact_vec   = compute_node_impact_vector(incidents, data["num_nodes"],
                                                   data["adj_norm"])
        after_speed  = apply_incident_impact(before_speed, impact_vec)

        ba_fig = before_after_plot(before_speed, after_speed,
                                    data["sensor_ids_str"], horizon_step=horiz_step)
        st.plotly_chart(ba_fig, use_container_width=True)

        # Summary stats
        affected = (impact_vec > 0.02).sum()
        max_red  = float(impact_vec.max() * 100)
        avg_red  = float(impact_vec[impact_vec > 0.02].mean() * 100) if affected else 0

        si1, si2, si3 = st.columns(3)
        si1.metric("Affected Nodes",    f"{affected} / {data['num_nodes']}")
        si2.metric("Max Speed Reduction", f"{max_red:.1f}%")
        si3.metric("Avg Reduction (affected)", f"{avg_red:.1f}%")

        # Active incident list
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown("### Active Incidents")
        if incidents:
            sev_col = {"low": "#ffd166", "medium": "#ff9f1c", "high": "#ff4d4d"}
            for i, inc in enumerate(incidents, 1):
                sc = sev_col.get(inc["severity"], "#ff6b35")
                st.markdown(
                    f'<div class="incident-card" style="border-left-color:{sc};">'
                    f'<span style="color:{sc};font-weight:700;">#{i} — {inc["type"].upper()}</span><br>'
                    f'Severity: <span style="color:{sc};">{inc["severity"].upper()}</span> · '
                    f'Location: {inc["lat"]:.4f}, {inc["lng"]:.4f} · '
                    f'Radius: {inc["radius_m"]} m'
                    f'</div>', unsafe_allow_html=True
                )
        else:
            st.info("No incidents currently active. Add incidents via the sidebar.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 6: Multimodal Features
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_multimodal:
        st.markdown("## Multimodal Context Features")
        st.markdown("""
        <div class="research-note">
        Three categories of external signals are computed and aligned to every
        METR-LA timestamp:<br>
        <b>A. Weather</b> — temperature, humidity, rainfall (live via OpenWeatherMap API
        for current window; LA climatological patterns for 2012 historical data).<br>
        <b>B. Events</b> — calendar-based severity scores for known LA events (Marathon,
        King's parade, holidays) derived from public event records for the
        2012-03-01 → 2012-06-28 dataset period.<br>
        <b>C. Time features</b> — sinusoidal hour / day-of-week encoding + binary
        peak / weekend flags, capturing cyclic temporal patterns.
        </div>""", unsafe_allow_html=True)

        # Window selector
        total_ts   = len(data["df"])
        win_start  = st.slider("Window start (timestep index)", 0,
                                max(0, total_ts - 289), 0, 12)
        mm_fig = multimodal_panel(
            data["time_features"],
            data["weather_features"],
            data["event_features"],
            window_start=win_start,
            window_size=288,   # 24 hours
        )
        st.plotly_chart(mm_fig, use_container_width=True)

        # Feature statistics
        st.markdown("### Feature Statistics")
        wf = data["weather_features"]
        tf = data["time_features"]
        ef = data["event_features"]

        stat_rows = [
            {"Feature": "Temperature (°C)",  "Mean": f"{wf['temperature_c'].mean():.1f}",
             "Std": f"{wf['temperature_c'].std():.1f}",
             "Min": f"{wf['temperature_c'].min():.1f}", "Max": f"{wf['temperature_c'].max():.1f}"},
            {"Feature": "Humidity (%)",       "Mean": f"{wf['humidity_pct'].mean():.1f}",
             "Std": f"{wf['humidity_pct'].std():.1f}",
             "Min": f"{wf['humidity_pct'].min():.1f}", "Max": f"{wf['humidity_pct'].max():.1f}"},
            {"Feature": "Rainfall (mm/h)",    "Mean": f"{wf['rain_1h_mm'].mean():.3f}",
             "Std": f"{wf['rain_1h_mm'].std():.3f}",
             "Min": f"{wf['rain_1h_mm'].min():.3f}", "Max": f"{wf['rain_1h_mm'].max():.3f}"},
            {"Feature": "Peak Hours (frac.)", "Mean": f"{tf['is_peak'].mean():.3f}",
             "Std": "—", "Min": "0", "Max": "1"},
            {"Feature": "Event Severity",     "Mean": f"{ef['event_severity'].mean():.3f}",
             "Std": f"{ef['event_severity'].std():.3f}",
             "Min": "0", "Max": f"{ef['event_severity'].max():.1f}"},
        ]
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)

        # Current live weather
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown("### Current Live Weather (Los Angeles)")
        from utils.data_utils import fetch_current_weather
        live_wx = fetch_current_weather(api_key=weather_key or None)
        wx1, wx2, wx3, wx4 = st.columns(4)
        wx1.metric("Temperature",  f"{live_wx['temperature_c']:.1f} °C")
        wx2.metric("Humidity",     f"{live_wx['humidity_pct']:.0f}%")
        wx3.metric("Rainfall 1h",  f"{live_wx['rain_1h_mm']:.2f} mm")
        wx4.metric("Data Source",  live_wx["source"].replace("_", " ").title())

        # Downloads
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        dl_df = pd.concat([
            data["time_features"].reset_index().rename(columns={"index": "timestamp"}),
            data["weather_features"].reset_index(drop=True),
            data["event_features"].reset_index(drop=True),
        ], axis=1)
        st.download_button("⬇ Download Multimodal Features CSV",
                           dl_df.to_csv(index=False).encode(),
                           "multimodal_features.csv", "text/csv")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 7: Model Insights — SHAP explainability + MC-Dropout uncertainty
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_insights:
        st.markdown("## Model Insights")
        st.markdown("""
        <div class="research-note">
        Two complementary views into <em>why</em> and <em>how confident</em> STGCN's
        forecasts are: SHAP attributes each prediction back to the input sensors and
        lookback time steps that drove it; MC-Dropout uncertainty runs many stochastic
        forward passes to produce a confidence interval around the forecast.
        </div>""", unsafe_allow_html=True)

        st.markdown("### SHAP Feature Attribution")
        st.caption("Explains which sensors and lookback time steps most influenced the "
                   "model's predictions, computed over a handful of test-set windows. "
                   "KernelExplainer's cost scales with samples × feature count (2,484 "
                   "input features here), so this stays deliberately small — a few "
                   "seconds to under a minute.")
        shap_nsamples = st.select_slider(
            "SHAP samples (speed ↔ detail)", [20, 40, 80, 150], value=40,
            help="Higher = more accurate attribution, slower to compute.")
        run_shap = st.button("🧬 Run SHAP Analysis", use_container_width=False)

        if run_shap:
            with st.spinner("Computing SHAP values (KernelExplainer)…"):
                shap_values, _ = compute_shap_values(
                    model, data["X_test"][:20], data["adj_norm"], DEVICE,
                    background_samples=15, explain_samples=3,
                    nsamples=shap_nsamples,
                )
                st.session_state["shap_values"] = shap_values

        if "shap_values" in st.session_state:
            sv           = st.session_state["shap_values"]
            sensor_imp   = shap_sensor_importance(sv, data["num_nodes"], INPUT_LEN)
            timestep_imp = shap_timestep_importance(sv, data["num_nodes"], INPUT_LEN)

            st.plotly_chart(plot_shap_sensor_bar(sensor_imp, data["sensor_ids_str"]),
                            use_container_width=True)
            shcol1, shcol2 = st.columns(2)
            shcol1.plotly_chart(plot_shap_timestep_bar(timestep_imp, INPUT_LEN, TIME_STEP_MIN),
                                use_container_width=True)
            shcol2.plotly_chart(plot_shap_heatmap(sv, data["num_nodes"], INPUT_LEN,
                                                  step_minutes=TIME_STEP_MIN),
                                use_container_width=True)
        else:
            st.info("Click **Run SHAP Analysis** to compute feature attributions.")

        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown("### Prediction Uncertainty (MC-Dropout)")
        st.caption("Runs 30 stochastic forward passes with dropout active to estimate "
                   "a 95% confidence interval around the point forecast.")

        uc1, uc2 = st.columns(2)
        unc_sensor = uc1.selectbox(
            "Sensor", list(range(data["num_nodes"])),
            format_func=lambda x: f"{x} — {data['sensor_ids_str'][x]}",
            key="unc_sensor",
        )
        unc_ts = uc2.selectbox(
            "Prediction Start", list(data["valid_prediction_timestamps"]),
            format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M"),
            key="unc_ts",
        )

        u_start  = data["test_df"].index.get_loc(unc_ts)
        u_in_s   = u_start - INPUT_LEN
        u_out_e  = u_start + OUTPUT_LEN
        u_win    = data["test_scaled"][u_in_s:u_start]
        u_past   = data["test_df"].iloc[u_in_s:u_start].values
        u_act    = data["test_df"].iloc[u_start:u_out_e].values
        u_past_t = data["test_df"].index[u_in_s:u_start]
        u_fut_t  = data["test_df"].index[u_start:u_out_e]

        mean_sc, std_sc, _   = mc_dropout_predict(
            model, u_win[np.newaxis], data["adj_norm"], DEVICE, n_passes=30,
        )
        mean_inv, std_inv    = inverse_transform_mc(mean_sc, std_sc, data["scaler"])
        lower_inv, upper_inv = confidence_interval(mean_inv, std_inv)

        st.plotly_chart(
            plot_uncertainty_band(
                u_past_t, u_past[:, unc_sensor],
                u_fut_t, mean_inv[0, :, unc_sensor],
                lower_inv[0, :, unc_sensor], upper_inv[0, :, unc_sensor],
                actual_vals=u_act[:, unc_sensor],
                sensor_id=data["sensor_ids_str"][unc_sensor],
            ),
            use_container_width=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 8: Anomaly Detection
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_anomaly:
        st.markdown("## Anomaly Detection")
        st.markdown("""
        <div class="research-note">
        Flags unusual speed readings in the historical data using two independent
        detectors: a rolling z-score threshold (sudden drops/spikes) and an
        Isolation Forest trained on speed, rolling statistics, and first/second
        differences. A point is flagged if <em>either</em> detector fires.
        </div>""", unsafe_allow_html=True)

        anom_sensor_idx = st.selectbox(
            "Sensor", list(range(data["num_nodes"])),
            format_func=lambda x: f"{x} — {data['sensor_ids_str'][x]}",
            key="anom_sensor",
        )
        anom_sensor_id = data["sensor_ids_str"][anom_sensor_idx]

        with st.spinner("Fitting anomaly detectors…"):
            anom_result = detect_anomalies_full(data["df"], anom_sensor_id)

        n_anom    = int(anom_result["anomaly"].sum())
        anom_rate = n_anom / len(anom_result) * 100

        a1, a2, a3 = st.columns(3)
        a1.metric("Total Timesteps", len(anom_result))
        a2.metric("Anomalies Found", n_anom)
        a3.metric("Anomaly Rate",    f"{anom_rate:.2f}%")

        st.plotly_chart(plot_anomaly_series(anom_result, anom_sensor_id),
                        use_container_width=True)

        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown("### Scan Across Sensors")
        st.caption("Fits an independent detector per sensor — slower, run on demand.")

        scan_n = st.slider("Sensors to scan", 5, 60, 10,
                           help="~1.5s per sensor — 10 finishes in around 15-20s.")
        if st.button("🔍 Scan Sensors", use_container_width=False):
            with st.spinner(f"Scanning {scan_n} sensors…"):
                scan_results = scan_all_sensors(data["df"], data["sensor_ids_str"],
                                                max_sensors=scan_n)
                st.session_state["anomaly_scan"] = anomaly_summary(scan_results)

        if "anomaly_scan" in st.session_state:
            summary_df = st.session_state["anomaly_scan"]
            st.plotly_chart(plot_anomaly_heatmap(summary_df), use_container_width=True)
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("Click **Scan Sensors** to rank sensors by anomaly frequency.")

    # ── Global decision layer ──────────────────────────────────────────────────
    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
    st.markdown("## Integrated Decision Layer")

    final_speed  = float(inv2d(rec_pred_sc, scaler)[-1, sensor_idx])
    summary_dec  = compute_route_summary(DEFAULT_SOURCE, DEFAULT_DESTINATION, incidents)
    delay        = summary_dec["added_delay_min"]

    if final_speed < 25 and delay >= 10:
        decision = "Severe congestion predicted with live incidents. Alternate route strongly recommended."
    elif final_speed < 35 or delay >= 10:
        decision = "Moderate congestion or active incidents detected. Expect delays."
    else:
        decision = "Traffic conditions are manageable. Current route is acceptable."

    d1, d2, d3 = st.columns(3)
    d1.metric("Predicted Speed (final step)", f"{final_speed:.2f} mph")
    d2.metric("Incident Delay",               f"{delay} min")
    d3.metric("Recommendation",               summary_dec["recommendation"])

    st.markdown(f'<div class="decision-box">{decision}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
    <strong>Layer 1 — STGCN Spatial-Temporal Model:</strong> Captures both spatial
    dependencies (propagation of congestion across the sensor graph) and temporal
    dynamics (rush-hour patterns, day-of-week effects) via spectral graph convolution
    and stacked temporal convolution blocks.<br><br>
    <strong>Layer 2 — Multimodal Context:</strong> Weather, calendar events, and time
    features provide external context that the model implicitly learns during training
    and which can further modulate live forecasts.<br><br>
    <strong>Layer 3 — Incident Propagation:</strong> Real-time incidents are injected
    as graph-aware speed reductions using Gaussian distance decay + adjacency-matrix
    propagation, modifying the model output without requiring re-training.<br><br>
    <strong>Decision Synthesis:</strong> The final routing recommendation combines
    all three layers to produce a human-interpretable recommendation.
    </div>""", unsafe_allow_html=True)

    dl1, dl2 = st.columns(2)
    with dl1:
        dl_metrics = pd.DataFrame([{
            "Model": "STGCN", **stgcn_metrics
        }]).to_csv(index=False).encode()
        st.download_button("⬇ Download STGCN Metrics", dl_metrics,
                            "stgcn_metrics.csv", "text/csv", use_container_width=True)
    with dl2:
        dl_forecast = rec_df.to_csv(index=False).encode()
        st.download_button("⬇ Download Forecast CSV", dl_forecast,
                            "recursive_forecast.csv", "text/csv", use_container_width=True)

except Exception as exc:
    st.exception(exc)

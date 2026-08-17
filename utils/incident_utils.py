"""
utils/incident_utils.py
=======================
Incident management with graph-aware propagation.

Features:
  - Add / remove incidents (accident, roadblock, construction, rally, waterlogging)
  - Compute Gaussian-decay speed penalty for every graph node based on
    spatial distance from incident and incident severity
  - Before vs After comparison on model predictions
  - Haversine distance for realistic spatial attenuation
"""

import json
import math
from pathlib import Path

import numpy as np
import streamlit as st

DATA_FILE           = Path("incidents.json")
DEFAULT_SOURCE      = {"lat": 34.0522, "lng": -118.2437}   # Downtown LA
DEFAULT_DESTINATION = {"lat": 34.0195, "lng": -118.4912}   # Santa Monica

# METR-LA sensor approximate bounding box (for node geocoding approximation)
METR_LA_BOUNDS = {
    "lat_min": 33.90, "lat_max": 34.15,
    "lng_min": -118.60, "lng_max": -118.10,
}

# Severity → speed reduction fraction
SEVERITY_REDUCTION = {"low": 0.10, "medium": 0.25, "high": 0.45}

# Penalty minutes for route ETA calculation
PENALTY_MAP = {
    ("accident",      "low"):  3,  ("accident",      "medium"): 8,   ("accident",      "high"): 20,
    ("rally",         "low"):  5,  ("rally",         "medium"): 12,  ("rally",         "high"): 30,
    ("roadblock",     "low"):  8,  ("roadblock",     "medium"): 18,  ("roadblock",     "high"): 40,
    ("construction",  "low"):  4,  ("construction",  "medium"): 10,  ("construction",  "high"): 20,
    ("waterlogging",  "low"):  6,  ("waterlogging",  "medium"): 14,  ("waterlogging",  "high"): 25,
}


# ──────────────────────────────────────────────────────────────────────────────
# State management
# ──────────────────────────────────────────────────────────────────────────────

def initialize_state():
    if "incidents" not in st.session_state:
        st.session_state.incidents = load_incidents()


def load_incidents() -> list:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_incidents(incidents: list):
    DATA_FILE.write_text(json.dumps(incidents, indent=2), encoding="utf-8")


def clear_incidents():
    st.session_state.incidents = []


def add_incident(inc_type: str, severity: str,
                 lat: float, lng: float, radius_m: float):
    st.session_state.incidents.append({
        "type":     inc_type,
        "severity": severity,
        "lat":      float(lat),
        "lng":      float(lng),
        "radius_m": int(radius_m),
    })


def get_default_incidents() -> list:
    return [
        {"type": "accident",     "severity": "high",   "lat": 34.052, "lng": -118.243, "radius_m": 400},
        {"type": "construction", "severity": "medium",  "lat": 34.035, "lng": -118.280, "radius_m": 700},
        {"type": "roadblock",    "severity": "low",     "lat": 34.065, "lng": -118.310, "radius_m": 300},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Spatial helpers
# ──────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in kilometres."""
    R   = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ  = math.radians(lat2 - lat1)
    dλ  = math.radians(lon2 - lon1)
    a   = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0, a)))


def point_to_segment_km(p: dict, a: dict, b: dict) -> float:
    """Minimum distance from point p to segment a–b (all lat/lng dicts)."""
    px, py = p["lat"], p["lng"]
    ax, ay = a["lat"], a["lng"]
    bx, by = b["lat"], b["lng"]
    abx, aby   = bx - ax, by - ay
    apx, apy   = px - ax, py - ay
    ab_sq      = abx ** 2 + aby ** 2
    if ab_sq == 0:
        return haversine_km(px, py, ax, ay)
    t    = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_sq))
    cx, cy = ax + t * abx, ay + t * aby
    return haversine_km(px, py, cx, cy)


def approx_node_coords(node_idx: int, num_nodes: int) -> tuple[float, float]:
    """
    Approximate lat/lng for a METR-LA sensor node by linearly interpolating
    its index across the LA bounding box.
    This is used only for incident propagation distance calculation —
    actual sensor positions are not stored in the public METR-LA pickle.
    """
    frac = node_idx / max(num_nodes - 1, 1)
    lat  = METR_LA_BOUNDS["lat_min"] + frac * (METR_LA_BOUNDS["lat_max"] - METR_LA_BOUNDS["lat_min"])
    lng  = METR_LA_BOUNDS["lng_min"] + frac * (METR_LA_BOUNDS["lng_max"] - METR_LA_BOUNDS["lng_min"])
    return lat, lng


# ──────────────────────────────────────────────────────────────────────────────
# Graph-aware propagation
# ──────────────────────────────────────────────────────────────────────────────

def compute_node_impact_vector(incidents: list, num_nodes: int,
                                adj_norm: np.ndarray,
                                propagation_hops: int = 2) -> np.ndarray:
    """
    For each graph node, compute a speed-reduction factor in [0, 1].
    Factor = 0   → no impact (multiply predicted speed by 1.0)
    Factor = 0.4 → 40% speed reduction (multiply by 0.6)

    Algorithm:
      1. For each incident, compute Gaussian distance decay to every node.
      2. Combine primary impact with graph-propagated secondary impact
         (neighbouring nodes are also affected with diminishing weight).
      3. Return element-wise maximum across all incidents.
    """
    impact = np.zeros(num_nodes, dtype=np.float32)

    for inc in incidents:
        base_reduction = SEVERITY_REDUCTION.get(inc["severity"], 0.2)
        radius_km      = inc["radius_m"] / 1000.0

        # Primary impact: Gaussian decay by distance.
        # sigma = max(radius_km, 2.0) so that the coarse linear node-coordinate
        # approximation (nodes spaced ~12 km apart across the LA bounding box)
        # still produces visible, realistic impact on nearby nodes.
        sigma   = max(radius_km, 2.0)
        primary = np.zeros(num_nodes, dtype=np.float32)
        for n in range(num_nodes):
            lat, lng = approx_node_coords(n, num_nodes)
            dist_km  = haversine_km(inc["lat"], inc["lng"], lat, lng)
            decay       = math.exp(-0.5 * (dist_km / sigma) ** 2)
            primary[n]  = base_reduction * decay

        # Secondary propagation via adjacency (up to `propagation_hops` hops)
        propagated = primary.copy()
        for hop in range(propagation_hops):
            propagated = propagated + 0.3 ** (hop + 1) * (adj_norm @ propagated)

        impact = np.maximum(impact, np.clip(propagated, 0.0, base_reduction))

    return impact


def apply_incident_impact(predictions: np.ndarray,
                           impact_vector: np.ndarray) -> np.ndarray:
    """
    Apply speed reduction to model predictions.
    predictions : (horizon, nodes) — inverse-transformed speed (mph)
    impact_vector: (nodes,) in [0, 1]
    Returns modified predictions.
    """
    reduction = 1.0 - impact_vector                        # (nodes,)
    return predictions * reduction[np.newaxis, :]          # broadcast over horizon


# ──────────────────────────────────────────────────────────────────────────────
# Route summary
# ──────────────────────────────────────────────────────────────────────────────

def compute_route_summary(source: dict, destination: dict,
                           incidents: list) -> dict:
    """Estimate travel time and incident delay for a source→dest route."""
    dist_km      = haversine_km(source["lat"], source["lng"],
                                destination["lat"], destination["lng"])
    base_min     = max(8, round(dist_km * 3.2 + 6))
    added_delay  = 0
    risk_score   = 0

    for inc in incidents:
        d_km        = point_to_segment_km(inc, source, destination)
        threshold   = inc["radius_m"] / 1000.0
        if d_km <= threshold:
            delay       = PENALTY_MAP.get((inc["type"], inc["severity"]), 0)
            added_delay += delay
            risk_score  += delay

    total = base_min + added_delay
    if added_delay >= 25:
        recommendation = "Use alternate route"
    elif added_delay >= 10:
        recommendation = "Expect delays"
    else:
        recommendation = "Current route acceptable"

    return {
        "travel_time_min": total,
        "added_delay_min": added_delay,
        "risk_score":      risk_score,
        "recommendation":  recommendation,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Map HTML builder (Google Maps)
# ──────────────────────────────────────────────────────────────────────────────

def build_map_html(api_key: str, source: dict, destination: dict,
                    incidents: list, geocode_mode: bool = False) -> str:
    """
    Generate a self-contained Google Maps HTML page with:
      - Route directions (Directions API)
      - Traffic layer
      - Incident markers + radius circles
      - Places Autocomplete for source / destination (when geocode_mode=True)
    """
    center_lat = (source["lat"] + destination["lat"]) / 2
    center_lng = (source["lng"] + destination["lng"]) / 2
    inc_json   = json.dumps(incidents)
    src_json   = json.dumps(source)
    dst_json   = json.dumps(destination)

    if not api_key:
        return """
        <html><body style="font-family:Arial;padding:24px;background:#0d1422;color:#cde2f5;">
        <h3 style="color:#00d2ff;">⚠ Google Maps API key not set</h3>
        <p>Set <b>GOOGLE_MAPS_API_KEY</b> as an environment variable or in the sidebar.</p>
        <p>The live map with traffic layer, route directions, and incident markers
        will appear here once the key is provided.</p>
        </body></html>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <style>
    html,body,#map {{ height:100%; width:100%; margin:0; padding:0; }}
    #controls {{
      position:absolute; top:10px; left:10px; z-index:5;
      background:rgba(8,12,20,0.92); border:1px solid rgba(0,210,255,0.2);
      border-radius:8px; padding:10px 14px; font-family:Arial,sans-serif;
      color:#cde2f5; font-size:13px; min-width:260px;
    }}
    #controls input {{
      width:100%; padding:6px 8px; margin:4px 0;
      background:#111b2e; border:1px solid rgba(0,210,255,0.25);
      border-radius:4px; color:#cde2f5; font-size:13px; box-sizing:border-box;
    }}
    #controls button {{
      width:100%; margin-top:6px; padding:7px;
      background:transparent; border:1px solid #00d2ff;
      color:#00d2ff; border-radius:4px; cursor:pointer; font-size:12px;
      letter-spacing:.06em; text-transform:uppercase;
    }}
    #controls button:hover {{ background:#00d2ff; color:#080c14; }}
    #status {{ margin-top:6px; font-size:11px; color:#4a6070; min-height:16px; }}
  </style>
</head>
<body>
<div id="controls">
  <div style="color:#00d2ff;font-weight:700;margin-bottom:8px;font-size:12px;letter-spacing:.1em;">ROUTE PLANNER</div>
  <input id="from-input" type="text" placeholder="From (e.g. LAX Airport)"/>
  <input id="to-input"   type="text" placeholder="To   (e.g. Downtown LA)"/>
  <button onclick="calcRoute()">Calculate Route</button>
  <div id="status"></div>
</div>
<div id="map"></div>
<script>
  var map, directionsService, directionsRenderer;
  var source      = {src_json};
  var destination = {dst_json};
  var incidents   = {inc_json};

  function getIncidentIcon(type) {{
    var icons = {{
      accident:     'http://maps.google.com/mapfiles/ms/icons/red-dot.png',
      rally:        'http://maps.google.com/mapfiles/ms/icons/orange-dot.png',
      roadblock:    'http://maps.google.com/mapfiles/ms/icons/purple-dot.png',
      construction: 'http://maps.google.com/mapfiles/ms/icons/yellow-dot.png',
      waterlogging: 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png'
    }};
    return icons[type] || icons.accident;
  }}

  function getCircleColor(severity) {{
    return {{low:'#ffd166', medium:'#ff9f1c', high:'#ff4d4d'}}[severity] || '#ff6b35';
  }}

  function initMap() {{
    var center = {{ lat: {center_lat}, lng: {center_lng} }};
    map = new google.maps.Map(document.getElementById('map'), {{
      zoom: 12, center: center,
      styles: [
        {{elementType:'geometry', stylers:[{{color:'#080c14'}}]}},
        {{elementType:'labels.text.fill', stylers:[{{color:'#cde2f5'}}]}},
        {{elementType:'labels.text.stroke', stylers:[{{color:'#0d1422'}}]}},
        {{featureType:'road', elementType:'geometry', stylers:[{{color:'#1a2744'}}]}},
        {{featureType:'road.arterial', elementType:'geometry', stylers:[{{color:'#1e3a5f'}}]}},
        {{featureType:'water', elementType:'geometry', stylers:[{{color:'#0a1628'}}]}}
      ]
    }});

    // Traffic layer
    var traffic = new google.maps.TrafficLayer();
    traffic.setMap(map);

    directionsService  = new google.maps.DirectionsService();
    directionsRenderer = new google.maps.DirectionsRenderer({{
      map: map,
      polylineOptions: {{ strokeColor:'#00d2ff', strokeWeight:5, strokeOpacity:0.85 }}
    }});

    // Autocomplete
    var fromInput = document.getElementById('from-input');
    var toInput   = document.getElementById('to-input');
    new google.maps.places.Autocomplete(fromInput);
    new google.maps.places.Autocomplete(toInput);

    // Default route from data coords
    calcRouteCoords(source, destination);

    // Incident markers
    incidents.forEach(function(inc) {{
      new google.maps.Marker({{
        position: {{lat: inc.lat, lng: inc.lng}},
        map: map,
        title: inc.type + ' — ' + inc.severity,
        icon: getIncidentIcon(inc.type),
        animation: google.maps.Animation.DROP
      }});
      new google.maps.Circle({{
        strokeColor: getCircleColor(inc.severity),
        strokeOpacity: 0.7,
        strokeWeight: 2,
        fillColor: getCircleColor(inc.severity),
        fillOpacity: 0.1,
        map: map,
        center: {{lat: inc.lat, lng: inc.lng}},
        radius: inc.radius_m
      }});
    }});
  }}

  function calcRoute() {{
    var from = document.getElementById('from-input').value.trim();
    var to   = document.getElementById('to-input').value.trim();
    if (!from || !to) {{ document.getElementById('status').innerText='Enter both locations.'; return; }}
    document.getElementById('status').innerText = 'Calculating…';
    directionsService.route({{
      origin: from, destination: to,
      travelMode: google.maps.TravelMode.DRIVING,
      drivingOptions: {{ departureTime: new Date(), trafficModel: 'bestguess' }}
    }}, function(result, status) {{
      if (status === 'OK') {{
        directionsRenderer.setDirections(result);
        var leg = result.routes[0].legs[0];
        document.getElementById('status').innerText =
          leg.distance.text + ' — ' + leg.duration_in_traffic.text + ' (with traffic)';
      }} else {{
        document.getElementById('status').innerText = 'Route error: ' + status;
      }}
    }});
  }}

  function calcRouteCoords(src, dst) {{
    directionsService.route({{
      origin: new google.maps.LatLng(src.lat, src.lng),
      destination: new google.maps.LatLng(dst.lat, dst.lng),
      travelMode: google.maps.TravelMode.DRIVING
    }}, function(result, status) {{
      if (status === 'OK') directionsRenderer.setDirections(result);
    }});
  }}
</script>
<script async defer
  src="https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=places&callback=initMap">
</script>
</body>
</html>"""

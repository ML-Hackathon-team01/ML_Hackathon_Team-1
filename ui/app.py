"""
PathSnap Evaluator User Interface (Laptop 2).

Simple, clean, professional Streamlit interface that:
  1. Connects to Laptop 1's FastAPI backend over HTTP/LAN.
  2. Displays the 10 available evaluation trips fetched live from GET /trips.
  3. Sends the selected trip_id to POST /predict on Laptop 1.
  4. Renders the returned reconstructed route on an interactive Folium map
     alongside server-side evaluation metrics and API response timing.

Ground-truth labels remain strictly server-side on Laptop 1.
"""

from __future__ import annotations

import folium
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="PathSnap | Distributed Map-Matching Evaluator",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle, clean styling
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      .header-bar {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.25rem;
      }
      .header-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
      }
      .header-sub {
        font-size: 0.92rem;
        color: #475569;
        margin: 0.2rem 0 0 0;
      }
      .status-pill-ok {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-weight: 600;
        font-size: 0.82rem;
      }
      .status-pill-err {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        background: #fee2e2;
        color: #991b1b;
        font-weight: 600;
        font-size: 0.82rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_health(api_url: str) -> dict | None:
    try:
        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_trips(api_url: str) -> list[dict]:
    r = requests.get(f"{api_url.rstrip('/')}/trips", timeout=8)
    r.raise_for_status()
    return r.json().get("trips", [])


def run_prediction(api_url: str, trip_id: str) -> dict:
    r = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json={"trip_id": trip_id},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def build_route_map(result: dict) -> str:
    """Build interactive OpenStreetMap Folium HTML from API response payload."""
    map_data = result["map_data"]
    gps_pts = map_data["gps_points"]
    pred_segs = map_data["predicted_route_segments"]
    center = map_data["center"]
    bounds = map_data["bounds"]
    trip_id = result["trip_id"]
    metrics = result["metrics"]

    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    # Layer 1: Reconstructed route (Classifier + HMM/Viterbi)
    fg_route = folium.FeatureGroup(name="Reconstructed Route (Gradient Boosting + HMM)", show=True)
    if pred_segs:
        folium.PolyLine(
            pred_segs,
            color="#dc2626",
            weight=4.5,
            opacity=0.9,
            tooltip=f"Predicted Route — Trip {trip_id}",
        ).add_to(fg_route)
    fg_route.add_to(m)

    # Layer 2: Raw noisy GPS pings
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    for pt in gps_pts:
        folium.CircleMarker(
            location=pt,
            radius=2.6,
            color="#2563eb",
            fill=True,
            fill_color="#2563eb",
            fill_opacity=0.75,
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # Start & End markers
    if gps_pts:
        folium.Marker(
            location=gps_pts[0],
            popup="Trip Start",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)
        folium.Marker(
            location=gps_pts[-1],
            popup="Trip End",
            icon=folium.Icon(color="darkred", icon="flag-checkered", prefix="fa"),
        ).add_to(m)

    acc_txt = (
        f"{metrics['hmm_top1_accuracy'] * 100:.2f}%"
        if metrics.get("hmm_top1_accuracy") is not None
        else "N/A"
    )
    cont_txt = f"{metrics['hmm_route_continuity'] * 100:.2f}%"
    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background: white; border: 1px solid #cbd5e1; border-radius: 8px;
                padding: 10px 14px; font-size: 13px; box-shadow: 0 2px 8px rgba(15,23,42,0.12);">
      <b>Trip {trip_id} — Reconstructed Route</b><br>
      Top-1 Accuracy: <b>{acc_txt}</b> &nbsp;|&nbsp; Continuity: <b>{cont_txt}</b><br>
      <span style="color:#2563eb;">&#9679;</span> Raw GPS Observations<br>
      <span style="color:#dc2626;">&#9473;</span> Predicted Route (GB + HMM/Viterbi)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(bounds)
    return m.get_root().render()


# ==================== HEADER ====================
st.markdown(
    """
    <div class="header-bar">
      <p class="header-title">PathSnap — Distributed Map-Matching &amp; Route Reconstruction</p>
      <p class="header-sub">Team-01 Evaluator Interface (Laptop 2) &nbsp;&bull;&nbsp; Connected to FastAPI Inference Backend (Laptop 1)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.subheader("Backend Connection")
    api_url = st.text_input(
        "Laptop 1 API URL",
        value="http://127.0.0.1:8000",
        help="Use http://<LAPTOP1_IP>:8000 for two-laptop evaluation over LAN/Wi-Fi.",
    )

    health = fetch_health(api_url)
    if health is not None:
        st.markdown('<span class="status-pill-ok">&#9679; API Online</span>', unsafe_allow_html=True)
        st.caption(
            f"**Model:** {health['model_name']}  \n"
            f"**Decoder:** {health['decoder']}  \n"
            f"**Partition:** `{health['eval_partition']}` ({health['available_trips']} trips)"
        )
    else:
        st.markdown('<span class="status-pill-err">&#9679; API Offline</span>', unsafe_allow_html=True)
        st.warning("Cannot reach Laptop 1 API. Verify FastAPI is running and check the URL/IP.")

    st.divider()
    st.subheader("Evaluation Trip Selection")

    trips = []
    if health is not None:
        try:
            trips = fetch_trips(api_url)
        except Exception as exc:
            st.error(f"Failed to fetch trips: {exc}")

    selected_trip_id = None
    if trips:
        options = {f"Trip {t['trip_id']} ({t['n_gps_points']:,} GPS pts)": t["trip_id"] for t in trips}
        chosen_label = st.selectbox("Select one of the evaluation trips", list(options.keys()))
        selected_trip_id = options[chosen_label]
        run_btn = st.button("Reconstruct Route", type="primary", use_container_width=True)
    else:
        run_btn = False
        st.info("Connect to Laptop 1 API to load the 10 evaluation trips.")

    st.divider()
    st.caption(
        "Ground-truth `.route` files remain strictly on Laptop 1. "
        "This UI sends only the selected `trip_id` to `/predict` and displays the API response."
    )

# ==================== MAIN CONTENT ====================
if run_btn and selected_trip_id:
    with st.spinner(f"Sending Trip {selected_trip_id} to Laptop 1 API for live inference & HMM decoding..."):
        try:
            st.session_state["last_result"] = run_prediction(api_url, selected_trip_id)
        except Exception as exc:
            st.error(f"API request failed: {exc}")

result = st.session_state.get("last_result")

if result is None:
    st.info(
        "Select any of the **10 evaluation trips** in the left sidebar and click **Reconstruct Route** "
        "to invoke Laptop 1's live Gradient Boosting + HMM/Viterbi pipeline."
    )
else:
    metrics = result["metrics"]
    timing = result["timing_ms"]
    summary = result["trip_summary"]
    model_info = result["model_info"]

    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    hmm_acc = metrics.get("hmm_top1_accuracy")
    clf_acc = metrics.get("classifier_only_top1_accuracy")
    acc_delta = (
        f"{(hmm_acc - clf_acc) * 100:+.2f}% vs Classifier-only"
        if (hmm_acc is not None and clf_acc is not None)
        else None
    )
    c1.metric(
        "Observation Top-1 Accuracy",
        f"{hmm_acc * 100:.2f}%" if hmm_acc is not None else "N/A",
        delta=acc_delta,
    )

    hmm_cont = metrics["hmm_route_continuity"]
    clf_cont = metrics["classifier_only_route_continuity"]
    c2.metric(
        "Route Continuity (HMM)",
        f"{hmm_cont * 100:.2f}%",
        delta=f"{(hmm_cont - clf_cont) * 100:+.2f}% vs Classifier-only",
    )

    arc_f1 = metrics.get("route_arc_f1")
    c3.metric(
        "Route Arc F1-Score",
        f"{arc_f1 * 100:.2f}%" if arc_f1 is not None else "N/A",
    )
    c4.metric(
        "End-to-End API Latency",
        f"{timing['total_ms']:.0f} ms",
        delta=f"ML + HMM: {timing['classifier_inference_ms'] + timing['hmm_viterbi_ms']:.0f} ms",
        delta_color="off",
    )

    # Interactive Folium Map
    st.subheader(f"Reconstructed Trajectory — Trip {result['trip_id']}")
    map_html = build_route_map(result)
    components.html(map_html, height=520, scrolling=False)

    # Detailed Breakdown Columns
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Classifier-Only vs. Classifier + HMM/Viterbi")
        comp_df = pd.DataFrame([
            {
                "Stage": "Classifier-Only (Gradient Boosting)",
                "Top-1 Accuracy": f"{clf_acc * 100:.2f}%" if clf_acc is not None else "N/A",
                "Route Continuity": f"{clf_cont * 100:.2f}%",
            },
            {
                "Stage": "Classifier + HMM/Viterbi",
                "Top-1 Accuracy": f"{hmm_acc * 100:.2f}%" if hmm_acc is not None else "N/A",
                "Route Continuity": f"{hmm_cont * 100:.2f}%",
            },
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with right_col:
        st.subheader("Trip & Server Pipeline Diagnostics")
        diag_df = pd.DataFrame([
            {"Property": "Model Artifact", "Value": model_info["model_artifact"]},
            {"Property": "GPS Observations", "Value": f"{summary['n_gps_points']:,}"},
            {"Property": "Road Network Size", "Value": f"{summary['n_road_nodes']:,} nodes / {summary['n_road_arcs']:,} arcs"},
            {"Property": "Reconstructed Road Segments", "Value": f"{summary['n_reconstructed_segments']:,}"},
            {"Property": "Load + Feature Prep", "Value": f"{timing['load_and_clean_ms'] + timing['candidate_and_features_ms']:.1f} ms"},
            {"Property": "Classifier + Viterbi", "Value": f"{timing['classifier_inference_ms']:.1f} ms + {timing['hmm_viterbi_ms']:.1f} ms"},
        ])
        st.dataframe(diag_df, use_container_width=True, hide_index=True)

    with st.expander("Inspect Raw API Response Metadata (Proof of Live API Call)"):
        st.json({
            "trip_id": result["trip_id"],
            "partition": result["partition"],
            "model_info": model_info,
            "trip_summary": summary,
            "metrics": metrics,
            "timing_ms": timing,
        })

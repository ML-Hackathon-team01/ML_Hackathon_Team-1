"""
PathSnap: Interactive Folium Geographical Visualisation
Corresponds to: Deliverable 1 (Visualisations) & Deliverable 2 (Section 6: Geographical Visualisation)

Visualises 4 mandatory layers on an interactive map:
  1. Original GPS observations (Blue markers & track)
  2. Relevant road-network geometry (Subtle gray lines)
  3. Ground-truth reference route (Thick Green line)
  4. Model-predicted route (Red dashed line)
Includes LayerControl for toggling layers and a floating legend.
"""

import argparse
from pathlib import Path
import folium
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT.parent / "Student folder"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_trip_data(trip_id: str, partition: str = "train"):
    """Load the raw files for the selected trip."""
    trip_folder = DATASET_ROOT / partition / trip_id

    if not trip_folder.exists():
        raise FileNotFoundError(f"Trip folder not found: {trip_folder}")

    # 1. Track: longitude, latitude, timestamp
    track = pd.read_csv(
        trip_folder / f"{trip_id}.track",
        sep="\t",
        header=None,
        names=["longitude", "latitude", "timestamp"],
    )

    # 2. Nodes: longitude, latitude
    nodes = pd.read_csv(
        trip_folder / f"{trip_id}.nodes",
        sep="\t",
        header=None,
        names=["longitude", "latitude"],
    )
    nodes["node_id"] = np.arange(len(nodes))

    # 3. Arcs: from_node, to_node
    arcs = pd.read_csv(
        trip_folder / f"{trip_id}.arcs",
        sep="\t",
        header=None,
        names=["from_node", "to_node"],
    )
    arcs["arc_id"] = np.arange(len(arcs))

    # 4. Route (ground truth if available)
    route_file = trip_folder / f"{trip_id}.route"
    route = None
    if route_file.exists():
        route = pd.read_csv(route_file, header=None, names=["arc_id"])

    return track, nodes, arcs, route


def get_candidate_features_and_predictions(track, nodes, arcs, model):
    """
    Generate candidates using BallTree and predict the top candidate
    for each GPS point using the trained model.
    """
    node_lookup = nodes.set_index("node_id")
    start_nodes = node_lookup.loc[arcs["from_node"].to_numpy()]
    end_nodes = node_lookup.loc[arcs["to_node"].to_numpy()]

    mid_lon = (start_nodes["longitude"].to_numpy() + end_nodes["longitude"].to_numpy()) / 2
    mid_lat = (start_nodes["latitude"].to_numpy() + end_nodes["latitude"].to_numpy()) / 2

    # Build BallTree on road midpoints (in radians)
    road_coords_rad = np.radians(np.column_stack([mid_lat, mid_lon]))
    tree = BallTree(road_coords_rad, metric="haversine")

    gps_coords_rad = np.radians(track[["latitude", "longitude"]].to_numpy())
    k = min(5, len(arcs))
    distances_rad, positions = tree.query(gps_coords_rad, k=k)
    distances_m = distances_rad * 6371000

    arc_ids = arcs["arc_id"].to_numpy()[positions]

    # Build candidate dataframe
    candidates = pd.DataFrame({
        "observation_id": np.repeat(np.arange(len(track)), k),
        "candidate_rank": np.tile(np.arange(1, k + 1), len(track)),
        "arc_id": arc_ids.ravel(),
        "distance_m": distances_m.ravel(),
    })

    # Extract GPS movement features
    lon = track["longitude"].to_numpy()
    lat = track["latitude"].to_numpy()
    timestamp = track["timestamp"].to_numpy()

    delta_lon = np.diff(lon, prepend=lon[0])
    delta_lat = np.diff(lat, prepend=lat[0])
    delta_time = np.diff(timestamp, prepend=timestamp[0])

    gps_distance = np.sqrt(
        (delta_lon * 111320 * np.cos(np.radians(lat))) ** 2
        + (delta_lat * 110540) ** 2
    )
    gps_speed = np.divide(
        gps_distance,
        delta_time,
        out=np.zeros_like(gps_distance, dtype=float),
        where=delta_time > 0,
    )
    gps_bearing = (np.degrees(np.arctan2(delta_lon, delta_lat)) + 360) % 360

    # Extract road bearing
    arc_geom = pd.DataFrame({
        "arc_id": arcs["arc_id"].to_numpy(),
        "start_lon": start_nodes["longitude"].to_numpy(),
        "start_lat": start_nodes["latitude"].to_numpy(),
        "end_lon": end_nodes["longitude"].to_numpy(),
        "end_lat": end_nodes["latitude"].to_numpy(),
    }).set_index("arc_id")

    cand_geom = arc_geom.loc[candidates["arc_id"].to_numpy()]
    road_dlon = cand_geom["end_lon"].to_numpy() - cand_geom["start_lon"].to_numpy()
    road_dlat = cand_geom["end_lat"].to_numpy() - cand_geom["start_lat"].to_numpy()
    road_bearing = (np.degrees(np.arctan2(road_dlon, road_dlat)) + 360) % 360

    cand_gps_bearing = gps_bearing[candidates["observation_id"].to_numpy()]
    dir_diff = np.abs(road_bearing - cand_gps_bearing)
    dir_diff = np.minimum(dir_diff, 360 - dir_diff)

    candidates["direction_difference"] = dir_diff
    candidates["gps_speed_mps"] = gps_speed[candidates["observation_id"].to_numpy()]
    candidates["time_gap_s"] = delta_time[candidates["observation_id"].to_numpy()]

    # Predict using the trained model
    feature_cols = [
        "candidate_rank",
        "distance_m",
        "direction_difference",
        "gps_speed_mps",
        "time_gap_s",
    ]
    X = candidates[feature_cols].copy()
    probs = model.predict_proba(X)[:, 1]
    candidates["score"] = probs

    # Pick top candidate per GPS point
    best_cands = (
        candidates.sort_values(
            ["observation_id", "score"], ascending=[True, False]
        )
        .groupby("observation_id")
        .first()
        .reset_index()
    )

    return best_cands["arc_id"].tolist(), arc_geom


def create_folium_map(trip_id: str, partition: str = "train"):
    print(f"\n--- Generating Folium Map for Trip {trip_id} ({partition}) ---")
    track, nodes, arcs, route = load_trip_data(trip_id, partition)

    # Load trained model
    model_path = MODELS_DIR / "decision_tree_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_models.py first.")
    model = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")

    # Get predictions
    predicted_arc_ids, arc_geom = get_candidate_features_and_predictions(
        track, nodes, arcs, model
    )

    # Center map on median GPS coordinate
    center_lat = track["latitude"].median()
    center_lon = track["longitude"].median()
    # Using Esri WorldStreetMap tiles which do not block local browsers with 403
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom",
        name="Esri World Street Map",
    )

    # Alternative clean base layer: CartoDB Light
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CartoDB Light (Clean)",
        subdomains="abcd",
    ).add_to(fmap)

    # ----------------------------------------------------
    # LAYER 1: Road Network Geometry (Nearby Arcs)
    # Filter arcs within bounding box of track (+ 0.005 padding)
    # ----------------------------------------------------
    lat_min, lat_max = track["latitude"].min() - 0.005, track["latitude"].max() + 0.005
    lon_min, lon_max = track["longitude"].min() - 0.005, track["longitude"].max() + 0.005

    nearby_arcs = arc_geom[
        (arc_geom["start_lat"].between(lat_min, lat_max))
        & (arc_geom["start_lon"].between(lon_min, lon_max))
    ]

    layer_network = folium.FeatureGroup(name="1. Road Network Geometry (Nearby Arcs)", show=False)
    for _, arc in nearby_arcs.iterrows():
        folium.PolyLine(
            locations=[[arc["start_lat"], arc["start_lon"]], [arc["end_lat"], arc["end_lon"]]],
            color="#888888",
            weight=1.5,
            opacity=0.4,
        ).add_to(layer_network)
    layer_network.add_to(fmap)

    # ----------------------------------------------------
    # LAYER 2: Original GPS Observations
    # ----------------------------------------------------
    layer_gps = folium.FeatureGroup(name="2. Original GPS Observations (Raw)", show=True)
    gps_coords = track[["latitude", "longitude"]].values.tolist()

    # Blue path connecting GPS pings
    folium.PolyLine(
        gps_coords,
        color="#007bff",
        weight=2,
        opacity=0.6,
        dash_array="3, 3",
        tooltip=f"Trip {trip_id}: GPS Path",
    ).add_to(layer_gps)

    # Blue circle markers for each observation (sample every 2 points if > 500 to keep HTML fast)
    step = 1 if len(track) < 500 else 2
    for idx in range(0, len(track), step):
        row = track.iloc[idx]
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color="#0056b3",
            fill=True,
            fill_color="#17a2b8",
            fill_opacity=0.8,
            popup=f"Obs #{idx}<br>Time: {row['timestamp']:.1f}s",
        ).add_to(layer_gps)
    layer_gps.add_to(fmap)

    # ----------------------------------------------------
    # LAYER 3: Reference Ground-Truth Route (Green)
    # ----------------------------------------------------
    if route is not None:
        layer_route = folium.FeatureGroup(name="3. Ground-Truth Route (Reference)", show=True)
        route_arcs = arc_geom.reindex(route["arc_id"].values).dropna()

        for _, arc in route_arcs.iterrows():
            folium.PolyLine(
                locations=[[arc["start_lat"], arc["start_lon"]], [arc["end_lat"], arc["end_lon"]]],
                color="#28a745",
                weight=5,
                opacity=0.85,
                tooltip=f"True Arc #{int(arc.name)}",
            ).add_to(layer_route)
        layer_route.add_to(fmap)

    # ----------------------------------------------------
    # LAYER 4: Predicted Route - Decision Tree (Red Dashed)
    # ----------------------------------------------------
    layer_pred = folium.FeatureGroup(name="4. Predicted Route (Decision Tree)", show=True)
    # Drop consecutive identical arcs for cleaner drawing
    unique_pred_arcs = pd.Series(predicted_arc_ids).drop_duplicates().tolist()
    pred_arc_records = arc_geom.reindex(unique_pred_arcs).dropna()

    for _, arc in pred_arc_records.iterrows():
        folium.PolyLine(
            locations=[[arc["start_lat"], arc["start_lon"]], [arc["end_lat"], arc["end_lon"]]],
            color="#dc3545",
            weight=3.5,
            opacity=0.9,
            dash_array="6, 6",
            tooltip=f"Predicted Arc #{int(arc.name)}",
        ).add_to(layer_pred)
    layer_pred.add_to(fmap)

    # ----------------------------------------------------
    # ADD CONTROLS & FLOATING LEGEND
    # ----------------------------------------------------
    folium.LayerControl(collapsed=False).add_to(fmap)

    # Floating HTML Legend
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 270px;
        background-color: white;
        border: 2px solid #333333;
        border-radius: 8px;
        z-index: 9999;
        font-size: 13px;
        font-family: Arial, sans-serif;
        padding: 12px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
        <b>PathSnap Map Matching (Trip {trip_id})</b><br><br>
        <span style="color: #0056b3; font-size: 16px;">●</span> Raw GPS Observations<br>
        <span style="color: #28a745; font-size: 18px; font-weight: bold;">―</span> Ground-Truth Route (.route)<br>
        <span style="color: #dc3545; font-size: 18px; font-weight: bold;">- -</span> Predicted Route (Decision Tree)<br>
        <span style="color: #888888; font-size: 18px;">―</span> Road Network Geometry<br>
        <hr style="margin: 8px 0;">
        <small style="color: #666;">Use layer control in top-right to toggle layers.</small>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # Save output
    output_path = RESULTS_DIR / f"trip_{trip_id}_folium_map.html"
    fmap.save(str(output_path))
    print(f"Map successfully generated and saved to:\n  {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Folium Map for PathSnap Trip")
    parser.add_argument("--trip", type=str, default="00000001", help="Trip ID (e.g., 00000001)")
    parser.add_argument("--partition", type=str, default="train", help="Dataset partition (train or student_test)")
    args = parser.parse_args()

    create_folium_map(args.trip, args.partition)

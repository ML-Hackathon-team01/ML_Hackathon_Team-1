
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT.parent / "Student folder"

TRAIN_FOLDER = DATASET_ROOT / "train"
OUTPUT_FOLDER = PROJECT_ROOT / "results"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 3: CLEAN DATA
# ============================================================

def load_trip(trip_folder):

    trip_id = trip_folder.name

    # Load GPS observations
    track = pd.read_csv(
        trip_folder / f"{trip_id}.track",
        sep="\t",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    # Load road nodes
    nodes = pd.read_csv(
        trip_folder / f"{trip_id}.nodes",
        sep="\t",
        header=None,
        names=["longitude", "latitude"]
    )

    # Load road connections
    arcs = pd.read_csv(
        trip_folder / f"{trip_id}.arcs",
        sep="\t",
        header=None,
        names=["from_node", "to_node"]
    )

    # Load reference route
    route = pd.read_csv(
        trip_folder / f"{trip_id}.route",
        header=None,
        names=["arc_id"]
    )

    # Preserve original node IDs
    nodes["node_id"] = np.arange(len(nodes))

    # Preserve original arc IDs
    arcs["arc_id"] = np.arange(len(arcs))

    return track, nodes, arcs, route


def clean_data(track, nodes, arcs, route):

    report = {}

    # ---------------- TRACK CLEANING ----------------

    original_track_count = len(track)

    # Remove missing values
    track = track.dropna().copy()

    # Remove infinite values
    track = track[
        np.isfinite(track["longitude"]) &
        np.isfinite(track["latitude"]) &
        np.isfinite(track["timestamp"])
    ].copy()

    # Keep valid geographical coordinates
    track = track[
        track["longitude"].between(-180, 180) &
        track["latitude"].between(-90, 90)
    ].copy()

    # Remove exact duplicate observations
    track = track.drop_duplicates()

    # Sort GPS observations by timestamp
    track = track.sort_values("timestamp").reset_index(drop=True)

    report["track_rows_removed"] = (
        original_track_count - len(track)
    )

    # ---------------- NODE CLEANING ----------------

    original_node_count = len(nodes)

    nodes = nodes.dropna().copy()

    nodes = nodes[
        np.isfinite(nodes["longitude"]) &
        np.isfinite(nodes["latitude"])
    ].copy()

    nodes = nodes[
        nodes["longitude"].between(-180, 180) &
        nodes["latitude"].between(-90, 90)
    ].copy()

    # Keep the original node IDs
    valid_node_ids = set(nodes["node_id"])

    report["nodes_removed"] = (
        original_node_count - len(nodes)
    )

    # ---------------- ARC CLEANING ----------------

    original_arc_count = len(arcs)

    arcs = arcs.dropna().copy()

    # Keep only arcs connected to existing nodes
    arcs = arcs[
        arcs["from_node"].isin(valid_node_ids) &
        arcs["to_node"].isin(valid_node_ids)
    ].copy()

    valid_arc_ids = set(arcs["arc_id"])

    report["arcs_removed"] = (
        original_arc_count - len(arcs)
    )

    # ---------------- ROUTE CLEANING ----------------

    original_route_count = len(route)

    route = route.dropna().copy()

    # Keep only route IDs that exist in the road network
    route = route[
        route["arc_id"].isin(valid_arc_ids)
    ].copy()

    report["route_entries_removed"] = (
        original_route_count - len(route)
    )

    # ---------------- SPEED CHECK ----------------

    lon = track["longitude"].to_numpy()
    lat = track["latitude"].to_numpy()
    timestamp = track["timestamp"].to_numpy()

    delta_lon = np.diff(lon, prepend=lon[0])
    delta_lat = np.diff(lat, prepend=lat[0])
    delta_time = np.diff(timestamp, prepend=timestamp[0])

    distance = np.sqrt(
        (delta_lon * 111320 * np.cos(np.radians(lat))) ** 2
        + (delta_lat * 110540) ** 2
    )

    speed = np.divide(
        distance,
        delta_time,
        out=np.zeros_like(distance, dtype=float),
        where=delta_time > 0
    )

    # Flag unusually high speeds
    # Do not delete these points automatically
    report["suspicious_speed_points"] = int(
        np.sum(speed > 55)
    )

    # Remaining data sizes
    report["remaining_track_rows"] = len(track)
    report["remaining_nodes"] = len(nodes)
    report["remaining_arcs"] = len(arcs)
    report["remaining_route_rows"] = len(route)

    return track, nodes, arcs, route, report


# ============================================================
# STEP 4: GENERATE CANDIDATES
# Find 5 nearest road segments for every GPS point
# ============================================================

def generate_candidates(track, nodes, arcs, k=5):

    # Look up road nodes by their original IDs
    node_lookup = nodes.set_index("node_id")

    # Get starting and ending nodes of every arc
    start_nodes = node_lookup.loc[
        arcs["from_node"].to_numpy()
    ]

    end_nodes = node_lookup.loc[
        arcs["to_node"].to_numpy()
    ]

    # Approximate each road segment using its midpoint
    mid_lon = (
        start_nodes["longitude"].to_numpy()
        + end_nodes["longitude"].to_numpy()
    ) / 2

    mid_lat = (
        start_nodes["latitude"].to_numpy()
        + end_nodes["latitude"].to_numpy()
    ) / 2

    # BallTree requires latitude-longitude in radians
    road_coordinates = np.radians(
        np.column_stack([mid_lat, mid_lon])
    )

    # Build spatial search tree
    tree = BallTree(
        road_coordinates,
        metric="haversine"
    )

    # GPS coordinates
    gps_coordinates = np.radians(
        track[["latitude", "longitude"]].to_numpy()
    )

    # Number of candidates
    k = min(k, len(arcs))

    # Find nearest road midpoints
    distances, positions = tree.query(
        gps_coordinates,
        k=k
    )

    # Convert distances from radians to meters
    distances = distances * 6371000

    # Convert positions to original arc IDs
    arc_ids = arcs["arc_id"].to_numpy()[positions]

    # Build candidate table
    candidates = pd.DataFrame({
        "observation_id": np.repeat(
            np.arange(len(track)), k
        ),
        "candidate_rank": np.tile(
            np.arange(1, k + 1), len(track)
        ),
        "arc_id": arc_ids.ravel(),
        "distance_m": distances.ravel()
    })

    # Road segment geometry table
    arc_geometry = pd.DataFrame({
        "arc_id": arcs["arc_id"].to_numpy(),
        "start_lon": start_nodes["longitude"].to_numpy(),
        "start_lat": start_nodes["latitude"].to_numpy(),
        "end_lon": end_nodes["longitude"].to_numpy(),
        "end_lat": end_nodes["latitude"].to_numpy()
    })

    return candidates, arc_geometry


# ============================================================
# STEP 5: EXTRACT FEATURES
# ============================================================

def extract_features(track, candidates, arc_geometry):

    lon = track["longitude"].to_numpy()
    lat = track["latitude"].to_numpy()
    timestamp = track["timestamp"].to_numpy()

    # ---------------- GPS MOVEMENT ----------------

    delta_lon = np.diff(lon, prepend=lon[0])
    delta_lat = np.diff(lat, prepend=lat[0])
    delta_time = np.diff(timestamp, prepend=timestamp[0])

    # Approximate GPS movement distance in meters
    gps_distance = np.sqrt(
        (delta_lon * 111320 * np.cos(np.radians(lat))) ** 2
        + (delta_lat * 110540) ** 2
    )

    # GPS speed
    gps_speed = np.divide(
        gps_distance,
        delta_time,
        out=np.zeros_like(gps_distance, dtype=float),
        where=delta_time > 0
    )

    # GPS movement direction
    gps_bearing = (
        np.degrees(
            np.arctan2(delta_lon, delta_lat)
        ) + 360
    ) % 360

    # ---------------- ROAD DIRECTION ----------------

    geometry = arc_geometry.set_index("arc_id")

    candidate_geometry = geometry.loc[
        candidates["arc_id"].to_numpy()
    ]

    road_delta_lon = (
        candidate_geometry["end_lon"].to_numpy()
        - candidate_geometry["start_lon"].to_numpy()
    )

    road_delta_lat = (
        candidate_geometry["end_lat"].to_numpy()
        - candidate_geometry["start_lat"].to_numpy()
    )

    road_bearing = (
        np.degrees(
            np.arctan2(road_delta_lon, road_delta_lat)
        ) + 360
    ) % 360

    # GPS direction for each candidate
    candidate_gps_bearing = gps_bearing[
        candidates["observation_id"].to_numpy()
    ]

    # Difference between GPS and road directions
    direction_difference = np.abs(
        road_bearing - candidate_gps_bearing
    )

    # Make direction difference between 0 and 180 degrees
    direction_difference = np.minimum(
        direction_difference,
        360 - direction_difference
    )

    # ---------------- ADD FEATURES ----------------

    candidates["direction_difference"] = (
        direction_difference
    )

    candidates["gps_speed_mps"] = gps_speed[
        candidates["observation_id"].to_numpy()
    ]

    candidates["time_gap_s"] = delta_time[
        candidates["observation_id"].to_numpy()
    ]

    return candidates


# ============================================================
# STEP 6: PREPARE LABELS
# Preliminary labels using reference route
# ============================================================

def prepare_labels(candidates, route):

    # All arc IDs present in the reference route
    route_arc_ids = set(
        route["arc_id"].tolist()
    )

    # Preliminary binary labels
    candidates["label"] = (
        candidates["arc_id"]
        .isin(route_arc_ids)
        .astype(int)
    )

    return candidates


# ============================================================
# STEP 7: PREPARE ML TABLE
# ============================================================

def prepare_ml_table(candidates, trip_id):

    candidates["trip_id"] = trip_id

    # Columns used in the ML dataset
    ml_table = candidates[[
        "trip_id",
        "observation_id",
        "candidate_rank",
        "distance_m",
        "direction_difference",
        "gps_speed_mps",
        "time_gap_s",
        "label"
    ]].copy()

    # Replace infinite values
    ml_table = ml_table.replace(
        [np.inf, -np.inf], np.nan
    )

    # Remove rows containing missing feature values
    ml_table = ml_table.dropna()

    return ml_table


# ============================================================
# MAIN PIPELINE
# Process all 80 training trips
# ============================================================

all_trip_tables = []
all_cleaning_reports = []

trip_folders = sorted(
    folder for folder in TRAIN_FOLDER.iterdir()
    if folder.is_dir()
)

print("Total training trips:", len(trip_folders))

for trip_folder in trip_folders:

    trip_id = trip_folder.name

    print("\nProcessing:", trip_id)

    # Step 3: Load and clean data
    track, nodes, arcs, route = load_trip(trip_folder)

    track, nodes, arcs, route, report = clean_data(
        track, nodes, arcs, route
    )

    report["trip_id"] = trip_id

    all_cleaning_reports.append(report)

    # Step 4: Generate road candidates
    candidates, arc_geometry = generate_candidates(
        track, nodes, arcs, k=5
    )

    # Step 5: Extract features
    candidates = extract_features(
        track, candidates, arc_geometry
    )

    # Step 6: Prepare preliminary labels
    candidates = prepare_labels(
        candidates, route
    )

    # Step 7: Prepare ML table
    ml_table = prepare_ml_table(
        candidates, trip_id
    )

    all_trip_tables.append(ml_table)


# ============================================================
# COMBINE ALL TRIPS
# ============================================================

final_dataset = pd.concat(
    all_trip_tables,
    ignore_index=True
)

cleaning_report = pd.DataFrame(
    all_cleaning_reports
)


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

# Save preliminary ML dataset
ml_output_file = (
    OUTPUT_FOLDER / "pathsnap_ml_dataset.csv"
)

final_dataset.to_csv(
    ml_output_file,
    index=False
)

# Save cleaning report
report_output_file = (
    OUTPUT_FOLDER / "cleaning_report.csv"
)

cleaning_report.to_csv(
    report_output_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n===================================")
print("PREPROCESSING COMPLETED")
print("===================================")

print("\nTotal ML rows:", len(final_dataset))
print("Total ML columns:", len(final_dataset.columns))

print("\nML dataset saved to:")
print(ml_output_file)

print("\nCleaning report saved to:")
print(report_output_file)

print("\nLabel distribution:")
print(final_dataset["label"].value_counts())

print("\nFirst 5 rows:")
print(final_dataset.head())

print("\nCleaning report summary:")
print(cleaning_report.sum(numeric_only=True))

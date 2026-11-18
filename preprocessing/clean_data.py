
from pathlib import Path
import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT.parent / "Student folder"
TRAIN_FOLDER = DATASET_ROOT / "train"
OUTPUT_FOLDER = PROJECT_ROOT / "results"

OUTPUT_FOLDER.mkdir(exist_ok=True)

# Load one trip
def load_trip(trip_folder):

    trip_id = trip_folder.name

    track = pd.read_csv(
        trip_folder / f"{trip_id}.track",
        sep="\t",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        trip_folder / f"{trip_id}.nodes",
        sep="\t",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        trip_folder / f"{trip_id}.arcs",
        sep="\t",
        header=None,
        names=["from_node", "to_node"]
    )

    route = pd.read_csv(
        trip_folder / f"{trip_id}.route",
        header=None,
        names=["arc_id"]
    )

    # Preserve original IDs
    nodes["node_id"] = np.arange(len(nodes))
    arcs["arc_id"] = np.arange(len(arcs))

    return track, nodes, arcs, route


# Clean data
def clean_trip(track, nodes, arcs, route):

    report = {}

    # ---------- TRACK CLEANING ----------

    original_track_rows = len(track)

    track = track.dropna().copy()

    track = track[
        track["longitude"].between(-180, 180) &
        track["latitude"].between(-90, 90)
    ].copy()

    track = track[np.isfinite(track["timestamp"])].copy()

    # Remove exact duplicate observations only
    track = track.drop_duplicates()

    # Sort observations by time
    track = track.sort_values("timestamp").reset_index(drop=True)

    report["track_removed_rows"] = original_track_rows - len(track)

    # ---------- NODE CLEANING ----------

    original_nodes = len(nodes)

    nodes = nodes.dropna().copy()

    nodes = nodes[
        nodes["longitude"].between(-180, 180) &
        nodes["latitude"].between(-90, 90)
    ].copy()

    report["invalid_nodes_removed"] = original_nodes - len(nodes)

    # Keep original node IDs, do not renumber
    valid_node_ids = set(nodes["node_id"])

    # ---------- ARC CLEANING ----------

    original_arcs = len(arcs)

    arcs = arcs.dropna().copy()

    arcs = arcs[
        arcs["from_node"].isin(valid_node_ids) &
        arcs["to_node"].isin(valid_node_ids)
    ].copy()

    report["invalid_arcs_removed"] = original_arcs - len(arcs)

    valid_arc_ids = set(arcs["arc_id"])

    # ---------- ROUTE CLEANING ----------

    original_route = len(route)

    route = route.dropna().copy()

    route = route[route["arc_id"].isin(valid_arc_ids)].copy()

    report["invalid_route_arcs_removed"] = (
        original_route - len(route)
    )

    # ---------- SPEED CHECK (FLAG ONLY) ----------

    delta_lon = np.diff(
        track["longitude"], prepend=track["longitude"].iloc[0]
    )

    delta_lat = np.diff(
        track["latitude"], prepend=track["latitude"].iloc[0]
    )

    delta_time = np.diff(
        track["timestamp"], prepend=track["timestamp"].iloc[0]
    )

    distance = np.sqrt(
        (delta_lon * 111320 * np.cos(np.radians(track["latitude"]))) ** 2
        + (delta_lat * 110540) ** 2
    )

    speed = np.divide(
        distance,
        delta_time,
        out=np.zeros_like(distance, dtype=float),
        where=delta_time > 0
    )

    # Flag suspicious speeds, do not remove them automatically
    report["suspicious_speed_points"] = int(
        np.sum(speed > 55)
    )

    report["remaining_track_rows"] = len(track)
    report["remaining_nodes"] = len(nodes)
    report["remaining_arcs"] = len(arcs)
    report["remaining_route"] = len(route)

    return track, nodes, arcs, route, report


# Process all 80 training trips
all_reports = []

trip_folders = sorted(
    folder for folder in TRAIN_FOLDER.iterdir()
    if folder.is_dir()
)

print("Total training trips:", len(trip_folders))

for trip_folder in trip_folders:

    trip_id = trip_folder.name

    track, nodes, arcs, route = load_trip(trip_folder)

    track, nodes, arcs, route, report = clean_trip(
        track, nodes, arcs, route
    )

    report["trip_id"] = trip_id

    all_reports.append(report)

    print("Cleaned:", trip_id)


# Save report
report_df = pd.DataFrame(all_reports)

report_file = OUTPUT_FOLDER / "cleaning_report.csv"

report_df.to_csv(report_file, index=False)

print("\n========== CLEANING COMPLETED ==========")
print(report_df.sum(numeric_only=True))
print("\nReport saved:", report_file)

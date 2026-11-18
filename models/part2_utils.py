"""
Shared Part 2 utilities for ML_Hackathon_Team-1.

Consumes the teammate's Part 1 dataset (`results/pathsnap_ml_dataset.csv` built
by `preprocessing/prepare_dataset.py`) and builds the matching `student_test`
dataset using the EXACT same Part 1 cleaning, candidate generation (k=5 BallTree
haversine), feature extraction (5 features), and labeling logic.

Never modifies teammate Part 1 scripts and strictly never touches evaluator_test.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT.parent / "Student folder"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# Exact 5 feature columns produced by preprocessing/prepare_dataset.py
FEATURE_COLS: list[str] = [
    "candidate_rank",
    "distance_m",
    "direction_difference",
    "gps_speed_mps",
    "time_gap_s",
]


def _load_and_clean_trip(trip_folder: Path):
    """Exact replica of preprocessing/prepare_dataset.py load_trip + clean_data."""
    trip_id = trip_folder.name
    track = pd.read_csv(
        trip_folder / f"{trip_id}.track",
        sep="\t",
        header=None,
        names=["longitude", "latitude", "timestamp"],
    )
    nodes = pd.read_csv(
        trip_folder / f"{trip_id}.nodes",
        sep="\t",
        header=None,
        names=["longitude", "latitude"],
    )
    arcs = pd.read_csv(
        trip_folder / f"{trip_id}.arcs",
        sep="\t",
        header=None,
        names=["from_node", "to_node"],
    )
    route = pd.read_csv(
        trip_folder / f"{trip_id}.route",
        header=None,
        names=["arc_id"],
    )
    nodes["node_id"] = np.arange(len(nodes))
    arcs["arc_id"] = np.arange(len(arcs))

    track = track.dropna().copy()
    track = track[
        np.isfinite(track["longitude"])
        & np.isfinite(track["latitude"])
        & np.isfinite(track["timestamp"])
    ].copy()
    track = track[
        track["longitude"].between(-180, 180)
        & track["latitude"].between(-90, 90)
    ].copy()
    track = track.drop_duplicates().sort_values("timestamp").reset_index(drop=True)

    nodes = nodes.dropna().copy()
    nodes = nodes[
        np.isfinite(nodes["longitude"])
        & np.isfinite(nodes["latitude"])
        & nodes["longitude"].between(-180, 180)
        & nodes["latitude"].between(-90, 90)
    ].copy()
    valid_node_ids = set(nodes["node_id"])

    arcs = arcs.dropna().copy()
    arcs = arcs[
        arcs["from_node"].isin(valid_node_ids)
        & arcs["to_node"].isin(valid_node_ids)
    ].copy()
    valid_arc_ids = set(arcs["arc_id"])

    route = route.dropna().copy()
    route = route[route["arc_id"].isin(valid_arc_ids)].copy()

    return track, nodes, arcs, route


def build_student_test_dataset(k: int = 5) -> pd.DataFrame:
    """
    Build `results/pathsnap_student_test.csv` from `Student folder/student_test`
    using the EXACT same Part 1 candidate generation and 5 features as
    `preprocessing/prepare_dataset.py` (plus arc_id/from_node/to_node metadata
    for Route Continuity evaluation).
    """
    test_cache = RESULTS_DIR / "pathsnap_student_test.csv"
    if test_cache.exists():
        return pd.read_csv(test_cache, dtype={"trip_id": str})

    student_test_dir = DATASET_ROOT / "student_test"
    if not student_test_dir.exists():
        raise FileNotFoundError(f"student_test folder not found: {student_test_dir}")

    trip_folders = sorted(f for f in student_test_dir.iterdir() if f.is_dir())
    print(f"Building matching student_test table for {len(trip_folders)} trips (k={k})...")

    all_tables = []
    for trip_folder in trip_folders:
        trip_id = trip_folder.name
        track, nodes, arcs, route = _load_and_clean_trip(trip_folder)

        node_lookup = nodes.set_index("node_id")
        start_nodes = node_lookup.loc[arcs["from_node"].to_numpy()]
        end_nodes = node_lookup.loc[arcs["to_node"].to_numpy()]

        mid_lon = (start_nodes["longitude"].to_numpy() + end_nodes["longitude"].to_numpy()) / 2.0
        mid_lat = (start_nodes["latitude"].to_numpy() + end_nodes["latitude"].to_numpy()) / 2.0

        road_coords = np.radians(np.column_stack([mid_lat, mid_lon]))
        tree = BallTree(road_coords, metric="haversine")

        gps_coords = np.radians(track[["latitude", "longitude"]].to_numpy())
        kk = min(k, len(arcs))
        distances, positions = tree.query(gps_coords, k=kk)
        distances = distances * 6371000.0

        arc_ids_arr = arcs["arc_id"].to_numpy()[positions]
        from_nodes_arr = arcs["from_node"].to_numpy()[positions]
        to_nodes_arr = arcs["to_node"].to_numpy()[positions]

        candidates = pd.DataFrame({
            "trip_id": trip_id,
            "observation_id": np.repeat(np.arange(len(track)), kk),
            "candidate_rank": np.tile(np.arange(1, kk + 1), len(track)),
            "arc_id": arc_ids_arr.ravel(),
            "from_node": from_nodes_arr.ravel(),
            "to_node": to_nodes_arr.ravel(),
            "distance_m": distances.ravel(),
        })

        # Extract exact same features as preprocessing/prepare_dataset.py
        lon = track["longitude"].to_numpy()
        lat = track["latitude"].to_numpy()
        ts = track["timestamp"].to_numpy()

        delta_lon = np.diff(lon, prepend=lon[0])
        delta_lat = np.diff(lat, prepend=lat[0])
        delta_time = np.diff(ts, prepend=ts[0])

        gps_dist = np.sqrt(
            (delta_lon * 111320 * np.cos(np.radians(lat))) ** 2
            + (delta_lat * 110540) ** 2
        )
        gps_speed = np.divide(
            gps_dist, delta_time, out=np.zeros_like(gps_dist, dtype=float), where=delta_time > 0
        )
        gps_bearing = (np.degrees(np.arctan2(delta_lon, delta_lat)) + 360.0) % 360.0

        arc_geom = pd.DataFrame({
            "arc_id": arcs["arc_id"].to_numpy(),
            "start_lon": start_nodes["longitude"].to_numpy(),
            "start_lat": start_nodes["latitude"].to_numpy(),
            "end_lon": end_nodes["longitude"].to_numpy(),
            "end_lat": end_nodes["latitude"].to_numpy(),
        }).set_index("arc_id")

        cg = arc_geom.loc[candidates["arc_id"].to_numpy()]
        r_dlon = cg["end_lon"].to_numpy() - cg["start_lon"].to_numpy()
        r_dlat = cg["end_lat"].to_numpy() - cg["start_lat"].to_numpy()
        road_bearing = (np.degrees(np.arctan2(r_dlon, r_dlat)) + 360.0) % 360.0

        obs_idx = candidates["observation_id"].to_numpy()
        dir_diff = np.abs(road_bearing - gps_bearing[obs_idx])
        dir_diff = np.minimum(dir_diff, 360.0 - dir_diff)

        candidates["direction_difference"] = dir_diff
        candidates["gps_speed_mps"] = gps_speed[obs_idx]
        candidates["time_gap_s"] = delta_time[obs_idx]

        route_arc_ids = set(route["arc_id"].tolist())
        candidates["label"] = candidates["arc_id"].isin(route_arc_ids).astype(int)

        candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna()
        all_tables.append(candidates)

    test_df = pd.concat(all_tables, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_cache, index=False)
    print(f"Saved matching student_test dataset ({len(test_df):,} rows) -> {test_cache}")
    return test_df


def load_train_and_test_datasets(n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load teammate's Part 1 training dataset (`results/pathsnap_ml_dataset.csv`)
    and attach deterministic 5-fold GroupKFold split by `trip_id`, plus load/build
    the matching `student_test` dataset.
    """
    train_csv = RESULTS_DIR / "pathsnap_ml_dataset.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Part 1 dataset not found: {train_csv}\n"
            "Run `python preprocessing/prepare_dataset.py` first!"
        )

    print(f"Loading Part 1 training dataset from {train_csv} ...")
    train_df = pd.read_csv(train_csv, dtype={"trip_id": str})

    # Assign 5-fold GroupKFold by trip_id (zero trip leakage across folds!)
    gkf = GroupKFold(n_splits=n_splits)
    folds = np.full(len(train_df), -1, dtype=np.int8)
    for f_idx, (_, val_idx) in enumerate(
        gkf.split(train_df, train_df["label"], groups=train_df["trip_id"])
    ):
        folds[val_idx] = f_idx
    train_df["cv_fold"] = folds

    test_df = build_student_test_dataset(k=5)
    return train_df, test_df


def evaluate_predictions(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """
    Compute candidate-level AND observation-level metrics consistently across all models:
      - Macro F1, Accuracy, Positive Precision, Positive Recall, Confusion Matrix, Log Loss
      - Observation-level Top-1 Accuracy, Candidate Miss Rate
      - Route Continuity (when arc_id, from_node, to_node are available)
    """
    y_true = df["label"].to_numpy(dtype=np.int8)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "candidate_accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "pos_f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 5),
        "pos_precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 5),
        "pos_recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 5),
        "confusion_matrix": cm.tolist(),
    }

    if y_proba is not None:
        p = np.clip(y_proba, 1e-7, 1.0 - 1e-7)
        metrics["log_loss"] = round(float(log_loss(y_true, p, labels=[0, 1])), 5)
        scores = y_proba
    else:
        metrics["log_loss"] = None
        scores = y_pred.astype(np.float64)

    # Observation-level Top-1 Accuracy & Candidate Miss Rate
    keep_cols = ["trip_id", "observation_id", "label"]
    has_graph = all(c in df.columns for c in ["arc_id", "from_node", "to_node"])
    if has_graph:
        keep_cols += ["arc_id", "from_node", "to_node"]

    work = df[keep_cols].copy()
    work["score"] = scores

    idx_top1 = work.groupby(["trip_id", "observation_id"], sort=False)["score"].idxmax()
    top1 = work.loc[idx_top1].sort_values(["trip_id", "observation_id"]).reset_index(drop=True)

    obs_has_pos = work.groupby(["trip_id", "observation_id"], sort=False)["label"].max()
    n_obs = len(top1)
    n_covered = int(obs_has_pos.sum())
    miss_rate = 1.0 - (n_covered / max(n_obs, 1))

    top1_correct = top1["label"].to_numpy() == 1
    metrics["n_observations"] = int(n_obs)
    metrics["top1_accuracy"] = round(float(np.mean(top1_correct)), 5)
    metrics["top1_accuracy_covered_only"] = round(float(np.sum(top1_correct) / max(n_covered, 1)), 5)
    metrics["candidate_miss_rate"] = round(float(miss_rate), 5)

    if has_graph:
        same_trip = top1["trip_id"].to_numpy()[1:] == top1["trip_id"].to_numpy()[:-1]
        same_arc = top1["arc_id"].to_numpy()[1:] == top1["arc_id"].to_numpy()[:-1]
        connected = top1["to_node"].to_numpy()[:-1] == top1["from_node"].to_numpy()[1:]
        continuous = (same_arc | connected) & same_trip
        metrics["route_continuity"] = round(float(np.sum(continuous) / max(int(np.sum(same_trip)), 1)), 5)
    else:
        metrics["route_continuity"] = None

    return metrics

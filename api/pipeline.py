"""
Live Map-Matching Inference, HMM/Viterbi Decoding, and Server-Side Evaluation Pipeline.

Strict Evaluation Integrity:
  - Prediction pipeline reads ONLY .track, .nodes, and .arcs (never .route).
  - Ground-truth .route files remain strictly server-side and are loaded ONLY
    AFTER route reconstruction completes, solely to compute evaluation accuracy.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from api.model_loader import ModelRegistry, _find_trip_file


def _load_and_clean_inference_inputs(trip_folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and clean .track, .nodes, .arcs ONLY (no .route access). Handles tabs or spaces."""
    track_file = _find_trip_file(trip_folder, "track")
    nodes_file = _find_trip_file(trip_folder, "nodes")
    arcs_file = _find_trip_file(trip_folder, "arcs")
    if not (track_file and nodes_file and arcs_file):
        raise FileNotFoundError(f"Missing .track/.nodes/.arcs in {trip_folder}")

    track = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"],
    )
    nodes = pd.read_csv(
        nodes_file,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"],
    )
    arcs = pd.read_csv(
        arcs_file,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
    )

    nodes["node_id"] = np.arange(len(nodes))
    arcs["arc_id"] = np.arange(len(arcs))

    # Clean track
    track = track.dropna().copy()
    track = track[
        np.isfinite(track["longitude"])
        & np.isfinite(track["latitude"])
        & np.isfinite(track["timestamp"])
        & track["longitude"].between(-180, 180)
        & track["latitude"].between(-90, 90)
    ].copy()
    track = track.drop_duplicates().sort_values("timestamp").reset_index(drop=True)

    # Clean nodes & arcs while preserving original IDs
    nodes = nodes.dropna().copy()
    nodes = nodes[
        np.isfinite(nodes["longitude"])
        & np.isfinite(nodes["latitude"])
        & nodes["longitude"].between(-180, 180)
        & nodes["latitude"].between(-90, 90)
    ].copy()
    valid_nodes = set(nodes["node_id"])

    arcs = arcs.dropna().copy()
    arcs = arcs[
        arcs["from_node"].isin(valid_nodes) & arcs["to_node"].isin(valid_nodes)
    ].copy()

    return track, nodes, arcs


def _generate_candidates_and_features(
    track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame, k: int = 5
) -> tuple[pd.DataFrame, np.ndarray]:
    """Exact replica of Part 1 BallTree(haversine) candidate generation + 5 features."""
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

    n_obs = len(track)
    candidates = pd.DataFrame({
        "observation_id": np.repeat(np.arange(n_obs), kk),
        "candidate_rank": np.tile(np.arange(1, kk + 1), n_obs),
        "arc_id": arc_ids_arr.ravel(),
        "from_node": from_nodes_arr.ravel(),
        "to_node": to_nodes_arr.ravel(),
        "distance_m": distances.ravel(),
    })

    # GPS kinematics
    lon = track["longitude"].to_numpy()
    lat = track["latitude"].to_numpy()
    ts = track["timestamp"].to_numpy()

    delta_lon = np.diff(lon, prepend=lon[0])
    delta_lat = np.diff(lat, prepend=lat[0])
    delta_time = np.diff(ts, prepend=ts[0])

    gps_dist = np.sqrt(
        (delta_lon * 111320.0 * np.cos(np.radians(lat))) ** 2
        + (delta_lat * 110540.0) ** 2
    )
    gps_speed = np.divide(
        gps_dist, delta_time, out=np.zeros_like(gps_dist, dtype=float), where=delta_time > 0
    )
    gps_bearing = (np.degrees(np.arctan2(delta_lon, delta_lat)) + 360.0) % 360.0

    # Road bearing
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

    return candidates, delta_time


def _viterbi_decode(
    candidates: pd.DataFrame,
    delta_time: np.ndarray,
    arcs: pd.DataFrame,
    hmm_cfg: dict[str, float],
    k: int = 5,
) -> np.ndarray:
    """
    HMM + Viterbi sequential route reconstruction over candidate road arcs.
    Uses classifier emission log-probabilities and directed road-graph transitions.
    """
    n_obs = len(delta_time)
    if n_obs == 0:
        return np.array([], dtype=np.int64)

    # Reshape candidate arrays to (T, k)
    cand_arcs = candidates["arc_id"].to_numpy(dtype=np.int64).reshape(n_obs, k)
    cand_from = candidates["from_node"].to_numpy(dtype=np.int64).reshape(n_obs, k)
    cand_to = candidates["to_node"].to_numpy(dtype=np.int64).reshape(n_obs, k)
    probs = np.clip(candidates["pred_proba"].to_numpy(dtype=np.float64).reshape(n_obs, k), 1e-4, 1.0 - 1e-4)
    emit_log = np.log(probs)

    # Build directed adjacency map for fast 2-hop connectivity checks
    out_neighbors: dict[int, set[int]] = defaultdict(set)
    for u_node, v_node in zip(arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()):
        out_neighbors[int(u_node)].add(int(v_node))

    p_stay = float(hmm_cfg["stay_log_prob"])
    p_1hop = float(hmm_cfg["connected_1hop_log_prob"])
    p_2hop = float(hmm_cfg["connected_2hop_log_prob"])
    p_uturn = float(hmm_cfg["reverse_uturn_log_prob"])
    p_gap = float(hmm_cfg["gap_transition_log_prob"])
    p_disc = float(hmm_cfg["disconnected_log_prob"])
    gap_thr = float(hmm_cfg["gps_gap_threshold_s"])

    dp = np.full((n_obs, k), -np.inf, dtype=np.float64)
    backptr = np.zeros((n_obs, k), dtype=np.int32)
    dp[0] = emit_log[0]

    for t in range(1, n_obs):
        is_gap = delta_time[t] > gap_thr
        prev_arcs = cand_arcs[t - 1]
        prev_from = cand_from[t - 1]
        prev_to = cand_to[t - 1]
        curr_arcs = cand_arcs[t]
        curr_from = cand_from[t]
        curr_to = cand_to[t]

        trans = np.full((k, k), p_gap if is_gap else p_disc, dtype=np.float64)
        for i in range(k):
            pa = prev_arcs[i]
            pf = prev_from[i]
            pt = prev_to[i]
            nbrs = out_neighbors.get(int(pt))
            for j in range(k):
                ca = curr_arcs[j]
                cf = curr_from[j]
                ct = curr_to[j]
                if ca == pa:
                    trans[i, j] = p_stay
                elif pt == cf and not (pf == ct):
                    trans[i, j] = p_1hop
                elif pf == ct and pt == cf:
                    trans[i, j] = p_uturn
                elif nbrs is not None and int(cf) in nbrs:
                    trans[i, j] = p_2hop

        scores = dp[t - 1][:, None] + trans
        best_prev = np.argmax(scores, axis=0)
        dp[t] = scores[best_prev, np.arange(k)] + emit_log[t]
        backptr[t] = best_prev

    # Backtrack optimal state sequence
    state_seq = np.empty(n_obs, dtype=np.int32)
    state_seq[-1] = int(np.argmax(dp[-1]))
    for t in range(n_obs - 1, 0, -1):
        state_seq[t - 1] = backptr[t, state_seq[t]]

    return cand_arcs[np.arange(n_obs), state_seq]


def _compute_route_continuity(pred_arcs: np.ndarray, arc_from_map: np.ndarray, arc_to_map: np.ndarray) -> float:
    if len(pred_arcs) <= 1:
        return 1.0
    same = pred_arcs[1:] == pred_arcs[:-1]
    connected = arc_to_map[pred_arcs[:-1]] == arc_from_map[pred_arcs[1:]]
    return float(np.mean(same | connected))


def run_trip_inference_and_evaluate(registry: ModelRegistry, trip_id: str) -> dict[str, Any]:
    """
    Execute the complete server-side prediction pipeline for one trip and evaluate
    against internal server-side `.route` ground truth AFTER inference completes.
    """
    t_total0 = time.perf_counter()
    trip_folder = registry.eval_dir / trip_id
    if not trip_folder.exists():
        raise FileNotFoundError(f"Trip '{trip_id}' not found in {registry.eval_dir}")

    # --- 1. Load & Clean (NO .route loaded here!) ---
    t0 = time.perf_counter()
    track, nodes, arcs = _load_and_clean_inference_inputs(trip_folder)
    t_load_ms = (time.perf_counter() - t0) * 1000.0

    # --- 2. Candidate Generation & Feature Extraction ---
    t0 = time.perf_counter()
    k = min(registry.top_k, len(arcs))
    candidates, delta_time = _generate_candidates_and_features(track, nodes, arcs, k=k)
    t_feat_ms = (time.perf_counter() - t0) * 1000.0

    # --- 3. Classifier Inference ---
    t0 = time.perf_counter()
    X = candidates[registry.feature_columns].to_numpy(dtype=np.float32)
    probs = registry.model.predict_proba(X)[:, 1]
    candidates["pred_proba"] = probs
    t_clf_ms = (time.perf_counter() - t0) * 1000.0

    # Classifier-only top-1 arc sequence
    n_obs = len(track)
    probs_2d = probs.reshape(n_obs, k)
    arcs_2d = candidates["arc_id"].to_numpy(dtype=np.int64).reshape(n_obs, k)
    clf_only_arcs = arcs_2d[np.arange(n_obs), np.argmax(probs_2d, axis=1)]

    # --- 4. HMM + Viterbi Route Reconstruction ---
    t0 = time.perf_counter()
    hmm_arcs = _viterbi_decode(candidates, delta_time, arcs, registry.config["hmm"], k=k)
    t_hmm_ms = (time.perf_counter() - t0) * 1000.0

    # Continuity maps (indexed by arc_id)
    max_arc_id = int(arcs["arc_id"].max()) + 1
    arc_from_map = np.full(max_arc_id, -1, dtype=np.int64)
    arc_to_map = np.full(max_arc_id, -1, dtype=np.int64)
    arc_from_map[arcs["arc_id"].to_numpy()] = arcs["from_node"].to_numpy()
    arc_to_map[arcs["arc_id"].to_numpy()] = arcs["to_node"].to_numpy()

    clf_continuity = _compute_route_continuity(clf_only_arcs, arc_from_map, arc_to_map)
    hmm_continuity = _compute_route_continuity(hmm_arcs, arc_from_map, arc_to_map)

    # --- 5. Server-Side Evaluation (Strictly AFTER prediction completes) ---
    route_file = _find_trip_file(trip_folder, "route")
    eval_metrics: dict[str, Any] = {
        "ground_truth_available": False,
        "classifier_only_top1_accuracy": None,
        "hmm_top1_accuracy": None,
        "classifier_only_route_continuity": round(clf_continuity, 5),
        "hmm_route_continuity": round(hmm_continuity, 5),
        "route_arc_precision": None,
        "route_arc_recall": None,
        "route_arc_f1": None,
        "candidate_miss_rate": None,
    }

    if route_file is not None and route_file.exists():
        route_df = pd.read_csv(route_file, sep=r"\s+", header=None, names=["arc_id"]).dropna()
        ref_arc_set = set(route_df["arc_id"].astype(int).tolist())

        # Observation-level Top-1 Accuracy
        clf_hits = np.isin(clf_only_arcs, list(ref_arc_set))
        hmm_hits = np.isin(hmm_arcs, list(ref_arc_set))

        # Candidate miss rate
        cand_has_ref = np.any(np.isin(arcs_2d, list(ref_arc_set)), axis=1)
        miss_rate = float(1.0 - np.mean(cand_has_ref))

        # Unique traversed route arc F1
        pred_unique = set(int(a) for a in hmm_arcs)
        tp_arcs = len(pred_unique & ref_arc_set)
        arc_prec = tp_arcs / max(len(pred_unique), 1)
        arc_rec = tp_arcs / max(len(ref_arc_set), 1)
        arc_f1 = (2.0 * arc_prec * arc_rec / (arc_prec + arc_rec)) if (arc_prec + arc_rec) > 0 else 0.0

        eval_metrics.update({
            "ground_truth_available": True,
            "classifier_only_top1_accuracy": round(float(np.mean(clf_hits)), 5),
            "hmm_top1_accuracy": round(float(np.mean(hmm_hits)), 5),
            "route_arc_precision": round(float(arc_prec), 5),
            "route_arc_recall": round(float(arc_rec), 5),
            "route_arc_f1": round(float(arc_f1), 5),
            "candidate_miss_rate": round(miss_rate, 5),
        })

    # --- 6. Build Lightweight Geometry Payload for UI Rendering (NO .route exposed!) ---
    node_lon = nodes.set_index("node_id")["longitude"]
    node_lat = nodes.set_index("node_id")["latitude"]

    # Deduplicate consecutive identical predicted arcs for crisp polyline rendering
    dedup_arcs = [int(hmm_arcs[0])] if len(hmm_arcs) > 0 else []
    for a in hmm_arcs[1:]:
        ai = int(a)
        if ai != dedup_arcs[-1]:
            dedup_arcs.append(ai)

    predicted_segments = []
    for aid in dedup_arcs:
        u_id = int(arc_from_map[aid])
        v_id = int(arc_to_map[aid])
        predicted_segments.append([
            [round(float(node_lat.loc[u_id]), 6), round(float(node_lon.loc[u_id]), 6)],
            [round(float(node_lat.loc[v_id]), 6), round(float(node_lon.loc[v_id]), 6)],
        ])

    # Subsample raw GPS points if > 600 for smooth browser rendering
    step = max(1, len(track) // 600)
    gps_points = [
        [round(float(track.iloc[i]["latitude"]), 6), round(float(track.iloc[i]["longitude"]), 6)]
        for i in range(0, len(track), step)
    ]

    total_ms = (time.perf_counter() - t_total0) * 1000.0

    return {
        "trip_id": trip_id,
        "partition": registry.partition_name,
        "model_info": {
            "model_name": registry.model_name,
            "algorithm_family": registry.algorithm_family,
            "model_artifact": registry.model_path.name,
            "decoder": "HMM + Viterbi",
        },
        "trip_summary": {
            "n_gps_points": int(len(track)),
            "n_road_nodes": int(len(nodes)),
            "n_road_arcs": int(len(arcs)),
            "n_reconstructed_segments": int(len(dedup_arcs)),
        },
        "metrics": eval_metrics,
        "timing_ms": {
            "load_and_clean_ms": round(t_load_ms, 2),
            "candidate_and_features_ms": round(t_feat_ms, 2),
            "classifier_inference_ms": round(t_clf_ms, 2),
            "hmm_viterbi_ms": round(t_hmm_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        "map_data": {
            "center": [round(float(track["latitude"].mean()), 6), round(float(track["longitude"].mean()), 6)],
            "bounds": [
                [round(float(track["latitude"].min()), 6), round(float(track["longitude"].min()), 6)],
                [round(float(track["latitude"].max()), 6), round(float(track["longitude"].max()), 6)],
            ],
            "gps_points": gps_points,
            "predicted_route_segments": predicted_segments,
        },
    }

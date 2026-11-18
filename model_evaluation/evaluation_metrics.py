"""
Model Evaluation Metrics Module (Team-01 — PathSnap).

Computes candidate-level and observation/route-level metrics consistently
across all five models:
  - Candidate Accuracy, Macro F1-score, Positive Precision/Recall/F1
  - Confusion Matrix & Binary Log Loss
  - Observation-level Top-1 Accuracy & Candidate Miss Rate
  - Route Continuity (consecutive road-graph connectivity)
"""

from __future__ import annotations

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


def evaluate_predictions(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """Evaluate candidate and observation/route predictions consistently."""
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

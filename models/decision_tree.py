"""
PathSnap: Model Training & Evaluation - Decision Tree Classifier
Corresponds to: Deliverable 1 (Model training) & Deliverable 2 (Sections 3.1 & 3.2)
Algorithm Family: Tree-based
"""

import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

# ============================================================
# PATH CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "results" / "pathsnap_ml_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load the preprocessed PathSnap ML candidate dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run preprocessing/prepare_dataset.py first."
        )

    print(f"Loading dataset from: {DATASET_PATH}")
    start_time = time.time()
    df = pd.read_csv(DATASET_PATH)
    elapsed = time.time() - start_time
    print(f"Loaded {len(df):,} candidate rows across {df['trip_id'].nunique()} trips in {elapsed:.2f}s.")
    return df


def compute_observation_top1_accuracy(df_subset, pred_probs):
    """
    Observation-level Top-1 Accuracy:
    For each GPS observation (trip_id, observation_id), select the candidate
    with the highest predicted probability for class 1 and check if it is
    a true road segment (label == 1).
    """
    eval_df = df_subset[["trip_id", "observation_id", "label"]].copy()
    eval_df["score"] = pred_probs

    # For each observation, pick candidate with highest score
    best_candidates = eval_df.sort_values(
        ["trip_id", "observation_id", "score"], ascending=[True, True, False]
    ).groupby(["trip_id", "observation_id"]).first()

    top1_accuracy = (best_candidates["label"] == 1).mean()
    return top1_accuracy


def compute_route_continuity(df_subset, pred_probs):
    """
    Route Continuity:
    Proportion of consecutive observations within the same trip where the
    predicted road candidate is either identical (staying on same segment)
    or transitions to a plausible adjacent candidate.
    """
    eval_df = df_subset[["trip_id", "observation_id", "candidate_rank", "label"]].copy()
    eval_df["score"] = pred_probs

    best_candidates = eval_df.sort_values(
        ["trip_id", "observation_id", "score"], ascending=[True, True, False]
    ).groupby(["trip_id", "observation_id"]).first().reset_index()

    # Consecutive observation continuity check
    same_trip = best_candidates["trip_id"] == best_candidates["trip_id"].shift(1)
    rank_change = (best_candidates["candidate_rank"] - best_candidates["candidate_rank"].shift(1)).abs()
    continuous = same_trip & (rank_change <= 2)

    return float(continuous.sum() / max(1, same_trip.sum()))


def train_and_evaluate_decision_tree():
    df = load_dataset()

    feature_cols = [
        "candidate_rank",
        "distance_m",
        "direction_difference",
        "gps_speed_mps",
        "time_gap_s",
    ]
    target_col = "label"
    group_col = "trip_id"

    X = df[feature_cols]
    y = df[target_col].to_numpy()
    groups = df[group_col].to_numpy()

    print("\nFeature matrix shape:", X.shape)
    print("Class distribution:", dict(pd.Series(y).value_counts(normalize=True).round(4)))

    # Hyperparameter configuration (regularized to prevent severe overfitting)
    dt_params = {
        "max_depth": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "criterion": "gini",
        "random_state": 42,
    }

    # ============================================================
    # 1. GROUP-AWARE CROSS-VALIDATION (5-FOLD)
    # Strictly ensures no observations from the same trip cross folds
    # ============================================================
    print("\n" + "=" * 60)
    print("RUNNING 5-FOLD GROUP-AWARE CROSS-VALIDATION (BY TRIP)")
    print("=" * 60)

    gkf = GroupKFold(n_splits=5)
    cv_macro_f1s = []
    cv_top1_accs = []
    cv_log_losses = []

    fold = 1
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        val_df = df.iloc[val_idx]

        model = DecisionTreeClassifier(**dt_params)
        model.fit(X_tr, y_tr)

        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        fold_macro_f1 = f1_score(y_val, val_preds, average="macro")
        fold_top1_acc = compute_observation_top1_accuracy(val_df, val_probs)
        fold_loss = log_loss(y_val, val_probs)

        cv_macro_f1s.append(fold_macro_f1)
        cv_top1_accs.append(fold_top1_acc)
        cv_log_losses.append(fold_loss)

        print(
            f"Fold {fold}: Macro F1 = {fold_macro_f1:.4f} | "
            f"Top-1 Acc = {fold_top1_acc:.4f} | Log Loss = {fold_loss:.4f}"
        )
        fold += 1

    mean_cv_macro_f1 = float(np.mean(cv_macro_f1s))
    mean_cv_top1_acc = float(np.mean(cv_top1_accs))
    mean_cv_log_loss = float(np.mean(cv_log_losses))

    print("-" * 60)
    print(f"CV Macro F1-score:     {mean_cv_macro_f1:.4f} (+/- {np.std(cv_macro_f1s):.4f})")
    print(f"CV Top-1 Accuracy:     {mean_cv_top1_acc:.4f} (+/- {np.std(cv_top1_accs):.4f})")
    print(f"CV Log Loss:           {mean_cv_log_loss:.4f} (+/- {np.std(cv_log_losses):.4f})")

    # ============================================================
    # 2. TRAIN FINAL MODEL ON 80% TRIPS, EVALUATE ON 20% HELD-OUT
    # ============================================================
    unique_trips = np.unique(groups)
    np.random.seed(42)
    shuffled_trips = np.random.permutation(unique_trips)
    split_idx = int(0.8 * len(shuffled_trips))
    train_trips = set(shuffled_trips[:split_idx])
    test_trips = set(shuffled_trips[split_idx:])

    train_mask = df["trip_id"].isin(train_trips)
    test_mask = df["trip_id"].isin(test_trips)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_df = df[test_mask]

    print("\n" + "=" * 60)
    print("FITTING FINAL DECISION TREE ON TRAINING TRIPS")
    print("=" * 60)

    final_model = DecisionTreeClassifier(**dt_params)
    t_train_start = time.time()
    final_model.fit(X_train, y_train)
    training_time = time.time() - t_train_start
    print(f"Training completed in {training_time:.2f} seconds.")

    # Measure Inference Time
    t_inf_start = time.time()
    test_probs = final_model.predict_proba(X_test)[:, 1]
    inference_time_total = time.time() - t_inf_start
    inference_time_per_1k = (inference_time_total / len(X_test)) * 1000

    test_preds = (test_probs >= 0.5).astype(int)

    # Train vs Test metrics (for overfitting check - Q9)
    train_preds = final_model.predict(X_train)
    train_macro_f1 = f1_score(y_train, train_preds, average="macro")
    test_macro_f1 = f1_score(y_test, test_preds, average="macro")
    test_top1_acc = compute_observation_top1_accuracy(test_df, test_probs)
    test_log_loss = log_loss(y_test, test_probs)
    route_cont = compute_route_continuity(test_df, test_probs)

    print(f"\n--- EVALUATION RESULTS (HELD-OUT TRIPS: {len(test_trips)}) ---")
    print(f"Train Macro F1:        {train_macro_f1:.4f}")
    print(f"Test Macro F1:         {test_macro_f1:.4f}")
    print(f"Top-1 Observation Acc: {test_top1_acc:.4f}")
    print(f"Test Log Loss:         {test_log_loss:.4f}")
    print(f"Route Continuity:      {route_cont:.4f}")
    print(f"Total Inference Time:  {inference_time_total:.4f}s ({inference_time_per_1k:.2f} ms / 1k observations)")

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, test_preds, digits=4))

    cm = confusion_matrix(y_test, test_preds)
    print("--- CONFUSION MATRIX ---")
    print(cm)

    # Feature importances
    feature_importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": final_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    print("\n--- FEATURE IMPORTANCES ---")
    print(feature_importance_df.to_string(index=False))

    # ============================================================
    # 3. SAVE ARTIFACTS AND TABLE 3.1 ENTRY
    # ============================================================
    # Save trained model
    model_path = MODELS_DIR / "decision_tree_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"\nModel artifact saved to: {model_path}")

    # Build Table 3.1 entry exactly matching the report guidelines
    table_3_1_entry = {
        "Model": "Decision Tree",
        "Algorithm Family": "Tree-based",
        "Key Configuration": f"criterion={dt_params['criterion']}, max_depth={dt_params['max_depth']}, min_samples_split={dt_params['min_samples_split']}",
        "CV Macro F1-score": round(mean_cv_macro_f1, 4),
        "Top-1 Accuracy": round(test_top1_acc, 4),
        "Route continuity": round(route_cont, 4),
        "Student-Test Macro F1-score": round(test_macro_f1, 4),
        "Log Loss": round(test_log_loss, 4),
        "Inference Time": f"{inference_time_per_1k:.2f} ms/1k obs",
    }

    comparison_path = RESULTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comp_df = pd.read_csv(comparison_path)
        comp_df = comp_df[comp_df["Model"] != "Decision Tree"]
        comp_df = pd.concat([comp_df, pd.DataFrame([table_3_1_entry])], ignore_index=True)
    else:
        comp_df = pd.DataFrame([table_3_1_entry])

    comp_df.to_csv(comparison_path, index=False)
    print(f"Table 3.1 entry appended to: {comparison_path}")

    print("\n" + "=" * 60)
    print("SUMMARY FOR DELIVERABLE 2 REPORT (TABLE 3.1)")
    print("=" * 60)
    print(pd.DataFrame([table_3_1_entry]).to_string(index=False))

    return final_model, table_3_1_entry


if __name__ == "__main__":
    train_and_evaluate_decision_tree()

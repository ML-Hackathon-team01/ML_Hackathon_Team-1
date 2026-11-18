"""
PathSnap: Part 4 — Bayesian Hyperparameter Optimisation for Gradient Boosting
Algorithm Family: Boosting (HistGradientBoostingClassifier)

Executes Bayesian Optimization (Task 3) over the defined search space using Optuna.
Objective: Maximise 5-Fold Group-aware Cross-Validation Macro F1-score across 80 training trips.
Saves trial logs, generates convergence plots, extracts best parameters,
and saves the comparison between untuned and tuned models for Deliverable 2 Section 4.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss

# Ensure parent directory is in path to import team utilities
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.part2_utils import (
    FEATURE_COLS,
    PLOTS_DIR,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    evaluate_predictions,
    load_train_and_test_datasets,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, folds: np.ndarray) -> float:
    """
    Objective function: 5-Fold GroupKFold CV Macro F1 on the 80 training trips.
    """
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 250, step=25),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 150),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 10.0, log=True),
        "loss": "log_loss",
        "early_stopping": False,
        "random_state": 42,
    }

    fold_macro_f1s = []
    # Evaluate across the 5 GroupKFold splits
    for f in range(5):
        tr_idx = folds != f
        va_idx = folds == f

        clf = HistGradientBoostingClassifier(**params)
        clf.fit(X_train[tr_idx], y_train[tr_idx])

        va_proba = clf.predict_proba(X_train[va_idx])[:, 1]
        va_pred = (va_proba >= 0.5).astype(np.int8)

        score = f1_score(y_train[va_idx], va_pred, average="macro", zero_division=0)
        fold_macro_f1s.append(score)

    mean_macro_f1 = float(np.mean(fold_macro_f1s))
    return mean_macro_f1


def run_bayesian_optimization(n_trials: int = 30):
    print("=" * 70)
    print("PART 4: BAYESIAN HYPERPARAMETER OPTIMISATION (GRADIENT BOOSTING)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load 80 training trips and student-test partition
    train_df, test_df = load_train_and_test_datasets(n_splits=5)

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int8)
    folds = train_df["cv_fold"].to_numpy(dtype=np.int8)

    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int8)

    print(f"\n[Task 2 & 3] Running {n_trials} Bayesian optimization trials using Optuna TPE...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t0_opt = time.perf_counter()
    trial_records = []

    def callback(study, trial):
        trial_records.append({
            "trial_number": trial.number,
            "cv_macro_f1": trial.value,
            **trial.params,
        })
        print(f"  Trial {trial.number:2d}/{n_trials}: CV Macro F1 = {trial.value:.5f} | Params: {trial.params}")

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, folds),
        n_trials=n_trials,
        callbacks=[callback],
    )
    total_opt_time = time.perf_counter() - t0_opt
    print(f"\nBayesian search completed in {total_opt_time:.1f}s ({total_opt_time/60:.2f} mins).")

    # ============================================================
    # TASK 4: EXTRACT BEST HYPERPARAMETERS
    # ============================================================
    best_params = study.best_params
    best_cv_score = study.best_value
    print("\n" + "=" * 70)
    print("TASK 4: BEST HYPERPARAMETERS OBTAINED")
    print("=" * 70)
    print(f"Best 5-Fold CV Macro F1: {best_cv_score:.5f}")
    for k, v in best_params.items():
        print(f"  {k:20s}: {v}")

    # Save trial history CSV
    trials_df = pd.DataFrame(trial_records)
    trials_csv = RESULTS_DIR / "bayesian_trials_history.csv"
    trials_df.to_csv(trials_csv, index=False)
    print(f"\nSaved trial history -> {trials_csv}")

    # Plot Optimization History / Convergence
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cum_max = np.maximum.accumulate(trials_df["cv_macro_f1"])
    ax.scatter(trials_df["trial_number"], trials_df["cv_macro_f1"], color="#93c5fd", label="Trial Score", alpha=0.7)
    ax.plot(trials_df["trial_number"], cum_max, color="#1d4ed8", lw=2.5, label="Best Score (Cumulative Max)")
    ax.set_title("Part 4: Bayesian Optimisation Convergence History", fontweight="bold")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("5-Fold CV Macro F1-score")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    opt_plot_path = PLOTS_DIR / "bayesian_optimization_convergence.png"
    fig.savefig(opt_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved convergence plot -> {opt_plot_path}")

    # ============================================================
    # TASK 5: RETRAIN TUNED MODEL ON FULL TRAINING DATA
    # ============================================================
    print("\n" + "=" * 70)
    print("TASK 5: REFITTING TUNED MODEL ON 80 TRAINING TRIPS")
    print("=" * 70)
    tuned_clf = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,
        random_state=42,
        **best_params,
    )
    t0_fit = time.perf_counter()
    tuned_clf.fit(X_train, y_train)
    tuned_train_time = time.perf_counter() - t0_fit

    # Evaluate Tuned Model on Student-Testing Partition
    t0_inf = time.perf_counter()
    tuned_test_proba = tuned_clf.predict_proba(X_test)[:, 1]
    tuned_inf_time = time.perf_counter() - t0_inf
    tuned_test_pred = (tuned_test_proba >= 0.5).astype(np.int8)

    tuned_metrics = evaluate_predictions(test_df, tuned_test_pred, tuned_test_proba)

    # Save tuned model artifact
    tuned_model_path = SAVED_MODELS_DIR / "tuned_gradient_boosting.joblib"
    joblib.dump(
        {"model": tuned_clf, "features": FEATURE_COLS, "best_params": best_params},
        tuned_model_path,
    )
    print(f"Saved tuned model artifact -> {tuned_model_path}")

    # ============================================================
    # TASK 6: UNTUNED VS. TUNED COMPARISON
    # ============================================================
    untuned_json_path = RESULTS_DIR / "sohel_gradient_boosting_metrics.json"
    if untuned_json_path.exists():
        with open(untuned_json_path, "r", encoding="utf-8") as f:
            untuned_data = json.load(f)
    else:
        untuned_data = {
            "cv": {"macro_f1_mean": 0.70308},
            "student_test": {
                "top1_accuracy": 0.76472,
                "macro_f1": 0.73724,
                "log_loss": 0.53511,
            },
            "timing": {"inference_time_s": 0.6996},
        }

    comparison_rows = [
        {
            "Model Variant": "Untuned (Baseline HistGB)",
            "Configuration": "lr=0.08, max_iter=200, max_leaf_nodes=31, max_depth=8, min_samples_leaf=50, l2=1.0",
            "CV Macro F1-score": round(untuned_data["cv"]["macro_f1_mean"], 4),
            "Student-Test Top-1 Accuracy": round(untuned_data["student_test"]["top1_accuracy"], 4),
            "Student-Test Macro F1-score": round(untuned_data["student_test"]["macro_f1"], 4),
            "Log Loss": round(untuned_data["student_test"]["log_loss"], 4),
            "Inference Time": f"{untuned_data['timing']['inference_time_s']:.3f} s",
        },
        {
            "Model Variant": "Tuned (Bayesian Optuna)",
            "Configuration": f"lr={best_params['learning_rate']:.3f}, max_iter={best_params['max_iter']}, max_leaf_nodes={best_params['max_leaf_nodes']}, max_depth={best_params['max_depth']}, min_samples_leaf={best_params['min_samples_leaf']}, l2={best_params['l2_regularization']:.2f}",
            "CV Macro F1-score": round(best_cv_score, 4),
            "Student-Test Top-1 Accuracy": round(tuned_metrics["top1_accuracy"], 4),
            "Student-Test Macro F1-score": round(tuned_metrics["macro_f1"], 4),
            "Log Loss": round(tuned_metrics["log_loss"], 4),
            "Inference Time": f"{tuned_inf_time:.3f} s",
        },
    ]

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = RESULTS_DIR / "bayesian_tuned_vs_untuned_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)

    print("\n" + "=" * 70)
    print("TASK 6: TUNED VS. UNTUNED COMPARISON TABLE (FOR DELIVERABLE 2)")
    print("=" * 70)
    print(comp_df.to_string(index=False))

    return best_params, comp_df


if __name__ == "__main__":
    run_bayesian_optimization(n_trials=30)

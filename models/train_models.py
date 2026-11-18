"""
Part 2 — Model Selection & Training (All 5 Models in Required Order).

Consumes the team's Part 1 dataset (`results/pathsnap_ml_dataset.csv` from
`preprocessing/prepare_dataset.py`) + matching `student_test` set and trains,
cross-validates (5-fold GroupKFold by trip_id), evaluates, and compares all 5
assigned models:

  1. Logistic Regression   (Navya)   — Simple linear baseline
  2. Decision Tree         (Neal)    — Nonlinear decision rules
  3. Random Forest         (Shreyas) — Bagged tree ensemble
  4. SVM (Linear + Platt)  (Navya)   — Margin-based linear boundary
  5. Gradient Boosting     (Sohel)   — Iterative boosting ensemble

Outputs:
  - results/model_comparison_table.csv
  - results/model_comparison_results.json
  - results/plots/part2_model_comparison.png
  - results/plots/part2_confusion_matrices.png
  - results/plots/sohel_gb_learning_curve.png
  - results/plots/sohel_gb_feature_importance.png
  - saved_models/<model>.joblib
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
from models.sohel_gradient_boosting import run_gradient_boosting


def get_other_model_specs() -> list[dict]:
    """Initial untuned configurations for Models 1–4 (Model 5 is Sohel's GB)."""
    return [
        {
            "order": 1,
            "slug": "navya_logistic_regression",
            "model_name": "Logistic Regression",
            "owner": "Navya",
            "algorithm_family": "Linear / Probabilistic",
            "key_configuration": "StandardScaler + LogisticRegression(C=1.0, solver=lbfgs, max_iter=500)",
            "builder": lambda seed: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=seed)),
            ]),
        },
        {
            "order": 2,
            "slug": "neal_decision_tree",
            "model_name": "Decision Tree",
            "owner": "Neal",
            "algorithm_family": "Tree-based",
            "key_configuration": "DecisionTreeClassifier(criterion=gini, max_depth=12, min_samples_leaf=50)",
            "builder": lambda seed: DecisionTreeClassifier(
                criterion="gini", max_depth=12, min_samples_leaf=50, random_state=seed
            ),
        },
        {
            "order": 3,
            "slug": "shreyas_random_forest",
            "model_name": "Random Forest",
            "owner": "Shreyas",
            "algorithm_family": "Ensemble (Bagging)",
            "key_configuration": "RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=30)",
            "builder": lambda seed: RandomForestClassifier(
                n_estimators=100, max_depth=14, min_samples_leaf=30, n_jobs=-1, random_state=seed
            ),
        },
        {
            "order": 4,
            "slug": "navya_svm",
            "model_name": "SVM (Linear + Platt)",
            "owner": "Navya",
            "algorithm_family": "Kernel / Margin-based",
            "key_configuration": "StandardScaler + CalibratedClassifierCV(LinearSVC(C=1.0), method=sigmoid, cv=3)",
            "builder": lambda seed: Pipeline([
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(C=1.0, dual="auto", max_iter=2000, random_state=seed),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]),
        },
    ]


def evaluate_model_spec(spec: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int8)
    folds = train_df["cv_fold"].to_numpy(dtype=np.int8)

    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)

    print(f"\n--- [Model {spec['order']} / {spec['owner']}] {spec['model_name']} ---")
    fold_metrics = []
    for f in range(5):
        tr_idx = folds != f
        va_idx = folds == f

        model = spec["builder"](42 + f)
        t0 = time.perf_counter()
        model.fit(X_train[tr_idx], y_train[tr_idx])
        tr_t = time.perf_counter() - t0

        va_proba = model.predict_proba(X_train[va_idx])[:, 1]
        va_pred = (va_proba >= 0.5).astype(np.int8)
        m = evaluate_predictions(train_df.iloc[va_idx], va_pred, va_proba)
        m["fold"] = f
        m["train_time_s"] = round(tr_t, 3)
        fold_metrics.append(m)
        print(
            f"  Fold {f}: Macro F1={m['macro_f1']:.4f} | Top-1={m['top1_accuracy']:.4f} | "
            f"LogLoss={m['log_loss']:.4f} | {tr_t:.1f}s"
        )

    # Final fit on all 80 training trips
    final_model = spec["builder"](42)
    t0 = time.perf_counter()
    final_model.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0

    # Train-fit diagnostics (overfitting check)
    tr_proba = final_model.predict_proba(X_train)[:, 1]
    tr_pred = (tr_proba >= 0.5).astype(np.int8)
    train_m = evaluate_predictions(train_df, tr_pred, tr_proba)

    # Student-test evaluation (10 trips)
    t_inf0 = time.perf_counter()
    test_proba = final_model.predict_proba(X_test)[:, 1]
    inf_time_s = time.perf_counter() - t_inf0
    test_pred = (test_proba >= 0.5).astype(np.int8)
    test_m = evaluate_predictions(test_df, test_pred, test_proba)

    joblib.dump(
        {"model": final_model, "features": FEATURE_COLS, "owner": spec["owner"]},
        SAVED_MODELS_DIR / f"{spec['slug']}.joblib",
    )

    cv_f1 = [m["macro_f1"] for m in fold_metrics]
    cv_top1 = [m["top1_accuracy"] for m in fold_metrics]
    cv_ll = [m["log_loss"] for m in fold_metrics]

    result = {
        "order": spec["order"],
        "model_name": spec["model_name"],
        "owner": spec["owner"],
        "algorithm_family": spec["algorithm_family"],
        "key_configuration": spec["key_configuration"],
        "cv": {
            "macro_f1_mean": round(float(np.mean(cv_f1)), 5),
            "macro_f1_std": round(float(np.std(cv_f1)), 5),
            "top1_accuracy_mean": round(float(np.mean(cv_top1)), 5),
            "log_loss_mean": round(float(np.mean(cv_ll)), 5),
            "folds": fold_metrics,
        },
        "train_fit": {
            "macro_f1": train_m["macro_f1"],
            "top1_accuracy": train_m["top1_accuracy"],
            "log_loss": train_m["log_loss"],
        },
        "student_test": test_m,
        "timing": {
            "train_time_s": round(train_time_s, 3),
            "inference_time_s": round(inf_time_s, 4),
        },
    }

    with open(RESULTS_DIR / f"{spec['slug']}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"  => [{spec['model_name']}] CV Macro F1={result['cv']['macro_f1_mean']:.4f} +/- {result['cv']['macro_f1_std']:.4f} | "
        f"Test Macro F1={test_m['macro_f1']:.4f} | Top-1={test_m['top1_accuracy']:.4f} | "
        f"Route Continuity={test_m['route_continuity']:.4f} | LogLoss={test_m['log_loss']:.4f} | "
        f"Train={train_time_s:.1f}s | Infer={inf_time_s:.3f}s"
    )
    return result


def plot_comparison_charts(comp_df: pd.DataFrame, all_results: list[dict]) -> None:
    """Create comparison plots (CV Macro F1, Test Metrics, Log Loss) & 5-model confusion matrices."""
    models = comp_df["Model"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.suptitle("PathSnap Part 2 — 5-Model Comparison (Team-01)", fontsize=14, fontweight="bold")

    # 1. CV Macro F1 with error bars
    axes[0].bar(
        models, comp_df["CV Macro F1"], yerr=comp_df["CV Macro F1 Std"],
        color="#2563eb", alpha=0.85, capsize=4,
    )
    axes[0].set_title("5-Fold GroupKFold CV Macro F1")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].grid(axis="y", alpha=0.3)

    # 2. Student-Test Top-1 Accuracy & Macro F1
    x = np.arange(len(models))
    w = 0.36
    axes[1].bar(x - w / 2, comp_df["Top-1 Accuracy"], width=w, label="Top-1 Accuracy", color="#16a34a")
    axes[1].bar(x + w / 2, comp_df["Student-Test Macro F1"], width=w, label="Test Macro F1", color="#f97316")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=22)
    axes[1].set_title("Student-Test Top-1 Accuracy & Macro F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    # 3. Log Loss (lower is better)
    axes[2].bar(models, comp_df["Log Loss"], color="#dc2626", alpha=0.85)
    axes[2].set_title("Student-Test Log Loss (Lower is Better)")
    axes[2].tick_params(axis="x", rotation=22)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOTS_DIR / "part2_model_comparison.png", dpi=160)
    plt.close(fig)

    # 5-Model Confusion Matrix Grid
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    plt.suptitle("Student-Test Confusion Matrices (All 5 Models)", fontsize=13, fontweight="bold")
    for ax, res in zip(axes, all_results):
        cm = np.array(res["student_test"]["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
            xticklabels=["P0", "P1"], yticklabels=["T0", "T1"], ax=ax,
        )
        ax.set_title(f"{res['model_name']}\n({res['owner']})", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(PLOTS_DIR / "part2_confusion_matrices.png", dpi=150)
    plt.close(fig)


def main() -> None:
    t_total = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_train_and_test_datasets(n_splits=5)
    print(f"Loaded train: {len(train_df):,} rows ({train_df['trip_id'].nunique()} trips) | test: {len(test_df):,} rows ({test_df['trip_id'].nunique()} trips)")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    all_results: list[dict] = []
    # Models 1–4 in exact required order
    for spec in get_other_model_specs():
        all_results.append(evaluate_model_spec(spec, train_df, test_df))

    # Model 5: Sohel's Gradient Boosting
    all_results.append(run_gradient_boosting(train_df=train_df, test_df=test_df))

    # Build consolidated Section 3.1 comparison table
    rows = []
    for r in all_results:
        rows.append({
            "Order": r["order"],
            "Model": r["model_name"],
            "Owner": r["owner"],
            "Algorithm Family": r["algorithm_family"],
            "Key Configuration": r["key_configuration"],
            "CV Macro F1": r["cv"]["macro_f1_mean"],
            "CV Macro F1 Std": r["cv"]["macro_f1_std"],
            "Train Macro F1": r["train_fit"]["macro_f1"],
            "Student-Test Macro F1": r["student_test"]["macro_f1"],
            "Top-1 Accuracy": r["student_test"]["top1_accuracy"],
            "Route Continuity": r["student_test"]["route_continuity"],
            "Log Loss": r["student_test"]["log_loss"],
            "Train Time (s)": r["timing"]["train_time_s"],
            "Inference Time (s)": r["timing"]["inference_time_s"],
        })

    comp_df = pd.DataFrame(rows)
    comp_csv = RESULTS_DIR / "model_comparison_table.csv"
    comp_df.to_csv(comp_csv, index=False)

    # Identify best model by CV Macro F1 & Student-Test Top-1 Accuracy
    best_by_cv = comp_df.loc[comp_df["CV Macro F1"].idxmax()]
    best_by_test = comp_df.loc[comp_df["Student-Test Macro F1"].idxmax()]

    summary = {
        "features_used": FEATURE_COLS,
        "cv_strategy": "5-fold GroupKFold grouped by trip_id (80 training trips)",
        "best_model_by_cv_macro_f1": best_by_cv["Model"],
        "best_model_by_student_test_macro_f1": best_by_test["Model"],
        "models": all_results,
    }
    with open(RESULTS_DIR / "model_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_comparison_charts(comp_df, all_results)

    elapsed = time.perf_counter() - t_total
    print(f"\n==================== PART 2 COMPLETE in {elapsed:.1f}s ====================")
    display_cols = [
        "Order", "Model", "Owner", "CV Macro F1", "Train Macro F1",
        "Student-Test Macro F1", "Top-1 Accuracy", "Route Continuity",
        "Log Loss", "Train Time (s)", "Inference Time (s)",
    ]
    print(comp_df[display_cols].to_string(index=False))
    print(f"\nBest model (CV Macro F1)          : {best_by_cv['Model']} ({best_by_cv['CV Macro F1']:.4f})")
    print(f"Best model (Student-Test Macro F1): {best_by_test['Model']} ({best_by_test['Student-Test Macro F1']:.4f})")
    print(f"\nSaved:\n  - {comp_csv}\n  - {RESULTS_DIR / 'model_comparison_results.json'}")
    print(f"  - {PLOTS_DIR / 'part2_model_comparison.png'}\n  - {PLOTS_DIR / 'part2_confusion_matrices.png'}")
    print(f"  - {PLOTS_DIR / 'sohel_gb_learning_curve.png'}\n  - {PLOTS_DIR / 'sohel_gb_feature_importance.png'}")


if __name__ == "__main__":
    main()

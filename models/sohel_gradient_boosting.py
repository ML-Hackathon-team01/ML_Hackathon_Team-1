"""
Part 2 (Sohel's Assigned Model): Standalone Gradient Boosting Script.

Model: HistGradientBoostingClassifier (Boosting family)
Owner: Sohel (Team-01)
Built directly on top of the team's Part 1 dataset (`results/pathsnap_ml_dataset.csv`).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

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


def build_gradient_boosting(random_state: int = 42) -> HistGradientBoostingClassifier:
    """Initial Gradient Boosting hyperparameters for Part 2 experimentation."""
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        max_depth=8,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=random_state,
    )


def run_gradient_boosting(train_df=None, test_df=None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if train_df is None or test_df is None:
        train_df, test_df = load_train_and_test_datasets(n_splits=5)

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int8)
    folds = train_df["cv_fold"].to_numpy(dtype=np.int8)

    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int8)

    print("\n--- [Model 5 / Sohel] Gradient Boosting (HistGradientBoostingClassifier) ---")
    fold_metrics = []
    curve_iters, curve_tr_ll, curve_va_ll = [], [], []

    for f in range(5):
        tr_idx = folds != f
        va_idx = folds == f

        clf = build_gradient_boosting(random_state=42 + f)
        t0 = time.perf_counter()
        clf.fit(X_train[tr_idx], y_train[tr_idx])
        tr_t = time.perf_counter() - t0

        va_proba = clf.predict_proba(X_train[va_idx])[:, 1]
        va_pred = (va_proba >= 0.5).astype(np.int8)
        m = evaluate_predictions(train_df.iloc[va_idx], va_pred, va_proba)
        m["fold"] = f
        m["train_time_s"] = round(tr_t, 3)
        fold_metrics.append(m)
        print(
            f"  Fold {f}: Macro F1={m['macro_f1']:.4f} | Top-1={m['top1_accuracy']:.4f} | "
            f"LogLoss={m['log_loss']:.4f} | {tr_t:.1f}s"
        )

        # True iteration-wise train vs validation log-loss curve on Fold 0
        if f == 0:
            rng = np.random.default_rng(42)
            tr_pool = np.where(tr_idx)[0]
            tr_sub = rng.choice(tr_pool, size=min(80000, len(tr_pool)), replace=False)
            for it, (p_tr, p_va) in enumerate(
                zip(clf.staged_predict_proba(X_train[tr_sub]), clf.staged_predict_proba(X_train[va_idx])), 1
            ):
                if it == 1 or it % 10 == 0 or it == clf.n_iter_:
                    curve_iters.append(it)
                    curve_tr_ll.append(round(float(log_loss(y_train[tr_sub], np.clip(p_tr[:, 1], 1e-7, 1 - 1e-7))), 5))
                    curve_va_ll.append(round(float(log_loss(y_train[va_idx], np.clip(p_va[:, 1], 1e-7, 1 - 1e-7))), 5))

    # Final model fit on all 80 training trips
    final_clf = build_gradient_boosting(random_state=42)
    t0 = time.perf_counter()
    final_clf.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0

    # Train-set check (for overfitting diagnosis)
    tr_proba = final_clf.predict_proba(X_train)[:, 1]
    tr_pred = (tr_proba >= 0.5).astype(np.int8)
    train_m = evaluate_predictions(train_df, tr_pred, tr_proba)

    # Student-test evaluation
    t_inf0 = time.perf_counter()
    test_proba = final_clf.predict_proba(X_test)[:, 1]
    inf_time_s = time.perf_counter() - t_inf0
    test_pred = (test_proba >= 0.5).astype(np.int8)
    test_m = evaluate_predictions(test_df, test_pred, test_proba)

    # Permutation importance on student_test (sampled for speed)
    rng = np.random.default_rng(42)
    imp_idx = rng.choice(len(X_test), size=min(30000, len(X_test)), replace=False)
    perm = permutation_importance(final_clf, X_test[imp_idx], y_test[imp_idx], n_repeats=5, random_state=42, n_jobs=-1)
    feat_imp = {FEATURE_COLS[i]: round(float(perm.importances_mean[i]), 5) for i in np.argsort(perm.importances_mean)[::-1]}

    # Save model artifact
    joblib.dump(
        {"model": final_clf, "features": FEATURE_COLS, "owner": "Sohel"},
        SAVED_MODELS_DIR / "sohel_gradient_boosting.joblib",
    )

    # Save learning curve & feature importance plots
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(curve_iters, curve_tr_ll, label="Train Log Loss", color="#2563eb", lw=2)
    ax.plot(curve_iters, curve_va_ll, label="Val Log Loss (Fold 0 Trips)", color="#f97316", lw=2)
    ax.set_title("Gradient Boosting (Sohel) — Iteration-wise Train vs Val Log Loss", fontweight="bold")
    ax.set_xlabel("Boosting Iterations")
    ax.set_ylabel("Log Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "sohel_gb_learning_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    order = np.argsort(perm.importances_mean)
    ax.barh(range(len(FEATURE_COLS)), perm.importances_mean[order], color="#0d9488")
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels([FEATURE_COLS[i] for i in order])
    ax.set_title("Gradient Boosting (Sohel) — Permutation Feature Importance", fontweight="bold")
    ax.set_xlabel("Mean Accuracy Decrease")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "sohel_gb_feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        np.array(test_m["confusion_matrix"]),
        annot=True, fmt=",d", cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"], ax=ax,
    )
    ax.set_title("Gradient Boosting (Sohel) — Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "sohel_gb_confusion_matrix.png", dpi=150)
    plt.close(fig)

    cv_f1 = [m["macro_f1"] for m in fold_metrics]
    cv_top1 = [m["top1_accuracy"] for m in fold_metrics]
    cv_ll = [m["log_loss"] for m in fold_metrics]

    result = {
        "order": 5,
        "model_name": "Gradient Boosting",
        "owner": "Sohel",
        "algorithm_family": "Boosting",
        "key_configuration": "HistGB(loss=log_loss, lr=0.08, max_iter=200, max_leaf_nodes=31, max_depth=8, min_samples_leaf=50, l2=1.0)",
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
        "feature_importance": feat_imp,
    }

    with open(RESULTS_DIR / "sohel_gradient_boosting_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"  => [Gradient Boosting] CV Macro F1={result['cv']['macro_f1_mean']:.4f} +/- {result['cv']['macro_f1_std']:.4f} | "
        f"Test Macro F1={test_m['macro_f1']:.4f} | Top-1={test_m['top1_accuracy']:.4f} | "
        f"Route Continuity={test_m['route_continuity']:.4f} | LogLoss={test_m['log_loss']:.4f} | "
        f"Train={train_time_s:.1f}s | Infer={inf_time_s:.3f}s"
    )
    return result


if __name__ == "__main__":
    run_gradient_boosting()

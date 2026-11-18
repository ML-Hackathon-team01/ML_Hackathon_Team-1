# Model Evaluation (Team-01 — PathSnap)

This folder contains all source code, metrics, comparison tables, plots, confusion matrices, and spatial visualisations used for **Model Experimentation and Comparison (Section 3)**.

## Contents

- `evaluation_metrics.py` — Shared evaluation metric definitions (`Macro F1`, `Observation-level Top-1 Accuracy`, `Candidate Miss Rate`, `Route Continuity`, `Log Loss`, `Precision`, `Recall`, `Confusion Matrix`).
- `generate_section3_3_figures.py` — Script that exports the consolidated 5-model CSV table and generates all Section 3.3 comparison bar charts, confusion matrix heatmaps, and the interactive Gradient Boosting Folium route map.
- `metrics/` — Consolidated model comparison tables (`model_comparison_table.csv`, `team01_final_model_metrics.csv`, `model_comparison_results.json`) and per-model experiment JSON logs (`*_metrics.json`).
- `plots/` — Comparison bar charts (`fig1_cv_macro_f1_comparison.png`, `fig2_student_test_top1_and_f1.png`, `fig3_log_loss_comparison.png`, `part2_model_comparison.png`), confusion matrix heatmaps (`fig4_all5_confusion_matrices.png`, `part2_confusion_matrices.png`, `sohel_gb_confusion_matrix.png`), Gradient Boosting training vs. validation loss curve (`sohel_gb_learning_curve.png`), feature importance (`sohel_gb_feature_importance.png`), and interactive Folium map (`fig6_gb_predicted_trip_map.html`).

## Run Figure Generation

```powershell
python "Model evaluation/generate_section3_3_figures.py"
```

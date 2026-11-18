"""Generate all Section 3.3 report figures (Tasks 1-8) from Team-01's final 5-model results."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import folium

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "section3_3_report_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Export Team-01 Final Metrics Table (Task 1)
metrics_rows = [
    {
        "Model": "Logistic Regression", "Algorithm Family": "Linear Model",
        "CV Macro F1": 0.7000, "Top-1 Accuracy": 0.7132, "Route Continuity": 0.9160,
        "Student-Test Macro F1": 0.7050, "Log Loss": 0.5250, "Inference Time (s)": 0.0160,
        "TN": 86013, "FP": 39119, "FN": 16836, "TP": 53107,
    },
    {
        "Model": "Decision Tree", "Algorithm Family": "Tree-based",
        "CV Macro F1": 0.6999, "Top-1 Accuracy": 0.6610, "Route Continuity": 0.9160,
        "Student-Test Macro F1": 0.7075, "Log Loss": 0.5251, "Inference Time (s)": 0.0250,
        "TN": 83180, "FP": 16784, "FN": 25098, "TP": 33623,
    },
    {
        "Model": "Random Forest", "Algorithm Family": "Ensemble",
        "CV Macro F1": 0.7435, "Top-1 Accuracy": 0.7500, "Route Continuity": 0.7797,
        "Student-Test Macro F1": 0.7357, "Log Loss": 0.2966, "Inference Time (s)": 2.3000,
        "TN": 1428662, "FP": 220023, "FN": 25595, "TP": 160268,
    },
    {
        "Model": "SVM", "Algorithm Family": "Linear Model",
        "CV Macro F1": 0.7000, "Top-1 Accuracy": 0.7102, "Route Continuity": 0.9100,
        "Student-Test Macro F1": 0.6980, "Log Loss": 0.5300, "Inference Time (s)": 0.0450,
        "TN": 86367, "FP": 40522, "FN": 16661, "TP": 53750,
    },
    {
        "Model": "Gradient Boosting", "Algorithm Family": "Boosting",
        "CV Macro F1": 0.7031, "Top-1 Accuracy": 0.7647, "Route Continuity": 0.7831,
        "Student-Test Macro F1": 0.7372, "Log Loss": 0.5351, "Inference Time (s)": 0.6996,
        "TN": 46981, "FP": 8043, "FN": 15801, "TP": 24975,
    },
]
df = pd.DataFrame(metrics_rows)
df.to_csv(OUT_DIR / "team01_final_model_metrics.csv", index=False)

models = df["Model"].tolist()
x = np.arange(len(models))

# 2. CV Macro F1 Bar Chart (Task 2)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
bars = ax.bar(models, df["CV Macro F1"], color=["#64748b", "#64748b", "#0d9488", "#64748b", "#2563eb"], width=0.55)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.4f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0.55, 0.82)
ax.set_ylabel("CV Macro F1-score")
ax.set_title("Cross-Validation Macro F1-Score Comparison Across All 5 Models", fontweight="bold")
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_cv_macro_f1_comparison.png", dpi=200)
plt.close(fig)

# 3. Student-Test Top-1 Accuracy & Macro F1 Grouped Bar Chart (Task 3)
fig, ax = plt.subplots(figsize=(9.5, 5))
w = 0.36
b1 = ax.bar(x - w / 2, df["Top-1 Accuracy"], width=w, label="Top-1 Accuracy", color="#16a34a")
b2 = ax.bar(x + w / 2, df["Student-Test Macro F1"], width=w, label="Student-Test Macro F1", color="#f97316")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.4f}", ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0.55, 0.85)
ax.set_ylabel("Score")
ax.set_title("Student-Testing Top-1 Accuracy and Macro F1-Score Comparison", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig2_student_test_top1_and_f1.png", dpi=200)
plt.close(fig)

# 4. Log Loss Comparison Bar Chart (Task 4)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
bars = ax.bar(models, df["Log Loss"], color=["#dc2626", "#dc2626", "#16a34a", "#dc2626", "#2563eb"], width=0.55)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.4f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0, 0.65)
ax.set_ylabel("Binary Log Loss (Lower is Better)")
ax.set_title("Student-Testing Log Loss Comparison Across All 5 Models", fontweight="bold")
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_log_loss_comparison.png", dpi=200)
plt.close(fig)

# 5. Confusion Matrix Heatmaps for All 5 Models (Task 5)
fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
plt.suptitle("Confusion Matrix Heatmaps — All Five Models (Team-01)", fontsize=14, fontweight="bold")
for ax, (_, r) in zip(axes, df.iterrows()):
    cm = np.array([[int(r["TN"]), int(r["FP"])], [int(r["FN"]), int(r["TP"])]])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"], ax=ax)
    ax.set_title(r["Model"], fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT_DIR / "fig4_all5_confusion_matrices.png", dpi=200)
plt.close(fig)

# 6. Interactive Folium Map for Gradient Boosting on Student-Test Trip 00000000 (Task 7)
trip_id = "00000000"
trip_dir = ROOT.parent / "Student folder" / "student_test" / trip_id
test_df = pd.read_csv(ROOT / "results" / "pathsnap_student_test.csv", dtype={"trip_id": str})
gb_pkg = joblib.load(ROOT / "saved_models" / "sohel_gradient_boosting.joblib")
gb_model, feat_cols = gb_pkg["model"], gb_pkg["features"]

sub = test_df[test_df["trip_id"] == trip_id].copy()
sub["proba"] = gb_model.predict_proba(sub[feat_cols].to_numpy(dtype=np.float32))[:, 1]
top1 = sub.loc[sub.groupby("observation_id")["proba"].idxmax()].sort_values("observation_id")

track = pd.read_csv(trip_dir / f"{trip_id}.track", sep="\t", header=None, names=["lon", "lat", "ts"])
nodes = pd.read_csv(trip_dir / f"{trip_id}.nodes", sep="\t", header=None, names=["lon", "lat"])
arcs = pd.read_csv(trip_dir / f"{trip_id}.arcs", sep="\t", header=None, names=["u", "v"])
route = pd.read_csv(trip_dir / f"{trip_id}.route", header=None, names=["arc_id"])

n_lon, n_lat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()

m = folium.Map(location=[float(track["lat"].mean()), float(track["lon"].mean())], zoom_start=14, tiles="OpenStreetMap")

fg_ref = folium.FeatureGroup(name="Ground-Truth Reference Route (.route)", show=True)
ref_segs = [[[float(n_lat[u[a]]), float(n_lon[u[a]])], [float(n_lat[v[a]]), float(n_lon[v[a]])]] for a in route["arc_id"].to_numpy()]
folium.PolyLine(ref_segs, color="#16a34a", weight=5, opacity=0.75).add_to(fg_ref)
fg_ref.add_to(m)

fg_pred = folium.FeatureGroup(name="Gradient Boosting Predicted Route (Sohel)", show=True)
pred_arcs = top1["arc_id"].to_numpy()
dedup = [int(pred_arcs[0])] + [int(a) for i, a in enumerate(pred_arcs[1:], 1) if int(a) != int(pred_arcs[i - 1])]
pred_segs = [[[float(n_lat[u[a]]), float(n_lon[u[a]])], [float(n_lat[v[a]]), float(n_lon[v[a]])]] for a in dedup]
folium.PolyLine(pred_segs, color="#dc2626", weight=3, opacity=0.9).add_to(fg_pred)
fg_pred.add_to(m)

fg_gps = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
for i in range(0, len(track), max(1, len(track) // 500)):
    folium.CircleMarker([float(track.iloc[i]["lat"]), float(track.iloc[i]["lon"])],
                        radius=2.5, color="#2563eb", fill=True, fill_opacity=0.75).add_to(fg_gps)
fg_gps.add_to(m)

acc_pct = float((top1["label"] == 1).mean()) * 100.0
legend = f"""
<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;border:2px solid #475569;
            border-radius:6px;padding:10px 14px;font-size:13px;box-shadow:0 2px 6px rgba(0,0,0,0.2);">
  <b>PathSnap — Gradient Boosting (Trip {trip_id})</b><br>
  Trip Top-1 Accuracy: <b>{acc_pct:.2f}%</b><br>
  <span style="color:#2563eb;">&#9679;</span> Raw GPS Observations<br>
  <span style="color:#16a34a;">&#9473;</span> Ground-Truth Reference Route<br>
  <span style="color:#dc2626;">&#9473;</span> Gradient Boosting Predicted Route
</div>"""
m.get_root().html.add_child(folium.Element(legend))
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds([[float(track["lat"].min()), float(track["lon"].min())], [float(track["lat"].max()), float(track["lon"].max())]])
m.save(str(OUT_DIR / "fig6_gb_predicted_trip_map.html"))

print(f"All Section 3.3 figures saved cleanly to: {OUT_DIR}")
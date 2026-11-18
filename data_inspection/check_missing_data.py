from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT.parent / "Student folder"
TRAIN_FOLDER = DATASET_ROOT / "train"

all_results = []

trip_folders = sorted(
    folder for folder in TRAIN_FOLDER.iterdir()
    if folder.is_dir()
)

print("Checking all training trips...")
print("Total trips:", len(trip_folders))

for trip_folder in trip_folders:

    trip_id = trip_folder.name

    files = {
        "track": trip_folder / f"{trip_id}.track",
        "nodes": trip_folder / f"{trip_id}.nodes",
        "arcs": trip_folder / f"{trip_id}.arcs",
        "route": trip_folder / f"{trip_id}.route"
    }

    for file_type, file_path in files.items():

        if file_type == "track":
            df = pd.read_csv(file_path, sep="\t", header=None)

        elif file_type == "nodes":
            df = pd.read_csv(file_path, sep="\t", header=None)

        elif file_type == "arcs":
            df = pd.read_csv(file_path, sep="\t", header=None)

        else:
            df = pd.read_csv(file_path, header=None)

        total_missing = df.isnull().sum().sum()

        all_results.append({
            "trip_id": trip_id,
            "file": file_type,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": total_missing,
            "missing_percentage": (
                df.isnull().sum().sum() / df.size * 100
                if df.size > 0 else 0
            )
        })

results = pd.DataFrame(all_results)

print("\n========== MISSING DATA REPORT ==========")
print(results.to_string(index=False))

print("\n========== TOTAL MISSING VALUES BY FILE ==========")

summary = results.groupby("file").agg({
    "rows": "sum",
    "missing_values": "sum"
})

print(summary)

print("\n========== TRIPS WITH MISSING DATA ==========")

missing_trips = results[results["missing_values"] > 0]

if missing_trips.empty:
    print("No missing values found in any training trip!")
else:
    print(missing_trips.to_string(index=False))

# Save complete report
output_dir = PROJECT_ROOT / "results"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "missing_data_report.csv"

results.to_csv(output_file, index=False)

print("\nReport saved to:", output_file)
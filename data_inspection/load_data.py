from pathlib import Path
import pandas as pd

# Find the PathSnap project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset folder (located outside PathSnap)
DATASET_ROOT = PROJECT_ROOT.parent / "Student folder"

# Training dataset folder
TRAIN_FOLDER = DATASET_ROOT / "train"

# Check whether the dataset exists
if not TRAIN_FOLDER.exists():
    raise FileNotFoundError(f"Train folder not found: {TRAIN_FOLDER}")

print("Dataset path:", DATASET_ROOT)
print("Train folder:", TRAIN_FOLDER)

# Get all training trip folders
trip_folders = sorted(
    [folder for folder in TRAIN_FOLDER.iterdir() if folder.is_dir()]
)

print("Total training trips:", len(trip_folders))

# Select the first trip for initial inspection
trip = trip_folders[0]

print("\nSelected trip:", trip.name)

# Read and inspect the dataset files
for extension in [".track", ".nodes", ".arcs", ".route"]:

    file_path = trip / f"{trip.name}{extension}"

    if file_path.exists():

        # Read tab-separated data correctly
        data = pd.read_csv(file_path, sep="\t", header=None)

        print(f"\n--- {extension} ---")
        print("Shape:", data.shape)
        print(data.head())

    else:
        print(f"\nFile not found: {file_path}")
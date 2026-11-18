"""
Model-Loading & Configuration Logic for PathSnap FastAPI Backend (Laptop 1).

Handles:
  - Loading deployment/config.json
  - Locating and loading the trained model artifact (.joblib)
  - Locating the 10-trip demonstration / evaluator directory (supports PATHSNAP_EVAL_DIR
    override, nested extracted zip folders, and auto-detection of evaluator/demo folders)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "deployment" / "config.json"


def load_config() -> dict[str, Any]:
    """Read deployment configuration JSON."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Deployment config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_path(cfg: dict[str, Any]) -> Path:
    """Resolve trained model artifact path (supports PATHSNAP_MODEL_PATH env override)."""
    env_override = os.environ.get("PATHSNAP_MODEL_PATH")
    if env_override:
        p = Path(env_override).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"PATHSNAP_MODEL_PATH does not exist: {p}")

    for rel in cfg["model"]["artifact_candidates"]:
        candidate = (PROJECT_ROOT / rel).resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No trained model artifact found. Expected one of: "
        + ", ".join(cfg["model"]["artifact_candidates"])
        + ". Run `python models/sohel_gradient_boosting.py` first."
    )


def _find_trip_file(trip_folder: Path, ext: str) -> Path | None:
    """Find `<trip_id>.<ext>` or fallback to first `*.<ext>` inside a trip folder."""
    exact = trip_folder / f"{trip_folder.name}.{ext}"
    if exact.exists():
        return exact
    matches = sorted(trip_folder.glob(f"*.{ext}"))
    return matches[0] if matches else None


def _is_trip_folder(folder: Path) -> bool:
    """True if folder directly contains .track, .nodes, and .arcs files."""
    return (
        folder.is_dir()
        and _find_trip_file(folder, "track") is not None
        and _find_trip_file(folder, "nodes") is not None
        and _find_trip_file(folder, "arcs") is not None
    )


def _unwrap_trip_parent_dir(candidate: Path) -> Path | None:
    """
    If `candidate` directly contains trip subfolders, return `candidate`.
    If `candidate` wraps a single subfolder (common after unzipping, e.g.
    `evaluator_test/evaluator_test/<trips>`), unwrap automatically.
    """
    if not candidate.exists() or not candidate.is_dir():
        return None
    subdirs = [d for d in candidate.iterdir() if d.is_dir()]
    if any(_is_trip_folder(d) for d in subdirs):
        return candidate
    if len(subdirs) == 1:
        inner = subdirs[0]
        inner_subs = [d for d in inner.iterdir() if d.is_dir()]
        if any(_is_trip_folder(d) for d in inner_subs):
            return inner
    return None


def resolve_eval_dir(cfg: dict[str, Any]) -> tuple[Path, str]:
    """
    Resolve the demonstration / evaluator trip folder:
      1. Explicit `PATHSNAP_EVAL_DIR` environment variable (recommended for live demo)
      2. Auto-detect any evaluator/demo folder in workspace or `Student folder`
      3. Fallback to `student_test` for local development testing
    """
    env_dir = os.environ.get("PATHSNAP_EVAL_DIR")
    if env_dir:
        raw_p = Path(env_dir).resolve()
        unwrapped = _unwrap_trip_parent_dir(raw_p)
        if unwrapped is not None:
            return unwrapped, unwrapped.name
        raise FileNotFoundError(
            f"PATHSNAP_EVAL_DIR does not contain valid trip folders (.track/.nodes/.arcs): {raw_p}"
        )

    for rel in cfg["data"]["eval_dir_candidates"]:
        candidate = (PROJECT_ROOT / rel).resolve()
        unwrapped = _unwrap_trip_parent_dir(candidate)
        if unwrapped is not None:
            return unwrapped, unwrapped.name

    raise FileNotFoundError(
        "No evaluation trip folder found. Pass `-EvalDir <path>` to start_laptop1_api.ps1 "
        "or set PATHSNAP_EVAL_DIR to the demonstration dataset folder."
    )


class ModelRegistry:
    """Holds the loaded classifier, feature list, config, and evaluation directory."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = load_config()
        self.model_path: Path = resolve_model_path(self.config)
        self.eval_dir: Path
        self.partition_name: str
        self.eval_dir, self.partition_name = resolve_eval_dir(self.config)

        artifact = joblib.load(self.model_path)
        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.feature_columns: list[str] = list(
                artifact.get(
                    "features",
                    [
                        "candidate_rank",
                        "distance_m",
                        "direction_difference",
                        "gps_speed_mps",
                        "time_gap_s",
                    ],
                )
            )
            self.owner: str = str(artifact.get("owner", "Team-01"))
        else:
            self.model = artifact
            self.feature_columns = [
                "candidate_rank",
                "distance_m",
                "direction_difference",
                "gps_speed_mps",
                "time_gap_s",
            ]
            self.owner = "Team-01"

        self.model_name: str = self.config["model"]["default_model_name"]
        self.algorithm_family: str = self.config["model"]["algorithm_family"]
        self.top_k: int = int(self.config["model"]["top_k_candidates"])

    def list_available_trips(self) -> list[dict[str, Any]]:
        """List available evaluation trips (IDs and GPS observation counts only)."""
        trips = []
        for folder in sorted(self.eval_dir.iterdir()):
            if not _is_trip_folder(folder):
                continue
            trip_id = folder.name
            track_file = _find_trip_file(folder, "track")
            assert track_file is not None
            with open(track_file, "r", encoding="utf-8") as f:
                n_obs = sum(1 for line in f if line.strip())
            trips.append({
                "trip_id": trip_id,
                "n_gps_points": n_obs,
                "partition": self.partition_name,
            })
        return trips

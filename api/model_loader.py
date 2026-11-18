"""
Model-Loading & Configuration Logic for PathSnap FastAPI Backend (Laptop 1).

Handles:
  - Loading deployment/config.json
  - Locating and loading the trained model artifact (.joblib)
  - Locating the 10-trip evaluation directory (evaluator_test or student_test fallback)
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
    """Resolve trained model artifact path (supports MODEL_PATH env override)."""
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


def resolve_eval_dir(cfg: dict[str, Any]) -> tuple[Path, str]:
    """
    Resolve the 10-trip evaluation folder (supports PATHSNAP_EVAL_DIR env override).
    Prefers `evaluator_test` when present during live evaluation, otherwise falls
    back cleanly to `student_test` for development & demo testing.
    """
    env_dir = os.environ.get("PATHSNAP_EVAL_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
        if p.exists():
            return p, p.name
        raise FileNotFoundError(f"PATHSNAP_EVAL_DIR does not exist: {p}")

    for rel in cfg["data"]["eval_dir_candidates"]:
        candidate = (PROJECT_ROOT / rel).resolve()
        if candidate.exists() and any(d.is_dir() for d in candidate.iterdir()):
            return candidate, candidate.name

    raise FileNotFoundError(
        "No evaluation trip folder found. Set PATHSNAP_EVAL_DIR or place trips in "
        "`../Student folder/evaluator_test` or `../Student folder/student_test`."
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
            if not folder.is_dir():
                continue
            trip_id = folder.name
            track_file = folder / f"{trip_id}.track"
            if not track_file.exists():
                continue
            with open(track_file, "r", encoding="utf-8") as f:
                n_obs = sum(1 for _ in f)
            trips.append({
                "trip_id": trip_id,
                "n_gps_points": n_obs,
                "partition": self.partition_name,
            })
        return trips

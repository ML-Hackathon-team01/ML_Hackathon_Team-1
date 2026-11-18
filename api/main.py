"""
PathSnap FastAPI Backend Server (Laptop 1).

Endpoints:
  - GET  /health             : Server & model health check
  - GET  /trips              : List the 10 available evaluation trips
  - POST /predict            : Run full live ML + HMM/Viterbi inference & server-side eval
  - GET  /evaluate/{trip_id} : Convenience GET endpoint for direct testing

Logs every incoming request and response summary cleanly on the receiving laptop's CMD.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.model_loader import ModelRegistry
from api.pipeline import run_trip_inference_and_evaluate

app = FastAPI(
    title="PathSnap Distributed Map-Matching API (Team-01)",
    description="Laptop 1 backend hosting trained Gradient Boosting classifier, HMM/Viterbi decoder, and server-side evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry: ModelRegistry | None = None


class PredictRequest(BaseModel):
    trip_id: str


def _log_cmd(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


@app.on_event("startup")
def startup_event() -> None:
    global registry
    registry = ModelRegistry()
    trips = registry.list_available_trips()
    print("=" * 72, flush=True)
    _log_cmd("PATHSNAP API SERVER STARTED (LAPTOP 1)")
    _log_cmd(f"Loaded Model   : {registry.model_name} ({registry.model_path.name})")
    _log_cmd(f"Features ({len(registry.feature_columns)})   : {registry.feature_columns}")
    _log_cmd(f"Eval Partition : {registry.partition_name} ({len(trips)} trips in {registry.eval_dir})")
    print("=" * 72, flush=True)


def _get_registry() -> ModelRegistry:
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized.")
    return registry


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    reg = _get_registry()
    client_ip = request.client.host if request.client else "unknown"
    trips = reg.list_available_trips()
    _log_cmd(f"GET /health from {client_ip} -> 200 OK (model={reg.model_path.name}, trips={len(trips)})")
    return {
        "status": "online",
        "project": reg.config["project"],
        "model_name": reg.model_name,
        "algorithm_family": reg.algorithm_family,
        "model_artifact": reg.model_path.name,
        "decoder": "HMM + Viterbi",
        "eval_partition": reg.partition_name,
        "available_trips": len(trips),
    }


@app.get("/trips")
def list_trips(request: Request) -> dict[str, Any]:
    reg = _get_registry()
    client_ip = request.client.host if request.client else "unknown"
    trips = reg.list_available_trips()
    _log_cmd(f"GET /trips from {client_ip} -> 200 OK ({len(trips)} trips returned)")
    return {
        "partition": reg.partition_name,
        "count": len(trips),
        "trips": trips,
    }


def _execute_prediction(trip_id: str, client_ip: str, route_label: str) -> dict[str, Any]:
    reg = _get_registry()
    _log_cmd(f"{route_label} from {client_ip} | trip_id={trip_id} | Starting live inference pipeline...")
    try:
        result = run_trip_inference_and_evaluate(reg, trip_id)
    except FileNotFoundError as exc:
        _log_cmd(f"ERROR 404 | trip_id={trip_id} | {exc}")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        _log_cmd(f"ERROR 500 | trip_id={trip_id} | {exc}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    m = result["metrics"]
    t = result["timing_ms"]
    acc_str = f"{m['hmm_top1_accuracy'] * 100:.2f}%" if m["hmm_top1_accuracy"] is not None else "N/A"
    clf_acc_str = (
        f"{m['classifier_only_top1_accuracy'] * 100:.2f}%"
        if m["classifier_only_top1_accuracy"] is not None
        else "N/A"
    )
    _log_cmd(
        f"200 OK | trip_id={trip_id} | GPS={result['trip_summary']['n_gps_points']} | "
        f"Top-1 Acc: {clf_acc_str} -> {acc_str} (HMM) | "
        f"Continuity: {m['classifier_only_route_continuity'] * 100:.2f}% -> {m['hmm_route_continuity'] * 100:.2f}% | "
        f"Total={t['total_ms']:.1f} ms"
    )
    return result


@app.post("/predict")
def predict(payload: PredictRequest, request: Request) -> dict[str, Any]:
    client_ip = request.client.host if request.client else "unknown"
    return _execute_prediction(payload.trip_id, client_ip, "POST /predict")


@app.get("/evaluate/{trip_id}")
def evaluate_get(trip_id: str, request: Request) -> dict[str, Any]:
    client_ip = request.client.host if request.client else "unknown"
    return _execute_prediction(trip_id, client_ip, f"GET /evaluate/{trip_id}")

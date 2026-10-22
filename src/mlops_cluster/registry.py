from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx

from mlops_cluster.planner import SlicePlan

EXPERIMENT = "inference-budget"


class RegistryError(RuntimeError):
    pass


def log_plan_file(plan: SlicePlan, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    payload = {"run_id": run_id, "backend": "file", "plan": plan.as_dict()}
    path = directory / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"run_id": run_id, "backend": "file", "uri": str(path)}


def log_plan_mlflow(plan: SlicePlan, tracking_uri: str, client: httpx.Client | None = None) -> dict:
    """Record the plan through the MLflow tracking REST API."""
    owns_client = client is None
    http = client or httpx.Client(base_url=tracking_uri.rstrip("/"), timeout=10.0)
    try:
        experiment_id = _experiment_id(http)
        created = http.post(
            "/api/2.0/mlflow/runs/create",
            json={
                "experiment_id": experiment_id,
                "run_name": f"{plan.model}-{plan.quant}",
                "tags": [
                    {"key": "model", "value": plan.model},
                    {"key": "quant", "value": plan.quant},
                    {"key": "accepted", "value": str(plan.accepted).lower()},
                ],
            },
        )
        _raise_for_status(created, "create run")
        run_id = created.json()["run"]["info"]["run_id"]
        estimate = plan.estimate
        metrics = []
        now = int(time.time() * 1000)
        if estimate is not None:
            for key, value in (
                ("weight_gib", estimate.weight_gib),
                ("kv_gib", estimate.kv_gib),
                ("total_gib", estimate.total_gib),
                ("reserved_vram_gib", plan.reserved_vram_gib),
            ):
                metrics.append({"key": key, "value": float(value), "timestamp": now, "step": 0})
        params = [
            {"key": "quant", "value": plan.quant},
            {"key": "strategy", "value": plan.strategy or "rejected"},
            {"key": "mig_profile", "value": plan.mig_profile or ""},
            {"key": "tensor_parallel", "value": str(plan.tensor_parallel)},
            {"key": "pipeline_parallel", "value": str(plan.pipeline_parallel)},
            {"key": "replicas", "value": str(plan.replicas)},
            {"key": "physical_gpus", "value": str(plan.physical_gpus)},
            {"key": "repo_id", "value": plan.repo_id},
        ]
        logged = http.post(
            "/api/2.0/mlflow/runs/log-batch",
            json={"run_id": run_id, "params": params, "metrics": metrics, "tags": []},
        )
        _raise_for_status(logged, "log batch")
        status = "FINISHED" if plan.accepted else "FAILED"
        updated = http.post(
            "/api/2.0/mlflow/runs/update",
            json={"run_id": run_id, "status": status, "end_time": now},
        )
        _raise_for_status(updated, "update run")
        return {"run_id": run_id, "backend": "mlflow", "uri": f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"}
    except httpx.HTTPError as exc:
        raise RegistryError(f"MLflow tracking server unreachable at {tracking_uri}") from exc
    finally:
        if owns_client:
            http.close()


def _experiment_id(http: httpx.Client) -> str:
    found = http.post(
        "/api/2.0/mlflow/experiments/get-by-name",
        json={"experiment_name": EXPERIMENT},
    )
    if found.status_code == 200:
        return str(found.json()["experiment"]["experiment_id"])
    if found.status_code not in {400, 404}:
        _raise_for_status(found, "get experiment")
    created = http.post("/api/2.0/mlflow/experiments/create", json={"name": EXPERIMENT})
    if created.status_code == 200:
        return str(created.json()["experiment_id"])
    # Another writer created it between the lookup and the insert.
    again = http.post(
        "/api/2.0/mlflow/experiments/get-by-name",
        json={"experiment_name": EXPERIMENT},
    )
    _raise_for_status(again, "get experiment after create")
    return str(again.json()["experiment"]["experiment_id"])


def _raise_for_status(response: httpx.Response, action: str) -> None:
    if response.status_code >= 400:
        raise RegistryError(f"MLflow {action} failed: HTTP {response.status_code} {response.text[:200]}")

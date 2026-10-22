import json

import httpx

from mlops_cluster.catalog import load_cluster, load_models
from mlops_cluster.planner import plan_deployment
from mlops_cluster.registry import log_plan_file, log_plan_mlflow


def _plan():
    return plan_deployment(load_models()["qwen2.5-7b-instruct"], "awq", 2, load_cluster())


def test_file_registry_writes_the_plan(tmp_path):
    recorded = log_plan_file(_plan(), tmp_path)
    payload = json.loads((tmp_path / f"{recorded['run_id']}.json").read_text(encoding="utf-8"))
    assert payload["plan"]["mig_profile"] == "2g.10gb"
    assert payload["backend"] == "file"


def test_mlflow_rest_client_logs_a_run():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content.decode())))
        if request.url.path.endswith("/experiments/get-by-name"):
            return httpx.Response(404, json={"error_code": "RESOURCE_DOES_NOT_EXIST"})
        if request.url.path.endswith("/experiments/create"):
            return httpx.Response(200, json={"experiment_id": "7"})
        if request.url.path.endswith("/runs/create"):
            return httpx.Response(200, json={"run": {"info": {"run_id": "run-1"}}})
        if request.url.path.endswith("/runs/log-batch"):
            return httpx.Response(200, json={})
        if request.url.path.endswith("/runs/update"):
            return httpx.Response(200, json={})
        return httpx.Response(500, json={"error": "unexpected"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mlflow:5000")
    recorded = log_plan_mlflow(_plan(), "http://mlflow:5000", client=client)
    assert recorded["run_id"] == "run-1"
    batch = next(body for path, body in seen if path.endswith("/runs/log-batch"))
    keys = {item["key"] for item in batch["params"]}
    assert "strategy" in keys
    assert "mig_profile" in keys
    metric_keys = {item["key"] for item in batch["metrics"]}
    assert "total_gib" in metric_keys

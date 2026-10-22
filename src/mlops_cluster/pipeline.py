from __future__ import annotations

import os
import subprocess
from pathlib import Path

from mlops_cluster.catalog import load_cluster, load_models, with_concurrency
from mlops_cluster.planner import plan_deployment
from mlops_cluster.registry import log_plan_file, log_plan_mlflow
from mlops_cluster.render import write_manifests


def deploy_model(
    model: str,
    quant: str = "awq",
    replicas: int = 1,
    *,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    out_dir: Path | None = None,
    tracking_uri: str | None = None,
    apply: bool = False,
) -> dict:
    """Plan, register, and render. ``apply`` shells out to kubectl."""
    spec = with_concurrency(load_models()[model], max_model_len, max_num_seqs)
    cluster = load_cluster()
    plan = plan_deployment(spec, quant, replicas, cluster)
    directory = out_dir or Path(os.environ.get("MLOPS_OUT_DIR", "generated")) / spec.name
    registry: dict = {"file": log_plan_file(plan, directory / "runs")}
    if tracking_uri:
        registry["mlflow"] = log_plan_mlflow(plan, tracking_uri)
    manifest_dir = directory / "manifests"
    written: list[str] = []
    if plan.accepted:
        written = sorted(write_manifests(plan, manifest_dir))
    kubectl = None
    if apply:
        if not plan.accepted:
            raise RuntimeError(plan.reason)
        kubectl = _kubectl_apply(manifest_dir)
    return {
        "accepted": plan.accepted,
        "plan": plan.as_dict(),
        "registry": registry,
        "manifests": written,
        "kubectl": kubectl,
    }


def _kubectl_apply(directory: Path) -> str:
    completed = subprocess.run(
        ["kubectl", "apply", "-k", str(directory)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(detail or "kubectl apply failed")
    return completed.stdout.strip()

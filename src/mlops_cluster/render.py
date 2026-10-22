from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

from mlops_cluster.planner import SlicePlan


def dns_name(model: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", model.lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned[:52] or "model"


def workload_name(plan: SlicePlan) -> str:
    return f"vllm-{dns_name(plan.model)}"


def host_memory_gi(plan: SlicePlan) -> int:
    if plan.estimate is None:
        return 4
    shards = max(1, plan.tensor_parallel * plan.pipeline_parallel)
    raw = plan.estimate.weight_gib + (plan.estimate.kv_gib / shards) + 4
    return min(64, max(2, math.ceil(raw)))


def render_plan(plan: SlicePlan, namespace: str = "inference") -> dict[str, str]:
    """Render the serving workload for an accepted plan.

    MIG plans request ``nvidia.com/mig-<profile>``. Dedicated and time-sliced
    plans request ``nvidia.com/gpu``. Pipeline-parallel plans are a StatefulSet
    so each stage has a stable DNS name; the client Service selects ordinal 0.
    """
    if not plan.accepted or plan.gpu_resource is None or plan.strategy is None:
        raise ValueError("cannot render a rejected plan")

    name = workload_name(plan)
    memory = host_memory_gi(plan)
    memory_limit = min(64, memory + 8)
    if plan.estimate and plan.estimate.weight_gib > 20:
        cpu_request, cpu_limit = "4", "8"
    else:
        cpu_request, cpu_limit = "2", "4"
    labels = {
        "app.kubernetes.io/name": name,
        "app.kubernetes.io/part-of": "mlops-vllm-k8s-cluster",
        "app.kubernetes.io/component": "llm",
        "mlops.stackmesh/quant": plan.quant,
        "mlops.stackmesh/strategy": plan.strategy,
    }
    annotations = {
        "mlops.stackmesh/reason": plan.reason[:240],
        "mlops.stackmesh/reserved-vram-gib": f"{plan.reserved_vram_gib:.2f}",
        "mlops.stackmesh/pools": ",".join(plan.pools),
        "mlops.stackmesh/repo-matches-quant": "true" if plan.repo_matches_quant else "false",
        "mlops.stackmesh/cross-node": "true" if plan.cross_node else "false",
    }
    resources = {
        "requests": {
            "cpu": cpu_request,
            "memory": f"{memory}Gi",
            plan.gpu_resource: str(plan.gpus_per_pod),
        },
        "limits": {
            "cpu": cpu_limit,
            "memory": f"{memory_limit}Gi",
            plan.gpu_resource: str(plan.gpus_per_pod),
        },
    }
    container = {
        "name": "vllm",
        "image": plan.image or "vllm/vllm-openai:v0.32.0",
        "imagePullPolicy": "IfNotPresent",
        "args": plan.vllm_args,
        "ports": [{"name": "http", "containerPort": 8000}],
        "resources": resources,
        "startupProbe": {
            "httpGet": {"path": "/health", "port": "http"},
            "periodSeconds": 10,
            "failureThreshold": 90,
        },
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": "http"},
            "periodSeconds": 10,
            "failureThreshold": 3,
        },
    }
    pod_spec: dict = {
        "priorityClassName": "inference-latency",
        "containers": [container],
    }
    if plan.pipeline_parallel > 1:
        pod_spec["affinity"] = {
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {"matchLabels": {"app.kubernetes.io/name": name}},
                        "topologyKey": "kubernetes.io/hostname",
                    }
                ]
            }
        }

    documents: dict[str, object] = {}
    pod_labels = dict(labels)
    pod_annotations = dict(annotations)
    if plan.pipeline_parallel > 1:
        documents["statefulset.yaml"] = {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": dict(labels),
                "annotations": dict(annotations),
            },
            "spec": {
                "serviceName": f"{name}-headless",
                "replicas": plan.pipeline_parallel,
                "podManagementPolicy": "Parallel",
                "selector": {"matchLabels": {"app.kubernetes.io/name": name}},
                "template": {
                    "metadata": {"labels": pod_labels, "annotations": pod_annotations},
                    "spec": pod_spec,
                },
            },
        }
        documents["headless-service.yaml"] = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": f"{name}-headless", "namespace": namespace, "labels": labels},
            "spec": {
                "clusterIP": "None",
                "publishNotReadyAddresses": True,
                "selector": {"app.kubernetes.io/name": name},
                "ports": [{"name": "http", "port": 8000, "targetPort": "http"}],
            },
        }
        selector = {
            "app.kubernetes.io/name": name,
            "statefulset.kubernetes.io/pod-name": f"{name}-0",
        }
        workload_file = "statefulset.yaml"
    else:
        documents["deployment.yaml"] = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": dict(labels),
                "annotations": dict(annotations),
            },
            "spec": {
                "replicas": plan.replicas,
                "selector": {"matchLabels": {"app.kubernetes.io/name": name}},
                "template": {
                    "metadata": {"labels": pod_labels, "annotations": pod_annotations},
                    "spec": pod_spec,
                },
            },
        }
        selector = {"app.kubernetes.io/name": name}
        workload_file = "deployment.yaml"

    documents["service.yaml"] = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "selector": selector,
            "ports": [{"name": "http", "port": 8000, "targetPort": "http"}],
        },
    }
    if plan.strategy == "time_slice":
        documents["time-slicing-config.yaml"] = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "nvidia-device-plugin-config", "namespace": "kube-system"},
            "data": {
                "config.yaml": (
                    "version: v1\n"
                    "sharing:\n"
                    "  timeSlicing:\n"
                    "    renameByDefault: false\n"
                    "    resources:\n"
                    "    - name: nvidia.com/gpu\n"
                    f"      replicas: {plan.time_slice_per_gpu}\n"
                )
            },
        }

    rendered = {filename: _dump(document) for filename, document in documents.items()}
    resource_names = [workload_file, "service.yaml"]
    if "headless-service.yaml" in rendered:
        resource_names.insert(1, "headless-service.yaml")
    kustomization = {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "namespace": namespace,
        "resources": resource_names,
    }
    rendered["kustomization.yaml"] = _dump(kustomization)
    return rendered


def write_manifests(plan: SlicePlan, directory: Path, namespace: str = "inference") -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    rendered = render_plan(plan, namespace=namespace)
    for filename, body in rendered.items():
        (directory / filename).write_text(body, encoding="utf-8")
    return rendered


def _dump(document: dict) -> str:
    return yaml.safe_dump(document, sort_keys=False)

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from mlops_cluster.paths import config_dir


@dataclass(frozen=True)
class MigProfile:
    """One NVIDIA GPU-instance profile.

    ``vram_gib`` is the capacity nvidia-smi reports, which is lower than the
    profile name. ``sm_fraction`` is compute slices over 7. ``slices_per_gpu``
    is how many of this profile fit on one device when the GPU is devoted to it.
    """

    name: str
    vram_gib: float
    slices_per_gpu: int
    sm_fraction: float
    full_device: bool = False


@dataclass(frozen=True)
class GpuSku:
    name: str
    vram_gib: float
    mig_profiles: tuple[MigProfile, ...] = ()


# A100-SXM4-40GB instance counts and memory are the MIG user-guide figures
# (1g.5gb = 4.75 GiB, 2g.10gb = 9.62 GiB, 3g.20gb = 19.50 GiB).
_A100_40 = (
    MigProfile("1g.5gb", 4.75, 7, 1 / 7),
    MigProfile("1g.10gb", 9.62, 4, 1 / 7),
    MigProfile("2g.10gb", 9.62, 3, 2 / 7),
    MigProfile("3g.20gb", 19.50, 2, 3 / 7),
    MigProfile("4g.20gb", 19.50, 1, 4 / 7),
    MigProfile("7g.40gb", 39.25, 1, 1.0, full_device=True),
)

# 80GB names follow the same slice geometry with doubled memory labels.
_A100_80 = (
    MigProfile("1g.10gb", 9.50, 7, 1 / 7),
    MigProfile("1g.20gb", 19.24, 4, 1 / 7),
    MigProfile("2g.20gb", 19.24, 3, 2 / 7),
    MigProfile("3g.40gb", 39.00, 2, 3 / 7),
    MigProfile("4g.40gb", 39.00, 1, 4 / 7),
    MigProfile("7g.80gb", 78.50, 1, 1.0, full_device=True),
)

SKUS: dict[str, GpuSku] = {
    "a100-40gb": GpuSku("a100-40gb", 40.0, _A100_40),
    "a100-80gb": GpuSku("a100-80gb", 80.0, _A100_80),
    "rtx-4090-24gb": GpuSku("rtx-4090-24gb", 24.0),
    "l4-24gb": GpuSku("l4-24gb", 24.0),
    "t4-16gb": GpuSku("t4-16gb", 16.0),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float
    n_layers: int
    n_kv_heads: int
    head_dim: int
    max_model_len: int
    max_num_seqs: int
    repo: str
    repos: dict[str, str]

    def repo_for(self, quant: str) -> tuple[str, bool]:
        if quant in self.repos:
            return self.repos[quant], True
        if quant in {"fp16", "bf16", "fp32"}:
            return self.repo, True
        return self.repo, False


@dataclass(frozen=True)
class Budget:
    max_physical_gpus: int
    max_vram_gib: float
    max_replicas: int
    gpu_memory_utilization: float = 0.90
    weight_overhead: float = 1.15
    max_time_slice_per_gpu: int = 4


@dataclass(frozen=True)
class NodePool:
    name: str
    sku: str
    nodes: int
    gpus_per_node: int
    mig_enabled: bool

    @property
    def physical_gpus(self) -> int:
        return self.nodes * self.gpus_per_node


@dataclass(frozen=True)
class Cluster:
    name: str
    budget: Budget
    pools: tuple[NodePool, ...]
    images: dict[str, str]


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    return data


def load_models(path: Path | None = None) -> dict[str, ModelSpec]:
    file = path or (config_dir() / "models.yaml")
    raw = _load_yaml(file)
    models: dict[str, ModelSpec] = {}
    for name, item in raw["models"].items():
        models[name] = ModelSpec(
            name=name,
            params_b=float(item["params_b"]),
            n_layers=int(item["n_layers"]),
            n_kv_heads=int(item["n_kv_heads"]),
            head_dim=int(item["head_dim"]),
            max_model_len=int(item["max_model_len"]),
            max_num_seqs=int(item["max_num_seqs"]),
            repo=str(item["repo"]),
            repos={str(key): str(value) for key, value in (item.get("repos") or {}).items()},
        )
    return models


def load_cluster(path: Path | None = None) -> Cluster:
    file = path or (config_dir() / "cluster.yaml")
    raw = _load_yaml(file)
    budget_raw = raw["budget"]
    budget = Budget(
        max_physical_gpus=int(budget_raw["max_physical_gpus"]),
        max_vram_gib=float(budget_raw["max_vram_gib"]),
        max_replicas=int(budget_raw["max_replicas"]),
        gpu_memory_utilization=float(budget_raw.get("gpu_memory_utilization", 0.90)),
        weight_overhead=float(budget_raw.get("weight_overhead", 1.15)),
        max_time_slice_per_gpu=int(budget_raw.get("max_time_slice_per_gpu", 4)),
    )
    pools = tuple(
        NodePool(
            name=str(item["name"]),
            sku=str(item["sku"]),
            nodes=int(item["nodes"]),
            gpus_per_node=int(item["gpus_per_node"]),
            mig_enabled=bool(item["mig_enabled"]),
        )
        for item in raw["pools"]
    )
    for pool in pools:
        if pool.sku not in SKUS:
            known = ", ".join(sorted(SKUS))
            raise ValueError(f"unknown sku {pool.sku!r} on {pool.name}; expected {known}")
    images = {str(key): str(value) for key, value in (raw.get("images") or {}).items()}
    return Cluster(name=str(raw["name"]), budget=budget, pools=pools, images=images)


def with_concurrency(model: ModelSpec, max_model_len: int | None, max_num_seqs: int | None) -> ModelSpec:
    changes = {}
    if max_model_len is not None:
        changes["max_model_len"] = max_model_len
    if max_num_seqs is not None:
        changes["max_num_seqs"] = max_num_seqs
    return replace(model, **changes) if changes else model

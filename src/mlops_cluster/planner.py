from __future__ import annotations

import math
from dataclasses import dataclass, field

from mlops_cluster.catalog import SKUS, Budget, Cluster, ModelSpec, NodePool
from mlops_cluster.memory import QUANT_BYTES, MemoryEstimate, estimate_memory

STRATEGY_RANK = {"mig": 0, "dedicated": 1, "time_slice": 2}


@dataclass(frozen=True)
class PlacementDomain:
    sku_name: str
    gpus_per_node: int
    mig_enabled: bool
    physical_gpus: int
    nodes: int
    pools: tuple[NodePool, ...]


@dataclass
class SlicePlan:
    accepted: bool
    reason: str
    hint: str | None
    model: str
    quant: str
    strategy: str | None = None
    mig_profile: str | None = None
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    replicas: int = 0
    physical_gpus: int = 0
    reserved_vram_gib: float = 0.0
    per_gpu_gib: float = 0.0
    gpu_memory_utilization: float = 0.0
    sm_fraction: float = 0.0
    gpus_per_pod: int = 0
    cross_node: bool = False
    pools: tuple[str, ...] = ()
    repo_id: str = ""
    repo_matches_quant: bool = False
    max_model_len: int = 0
    max_num_seqs: int = 0
    time_slice_per_gpu: int = 1
    estimate: MemoryEstimate | None = None
    vllm_args: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    image: str = ""

    def as_dict(self) -> dict:
        estimate = self.estimate.as_dict() if self.estimate else None
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "hint": self.hint,
            "model": self.model,
            "quant": self.quant,
            "strategy": self.strategy,
            "mig_profile": self.mig_profile,
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "replicas": self.replicas,
            "physical_gpus": self.physical_gpus,
            "reserved_vram_gib": round(self.reserved_vram_gib, 4),
            "per_gpu_gib": round(self.per_gpu_gib, 4),
            "gpu_memory_utilization": round(self.gpu_memory_utilization, 4),
            "sm_fraction": round(self.sm_fraction, 4),
            "gpus_per_pod": self.gpus_per_pod,
            "cross_node": self.cross_node,
            "pools": list(self.pools),
            "repo_id": self.repo_id,
            "repo_matches_quant": self.repo_matches_quant,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "time_slice_per_gpu": self.time_slice_per_gpu,
            "estimate": estimate,
            "vllm_args": list(self.vllm_args),
            "alternatives": list(self.alternatives),
            "image": self.image,
            "gpu_resource": self.gpu_resource,
        }

    @property
    def gpu_resource(self) -> str | None:
        if not self.accepted:
            return None
        if self.strategy == "mig" and self.mig_profile:
            return f"nvidia.com/mig-{self.mig_profile}"
        return "nvidia.com/gpu"


def domains_for(cluster: Cluster) -> list[PlacementDomain]:
    grouped: dict[tuple, list[NodePool]] = {}
    for pool in cluster.pools:
        key = (pool.sku, pool.gpus_per_node, pool.mig_enabled)
        grouped.setdefault(key, []).append(pool)
    domains = []
    for (sku_name, gpus_per_node, mig_enabled), pools in grouped.items():
        domains.append(
            PlacementDomain(
                sku_name=sku_name,
                gpus_per_node=gpus_per_node,
                mig_enabled=mig_enabled,
                physical_gpus=sum(pool.physical_gpus for pool in pools),
                nodes=sum(pool.nodes for pool in pools),
                pools=tuple(pools),
            )
        )
    return domains


def _assign_pools(domain: PlacementDomain, physical_gpus: int) -> tuple[str, ...]:
    remaining = physical_gpus
    names: list[str] = []
    for pool in domain.pools:
        if remaining <= 0:
            break
        take = min(pool.physical_gpus, remaining)
        if take:
            names.append(pool.name)
            remaining -= take
    return tuple(names)


def _sort_key(plan: SlicePlan) -> tuple:
    return (
        plan.physical_gpus,
        round(plan.reserved_vram_gib, 4),
        STRATEGY_RANK[plan.strategy or "time_slice"],
        -round(plan.sm_fraction, 4),
        plan.tensor_parallel + plan.pipeline_parallel,
    )


def _gpus(count: int) -> str:
    return f"{count} GPU" if count == 1 else f"{count} GPUs"


def _label(plan: SlicePlan) -> str:
    if plan.strategy == "mig":
        return (
            f"mig:{plan.mig_profile} gpus={plan.physical_gpus} "
            f"reserved={plan.reserved_vram_gib:.1f}"
        )
    return (
        f"{plan.strategy} tp={plan.tensor_parallel} pp={plan.pipeline_parallel} "
        f"gpus={plan.physical_gpus} reserved={plan.reserved_vram_gib:.1f}"
    )


def _vllm_args(plan: SlicePlan) -> list[str]:
    dtype = {"fp16": "float16", "bf16": "bfloat16"}.get(plan.quant, "auto")
    args = [
        "--model",
        plan.repo_id,
        "--served-model-name",
        plan.model,
        "--tensor-parallel-size",
        str(plan.tensor_parallel),
        "--pipeline-parallel-size",
        str(plan.pipeline_parallel),
        "--gpu-memory-utilization",
        f"{plan.gpu_memory_utilization:.4f}",
        "--max-model-len",
        str(plan.max_model_len),
        "--max-num-seqs",
        str(plan.max_num_seqs),
        "--dtype",
        dtype,
    ]
    if plan.quant in {"awq", "gptq"}:
        args.extend(["--quantization", plan.quant])
    elif plan.quant == "int8":
        args.extend(["--quantization", "bitsandbytes"])
    return args


def _blank(
    model: ModelSpec,
    quant: str,
    estimate: MemoryEstimate,
    *,
    strategy: str,
    mig_profile: str | None,
    tensor_parallel: int,
    pipeline_parallel: int,
    replicas: int,
    physical_gpus: int,
    reserved_vram_gib: float,
    gpu_memory_utilization: float,
    sm_fraction: float,
    pools: tuple[str, ...],
    time_slice_per_gpu: int,
    image: str,
) -> SlicePlan:
    repo_id, verified = model.repo_for(quant)
    per_gpu = estimate.total_gib / (tensor_parallel * pipeline_parallel)
    gpus_per_pod = 1 if strategy in {"mig", "time_slice"} else tensor_parallel
    plan = SlicePlan(
        accepted=True,
        reason="",
        hint=None,
        model=model.name,
        quant=quant,
        strategy=strategy,
        mig_profile=mig_profile,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        replicas=replicas,
        physical_gpus=physical_gpus,
        reserved_vram_gib=reserved_vram_gib,
        per_gpu_gib=per_gpu,
        gpu_memory_utilization=gpu_memory_utilization,
        sm_fraction=sm_fraction,
        gpus_per_pod=gpus_per_pod,
        cross_node=pipeline_parallel > 1 or len(pools) > 1,
        pools=pools,
        repo_id=repo_id,
        repo_matches_quant=verified,
        max_model_len=model.max_model_len,
        max_num_seqs=model.max_num_seqs,
        time_slice_per_gpu=time_slice_per_gpu,
        estimate=estimate,
        image=image,
    )
    where = f" {mig_profile}" if mig_profile else ""
    replica_word = "replica" if replicas == 1 else "replicas"
    gpu_word = "GPU" if physical_gpus == 1 else "GPUs"
    plan.reason = (
        f"Placed {replicas} {replica_word} of {model.name} ({quant}) via {strategy}{where} "
        f"on {physical_gpus} physical {gpu_word}, {reserved_vram_gib:.1f} GiB reserved."
    )
    plan.vllm_args = _vllm_args(plan)
    return plan


def _fits_budget(budget: Budget, physical_gpus: int, reserved_vram_gib: float, replicas: int) -> bool:
    return (
        replicas <= budget.max_replicas
        and physical_gpus <= budget.max_physical_gpus
        and reserved_vram_gib <= budget.max_vram_gib + 1e-6
    )


def candidates(model: ModelSpec, quant: str, replicas: int, cluster: Cluster) -> list[SlicePlan]:
    budget = cluster.budget
    estimate = estimate_memory(
        params_b=model.params_b,
        n_layers=model.n_layers,
        n_kv_heads=model.n_kv_heads,
        head_dim=model.head_dim,
        max_model_len=model.max_model_len,
        max_num_seqs=model.max_num_seqs,
        quant=quant,
        weight_overhead=budget.weight_overhead,
    )
    need = estimate.total_gib
    image = cluster.images.get("vllm", "vllm/vllm-openai:v0.32.0")
    found: list[SlicePlan] = []

    for domain in domains_for(cluster):
        sku = SKUS[domain.sku_name]
        gpus_allowed = min(domain.physical_gpus, budget.max_physical_gpus)

        def usable(vram: float) -> float:
            return vram * budget.gpu_memory_utilization

        if domain.mig_enabled and gpus_allowed >= 1:
            for profile in sku.mig_profiles:
                if profile.full_device or need > usable(profile.vram_gib):
                    continue
                slice_cap = min(
                    gpus_allowed * profile.slices_per_gpu,
                    math.floor(budget.max_vram_gib / profile.vram_gib),
                    budget.max_replicas,
                )
                if replicas > slice_cap or replicas < 1:
                    continue
                physical = math.ceil(replicas / profile.slices_per_gpu)
                reserved = replicas * profile.vram_gib
                if not _fits_budget(budget, physical, reserved, replicas):
                    continue
                found.append(
                    _blank(
                        model,
                        quant,
                        estimate,
                        strategy="mig",
                        mig_profile=profile.name,
                        tensor_parallel=1,
                        pipeline_parallel=1,
                        replicas=replicas,
                        physical_gpus=physical,
                        reserved_vram_gib=reserved,
                        gpu_memory_utilization=budget.gpu_memory_utilization,
                        sm_fraction=profile.sm_fraction,
                        pools=_assign_pools(domain, physical),
                        time_slice_per_gpu=1,
                        image=image,
                    )
                )

        if gpus_allowed < 1:
            continue
        tp_options = [tp for tp in (1, 2, 4, 8) if tp <= domain.gpus_per_node]
        for tp in tp_options:
            per_gpu = need / tp
            if per_gpu <= usable(sku.vram_gib):
                physical = replicas * tp
                reserved = physical * sku.vram_gib
                if _fits_budget(budget, physical, reserved, replicas):
                    found.append(
                        _blank(
                            model,
                            quant,
                            estimate,
                            strategy="dedicated",
                            mig_profile=None,
                            tensor_parallel=tp,
                            pipeline_parallel=1,
                            replicas=replicas,
                            physical_gpus=physical,
                            reserved_vram_gib=reserved,
                            gpu_memory_utilization=budget.gpu_memory_utilization,
                            sm_fraction=1.0,
                            pools=_assign_pools(domain, physical),
                            time_slice_per_gpu=1,
                            image=image,
                        )
                    )
                continue
            if tp != domain.gpus_per_node or replicas != 1:
                continue
            for pp in (2, 3, 4):
                if pp > domain.nodes:
                    continue
                group = tp * pp
                if need / group > usable(sku.vram_gib):
                    continue
                reserved = group * sku.vram_gib
                if not _fits_budget(budget, group, reserved, replicas):
                    continue
                found.append(
                    _blank(
                        model,
                        quant,
                        estimate,
                        strategy="dedicated",
                        mig_profile=None,
                        tensor_parallel=tp,
                        pipeline_parallel=pp,
                        replicas=replicas,
                        physical_gpus=group,
                        reserved_vram_gib=reserved,
                        gpu_memory_utilization=budget.gpu_memory_utilization,
                        sm_fraction=1.0,
                        pools=_assign_pools(domain, group),
                        time_slice_per_gpu=1,
                        image=image,
                    )
                )

        if need <= usable(sku.vram_gib):
            fit = math.floor((usable(sku.vram_gib) + 1e-9) / need)
            per_gpu_replicas = min(budget.max_time_slice_per_gpu, fit, replicas)
            if per_gpu_replicas >= 2:
                physical = math.ceil(replicas / per_gpu_replicas)
                reserved = physical * sku.vram_gib
                if _fits_budget(budget, physical, reserved, replicas):
                    found.append(
                        _blank(
                            model,
                            quant,
                            estimate,
                            strategy="time_slice",
                            mig_profile=None,
                            tensor_parallel=1,
                            pipeline_parallel=1,
                            replicas=replicas,
                            physical_gpus=physical,
                            reserved_vram_gib=reserved,
                            gpu_memory_utilization=budget.gpu_memory_utilization / per_gpu_replicas,
                            sm_fraction=1.0 / per_gpu_replicas,
                            pools=_assign_pools(domain, physical),
                            time_slice_per_gpu=per_gpu_replicas,
                            image=image,
                        )
                    )
    return found


def _suggest(model: ModelSpec, quant: str, replicas: int, cluster: Cluster) -> str | None:
    hints: list[str] = []
    current_bytes = QUANT_BYTES[quant]
    for alt in ("awq", "gptq", "int8"):
        if QUANT_BYTES[alt] >= current_bytes:
            continue
        alt_plan = plan_deployment(model, alt, replicas, cluster, _hints=False)
        if alt_plan.accepted:
            hints.append(
                f"{alt} fits ({alt_plan.strategy}, {_gpus(alt_plan.physical_gpus)}, "
                f"{alt_plan.reserved_vram_gib:.0f} GiB reserved)"
            )
            break
    if replicas > 1:
        smaller = plan_deployment(model, quant, 1, cluster, _hints=False)
        if smaller.accepted:
            hints.append(
                f"replicas=1 fits ({smaller.strategy}, {_gpus(smaller.physical_gpus)}, "
                f"{smaller.reserved_vram_gib:.0f} GiB reserved)"
            )
    return "; ".join(hints) if hints else None


def plan_deployment(
    model: ModelSpec,
    quant: str,
    replicas: int,
    cluster: Cluster,
    *,
    _hints: bool = True,
) -> SlicePlan:
    if quant not in QUANT_BYTES:
        known = ", ".join(sorted(QUANT_BYTES))
        raise ValueError(f"unsupported quant {quant!r}; expected one of {known}")
    if replicas < 1:
        raise ValueError("replicas must be >= 1")

    estimate = estimate_memory(
        params_b=model.params_b,
        n_layers=model.n_layers,
        n_kv_heads=model.n_kv_heads,
        head_dim=model.head_dim,
        max_model_len=model.max_model_len,
        max_num_seqs=model.max_num_seqs,
        quant=quant,
        weight_overhead=cluster.budget.weight_overhead,
    )
    options = candidates(model, quant, replicas, cluster)
    if not options:
        budget: Budget = cluster.budget
        reason = (
            f"{model.name} at {quant} needs {estimate.total_gib:.2f} GiB "
            f"({estimate.weight_with_overhead_gib:.2f} GiB weights incl. overhead + "
            f"{estimate.kv_gib:.2f} GiB KV). No MIG slice or parallel group fits within "
            f"{budget.max_physical_gpus} GPUs and {budget.max_vram_gib:.0f} GiB."
        )
        hint = _suggest(model, quant, replicas, cluster) if _hints else None
        return SlicePlan(
            accepted=False,
            reason=reason,
            hint=hint,
            model=model.name,
            quant=quant,
            replicas=replicas,
            estimate=estimate,
            max_model_len=model.max_model_len,
            max_num_seqs=model.max_num_seqs,
            repo_id=model.repo_for(quant)[0],
            repo_matches_quant=model.repo_for(quant)[1],
        )

    options.sort(key=_sort_key)
    chosen = options[0]
    chosen.alternatives = [_label(plan) for plan in options[1:6]]
    if not chosen.repo_matches_quant:
        chosen.reason += (
            f" Checkpoint {chosen.repo_id} is the unquantized repo; "
            f"set the {quant.upper()} checkpoint before apply."
        )
    return chosen

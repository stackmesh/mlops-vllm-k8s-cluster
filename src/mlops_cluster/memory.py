from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3

# Checkpoint bytes per parameter. KV cache stays fp16 for the 4-bit and 8-bit
# weight formats vLLM actually serves.
QUANT_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "awq": 0.5,
    "gptq": 0.5,
}

KV_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 2.0,
    "awq": 2.0,
    "gptq": 2.0,
}


@dataclass(frozen=True)
class MemoryEstimate:
    quant: str
    weight_gib: float
    weight_with_overhead_gib: float
    kv_gib: float
    total_gib: float
    kv_bytes_per_token: int

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "quant": self.quant,
            "weight_gib": round(self.weight_gib, 4),
            "weight_with_overhead_gib": round(self.weight_with_overhead_gib, 4),
            "kv_gib": round(self.kv_gib, 4),
            "total_gib": round(self.total_gib, 4),
            "kv_bytes_per_token": self.kv_bytes_per_token,
        }


def estimate_memory(
    *,
    params_b: float,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    max_model_len: int,
    max_num_seqs: int,
    quant: str,
    weight_overhead: float,
) -> MemoryEstimate:
    """Reserve weights plus the full configured KV cache.

    vLLM shards both terms across tensor and pipeline stages. The planner
    divides ``total_gib`` by ``tp * pp`` later. ``weight_overhead`` covers
    AWQ/GPTQ scales, the CUDA context, and activation scratch — it is not a
    measured profiler trace.
    """
    if quant not in QUANT_BYTES:
        known = ", ".join(sorted(QUANT_BYTES))
        raise ValueError(f"unsupported quant {quant!r}; expected one of {known}")
    if min(params_b, n_layers, n_kv_heads, head_dim, max_model_len, max_num_seqs) <= 0:
        raise ValueError("model geometry and concurrency must be positive")
    if weight_overhead < 1:
        raise ValueError("weight_overhead must be >= 1")

    weight = params_b * 1e9 * QUANT_BYTES[quant] / GIB
    kv_bytes_per_token = int(2 * n_layers * n_kv_heads * head_dim * KV_BYTES[quant])
    kv = kv_bytes_per_token * max_model_len * max_num_seqs / GIB
    inflated = weight * weight_overhead
    return MemoryEstimate(
        quant=quant,
        weight_gib=weight,
        weight_with_overhead_gib=inflated,
        kv_gib=kv,
        total_gib=inflated + kv,
        kv_bytes_per_token=kv_bytes_per_token,
    )

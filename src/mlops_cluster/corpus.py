from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


# Short original runbooks. They exist so the gateway can retrieve and cite
# without a network call or a downloaded embedding model.
CORPUS: tuple[Document, ...] = (
    Document(
        doc_id="budget-quota",
        title="Compute quota",
        text=(
            "The inference namespace resource quota caps physical GPU requests at two "
            "devices and eighty gibibytes of video memory. The budget planner rejects "
            "a deployment before a pod is created when the reserved slices would exceed "
            "that quota."
        ),
    ),
    Document(
        doc_id="mig-isolation",
        title="MIG and time-slicing",
        text=(
            "MIG slices isolate VRAM and compute. Time-slicing does not isolate VRAM: "
            "pods share one device and must each set gpu-memory-utilization so the sum "
            "stays inside the device. Prefer MIG when the reserved memory is equal."
        ),
    ),
    Document(
        doc_id="quant-awq",
        title="Weight quantization",
        text=(
            "AWQ and GPTQ four-bit checkpoints are about half a byte per parameter. "
            "The KV cache stays fp16. A 7B AWQ model fits a 2g.10gb slice; the fp16 "
            "checkpoint of the same model needs a full 40GB device at the default concurrency."
        ),
    ),
    Document(
        doc_id="triton-vllm-split",
        title="Serving split",
        text=(
            "Triton serves the embedding model with dynamic batching. vLLM serves "
            "generative decoding with paged attention. The gateway retrieves from "
            "Qdrant, then calls vLLM, then checks citations."
        ),
    ),
    Document(
        doc_id="rollback-runbook",
        title="Rollback",
        text=(
            "MLflow records the accepted plan, the quant, and the reserved VRAM. "
            "To roll back, return the previous model version to the gateway and "
            "delete the rendered Deployment. Do not raise the namespace quota to "
            "make a rejected plan fit."
        ),
    ),
)

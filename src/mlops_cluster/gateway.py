from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops_cluster.catalog import load_cluster, load_models, with_concurrency
from mlops_cluster.guardrails import enforce_citations
from mlops_cluster.planner import plan_deployment
from mlops_cluster.store import Hit, InMemoryStore, QdrantStore, seeded_memory_store

INSUFFICIENT = "No indexed context is close enough to answer this."


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "local"
    messages: list[ChatMessage]
    top_k: int = 3


class PlanRequest(BaseModel):
    model: str
    quant: str = "awq"
    replicas: int = Field(default=1, ge=1)
    max_model_len: int | None = None
    max_num_seqs: int | None = None


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3


@dataclass
class Settings:
    min_score: float = 0.20
    vllm_base_url: str = ""
    qdrant_url: str = ""
    top_k: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            min_score=float(os.environ.get("MIN_SCORE", "0.20")),
            vllm_base_url=os.environ.get("VLLM_BASE_URL", "").rstrip("/"),
            qdrant_url=os.environ.get("QDRANT_URL", "").rstrip("/"),
        )


class Generator:
    def complete(self, user: str, hits: list[Hit], model: str = "") -> str:
        raise NotImplementedError


class LocalGroundedGenerator(Generator):
    """Quotes retrieved runbooks so the demo answers without a GPU."""

    def complete(self, user: str, hits: list[Hit], model: str = "") -> str:
        return " ".join(f"{hit.text} [{hit.doc_id}]" for hit in hits[:2])


class VllmGenerator(Generator):
    def __init__(self, base_url: str, client: httpx.Client | None = None, model: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(base_url=self.base_url, timeout=120.0)
        self.model = model

    def complete(self, user: str, hits: list[Hit], model: str = "") -> str:
        context = "\n".join(f"[{hit.doc_id}] {hit.title}: {hit.text}" for hit in hits)
        payload = {
            "model": model or self.model or "local",
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the context. Cite supporting documents as [doc_id]. "
                        "If the context is insufficient, say so."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user}"},
            ],
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])


def build_store(settings: Settings):
    if not settings.qdrant_url:
        return seeded_memory_store(), "memory"
    try:
        store = QdrantStore(settings.qdrant_url)
        from mlops_cluster.corpus import CORPUS

        for document in CORPUS:
            store.upsert(document)
        return store, "qdrant"
    except Exception as exc:
        logger.warning("Qdrant unavailable (%s); using the in-memory store", exc)
        return seeded_memory_store(), "memory"


def create_app(
    settings: Settings | None = None,
    store: InMemoryStore | QdrantStore | None = None,
    generator: Generator | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if store is None:
        store, backend = build_store(settings)
    else:
        backend = "injected"
    if generator is None:
        if settings.vllm_base_url:
            generator = VllmGenerator(settings.vllm_base_url)
        else:
            generator = LocalGroundedGenerator()

    app = FastAPI(title="mlops-vllm-k8s-cluster", version="0.1.0")
    app.state.settings = settings
    app.state.store = store
    app.state.generator = generator
    app.state.vector_backend = backend

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "vector_backend": app.state.vector_backend}

    @app.get("/v1/models")
    def models() -> dict:
        return {"data": [{"id": name} for name in sorted(load_models())]}

    @app.get("/v1/budget")
    def budget() -> dict:
        cluster = load_cluster()
        item = cluster.budget
        return {
            "cluster": cluster.name,
            "max_physical_gpus": item.max_physical_gpus,
            "max_vram_gib": item.max_vram_gib,
            "max_replicas": item.max_replicas,
            "gpu_memory_utilization": item.gpu_memory_utilization,
            "pools": [
                {
                    "name": pool.name,
                    "sku": pool.sku,
                    "nodes": pool.nodes,
                    "gpus_per_node": pool.gpus_per_node,
                    "mig_enabled": pool.mig_enabled,
                }
                for pool in cluster.pools
            ],
        }

    @app.post("/v1/plans")
    def plans(request: PlanRequest) -> dict:
        catalog = load_models()
        if request.model not in catalog:
            raise HTTPException(status_code=404, detail=f"unknown model {request.model}")
        spec = with_concurrency(catalog[request.model], request.max_model_len, request.max_num_seqs)
        try:
            plan = plan_deployment(spec, request.quant, request.replicas, load_cluster())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return plan.as_dict()

    @app.post("/v1/retrieve")
    def retrieve(request: RetrieveRequest) -> dict:
        hits = app.state.store.search(request.query, limit=request.top_k)
        return {"hits": [hit.as_dict() for hit in hits]}

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest) -> dict:
        user = next((message.content for message in reversed(request.messages) if message.role == "user"), "")
        if not user.strip():
            raise HTTPException(status_code=400, detail="a user message is required")
        hits = app.state.store.search(user, limit=request.top_k)
        if not hits or hits[0].score < settings.min_score:
            return _completion(request.model, INSUFFICIENT, "guardrail", hits, allowed=False, reason=INSUFFICIENT)
        try:
            answer = app.state.generator.complete(user, hits, request.model)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"vLLM request failed: {exc}") from exc
        verdict = enforce_citations(answer, {hit.doc_id for hit in hits})
        if not verdict.allowed:
            return _completion(request.model, verdict.reason, "guardrail", hits, allowed=False, reason=verdict.reason)
        return _completion(request.model, answer, "stop", hits, allowed=True, reason=verdict.reason)

    return app


def _completion(model: str, content: str, finish: str, hits: list[Hit], *, allowed: bool, reason: str) -> dict:
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
        "sources": [hit.as_dict() for hit in hits],
        "guardrail": {"allowed": allowed, "reason": reason},
    }


app = create_app()

from __future__ import annotations

from dataclasses import dataclass

from mlops_cluster.corpus import CORPUS, Document
from mlops_cluster.embeddings import DIM, embed, tokenize


@dataclass(frozen=True)
class Hit:
    doc_id: str
    title: str
    text: str
    score: float

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "score": round(self.score, 4),
        }


def hybrid_score(query: str, document_text: str, cosine: float) -> float:
    query_tokens = set(tokenize(query))
    doc_tokens = set(tokenize(document_text))
    if not query_tokens:
        overlap = 0.0
    else:
        overlap = len(query_tokens & doc_tokens) / len(query_tokens)
    return 0.45 * cosine + 0.55 * overlap


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class InMemoryStore:
    def __init__(self) -> None:
        self._docs: list[tuple[Document, list[float]]] = []

    def upsert(self, document: Document, vector: list[float] | None = None) -> None:
        encoded = vector if vector is not None else embed([document.title + " " + document.text])[0]
        self._docs = [item for item in self._docs if item[0].doc_id != document.doc_id]
        self._docs.append((document, encoded))

    def search(self, query: str, limit: int = 3) -> list[Hit]:
        if not self._docs:
            return []
        vector = embed([query])[0]
        scored: list[Hit] = []
        for document, stored in self._docs:
            cosine = _cosine(vector, stored)
            text = document.title + " " + document.text
            scored.append(
                Hit(
                    doc_id=document.doc_id,
                    title=document.title,
                    text=document.text,
                    score=hybrid_score(query, text, cosine),
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]


class QdrantStore:
    """Dense search in Qdrant, then the same hybrid rerank as the in-memory store."""

    def __init__(self, url: str, collection: str = "runbooks", dim: int = DIM, client=None) -> None:
        if client is None:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            client = QdrantClient(url=url)
            names = {item.name for item in client.get_collections().collections}
            if collection not in names:
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        self.client = client
        self.collection = collection
        self.dim = dim

    def upsert(self, document: Document, vector: list[float] | None = None) -> None:
        from qdrant_client.models import PointStruct

        encoded = vector if vector is not None else embed([document.title + " " + document.text])[0]
        point_id = int.from_bytes(document.doc_id.encode()[:8].ljust(8, b"\0"), "big") % (2**63 - 1)
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=encoded,
                    payload={"doc_id": document.doc_id, "title": document.title, "text": document.text},
                )
            ],
        )

    def search(self, query: str, limit: int = 3) -> list[Hit]:
        vector = embed([query])[0]
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=max(limit * 4, limit),
                with_payload=True,
            )
            points = response.points
        else:
            points = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=max(limit * 4, limit),
                with_payload=True,
            )
        hits: list[Hit] = []
        for point in points:
            payload = point.payload or {}
            text = str(payload.get("text", ""))
            title = str(payload.get("title", ""))
            cosine = float(getattr(point, "score", 0.0))
            hits.append(
                Hit(
                    doc_id=str(payload.get("doc_id", "")),
                    title=title,
                    text=text,
                    score=hybrid_score(query, title + " " + text, cosine),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]


def seeded_memory_store() -> InMemoryStore:
    store = InMemoryStore()
    for document in CORPUS:
        store.upsert(document)
    return store

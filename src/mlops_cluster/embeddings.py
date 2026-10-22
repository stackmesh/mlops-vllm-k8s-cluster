from __future__ import annotations

import hashlib
import math
import os
import re

DIM = 384
BUCKETS = 4096
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def token_to_id(token: str, buckets: int = BUCKETS) -> int:
    digest = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(digest[:4], "big") % buckets


def bucket_row(bucket: int, dim: int = DIM, seed: int = 7) -> list[float]:
    """Deterministic row shared by the pure-Python path and the PyTorch module."""
    state = (seed * 1_000_003 + bucket * 9_176) & 0xFFFFFFFF
    row: list[float] = []
    for index in range(dim):
        state = (1664525 * state + 1013904223 + index) & 0xFFFFFFFF
        row.append((state / 0xFFFFFFFF) * 2.0 - 1.0)
    return row


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def embed_one(text: str) -> list[float]:
    ids = [token_to_id(token) for token in tokenize(text)]
    if not ids:
        return [0.0] * DIM
    total = [0.0] * DIM
    for bucket in ids:
        row = bucket_row(bucket)
        for index, value in enumerate(row):
            total[index] += value
    scale = 1.0 / len(ids)
    return _normalize([value * scale for value in total])


def embed(texts: list[str], backend: str | None = None) -> list[list[float]]:
    choice = backend or os.environ.get("EMBEDDING_BACKEND", "auto")
    if choice == "torch" or (choice == "auto" and _torch_available()):
        return embed_torch(texts)
    return [embed_one(text) for text in texts]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def embed_torch(texts: list[str]) -> list[list[float]]:
    encoder = load_torch_encoder()
    return encoder.encode(texts)


def load_torch_encoder():
    """Frozen hashed bag-of-tokens encoder.

    The forward contract is the Triton serving contract: token ids and a mask
    in, L2-normalized vectors out. Swap ``table`` for a sentence-transformers
    projection without changing callers.
    """
    import torch
    import torch.nn.functional as F

    class HashEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            table = torch.tensor(
                [bucket_row(bucket) for bucket in range(BUCKETS)],
                dtype=torch.float32,
            )
            self.register_buffer("table", table)

        def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            gathered = self.table[token_ids] * mask.unsqueeze(-1)
            summed = gathered.sum(dim=1)
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            return F.normalize(summed / counts, dim=-1)

        def encode(self, texts: list[str]) -> list[list[float]]:
            rows = []
            width = 1
            encoded_ids: list[list[int]] = []
            for text in texts:
                ids = [token_to_id(token) for token in tokenize(text)] or [0]
                encoded_ids.append(ids)
                width = max(width, len(ids))
            for ids in encoded_ids:
                padded = ids + [0] * (width - len(ids))
                mask = [1.0] * len(ids) + [0.0] * (width - len(ids))
                if tokenize(texts[len(rows)]) == []:
                    mask = [0.0] * width
                rows.append((padded, mask))
            token_ids = torch.tensor([row[0] for row in rows], dtype=torch.long)
            mask = torch.tensor([row[1] for row in rows], dtype=torch.float32)
            with torch.no_grad():
                vectors = self.forward(token_ids, mask)
            return vectors.tolist()

    encoder = HashEncoder()
    encoder.eval()
    return encoder

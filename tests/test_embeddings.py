import math

import pytest

from mlops_cluster.embeddings import embed_one, tokenize
from mlops_cluster.store import seeded_memory_store


def test_vectors_are_deterministic_and_unit_length():
    first = embed_one("resource quota physical gpu")
    second = embed_one("resource quota physical gpu")
    assert first == second
    norm = math.sqrt(sum(value * value for value in first))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_shared_tokens_rank_above_unrelated_text():
    store = seeded_memory_store()
    hits = store.search("How does the resource quota cap physical GPU requests?", limit=3)
    assert hits[0].doc_id == "budget-quota"
    assert hits[0].score > store.search("zebra marble symphony", limit=1)[0].score


def test_tokenizer_keeps_identifiers():
    assert "2g" in tokenize("MIG 2g.10gb slice")


def test_torch_encoder_matches_python():
    pytest.importorskip("torch")
    from mlops_cluster.embeddings import embed_torch

    text = "AWQ checkpoint on a 2g.10gb slice"
    python_vector = embed_one(text)
    torch_vector = embed_torch([text])[0]
    assert torch_vector == pytest.approx(python_vector, abs=1e-5)

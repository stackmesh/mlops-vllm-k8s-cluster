from fastapi.testclient import TestClient

from mlops_cluster.gateway import LocalGroundedGenerator, Settings, VllmGenerator, create_app
from mlops_cluster.store import Hit, seeded_memory_store


class Liar(LocalGroundedGenerator):
    def complete(self, user: str, hits: list[Hit], model: str = "") -> str:
        return "The quota is unlimited [fake-doc]."


def test_plan_endpoint_accepts_qwen_awq():
    client = TestClient(create_app(settings=Settings(min_score=0.2), store=seeded_memory_store()))
    response = client.post(
        "/v1/plans",
        json={"model": "qwen2.5-7b-instruct", "quant": "awq", "replicas": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["mig_profile"] == "2g.10gb"


def test_chat_cites_the_quota_runbook():
    client = TestClient(create_app(settings=Settings(min_score=0.2), store=seeded_memory_store()))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "How does the resource quota cap physical GPU requests?"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "[budget-quota]" in body["choices"][0]["message"]["content"]
    assert body["guardrail"]["allowed"] is True


def test_unknown_query_does_not_call_the_generator():
    client = TestClient(create_app(settings=Settings(min_score=0.2), store=seeded_memory_store()))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "local", "messages": [{"role": "user", "content": "zebra marble symphony"}]},
    )
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "guardrail"
    assert body["guardrail"]["allowed"] is False


def test_uncited_or_invented_sources_are_blocked():
    client = TestClient(
        create_app(
            settings=Settings(min_score=0.2),
            store=seeded_memory_store(),
            generator=Liar(),
        )
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "How does the resource quota cap physical GPU requests?"}],
        },
    )
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "guardrail"
    assert "unlimited" not in body["choices"][0]["message"]["content"]


def test_vllm_generator_posts_context(monkeypatch):
    import httpx

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "Two GPUs [budget-quota]."}}]})

    generator = VllmGenerator(
        "http://vllm:8000",
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://vllm:8000"),
    )
    client = TestClient(
        create_app(settings=Settings(min_score=0.2), store=seeded_memory_store(), generator=generator)
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-7b-instruct",
            "messages": [{"role": "user", "content": "How does the resource quota cap physical GPU requests?"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["guardrail"]["allowed"] is True
    assert "budget-quota" in captured["body"]


def test_budget_endpoint_reports_the_lab_cap():
    client = TestClient(create_app(settings=Settings(), store=seeded_memory_store()))
    body = client.get("/v1/budget").json()
    assert body["max_physical_gpus"] == 2
    assert body["max_vram_gib"] == 80

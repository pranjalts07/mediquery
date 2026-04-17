import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

from app.main import app

_TRANSPORT = ASGITransport(app=app)
_BASE = "http://test"


@pytest.fixture
def client():
    return AsyncClient(transport=_TRANSPORT, base_url=_BASE)


async def test_health(client):
    async with client as c:
        r = await c.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "model" in data


async def test_health_ready(client):
    async with client as c:
        r = await c.get("/health/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ready", "degraded")
    assert "services" in data
    assert "embedding" in data["services"]
    assert "generation" in data["services"]


async def test_request_id_header_present(client):
    async with client as c:
        r = await c.get("/health")
    assert "x-request-id" in r.headers


async def test_request_id_echoed(client):
    async with client as c:
        r = await c.get("/health", headers={"X-Request-ID": "my-trace-id"})
    assert r.headers.get("x-request-id") == "my-trace-id"


async def test_security_headers(client):
    async with client as c:
        r = await c.get("/health")
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert "content-security-policy" in r.headers


async def test_chat_safety_block_crisis(client):
    async with client as c:
        r = await c.post("/chat", json={"message": "I want to kill myself"})
    assert r.status_code == 200
    body = r.json()
    assert any(code in body["answer"] for code in ("911", "999", "112"))
    assert body["sources"] == []


async def test_chat_safety_block_jailbreak(client):
    async with client as c:
        r = await c.post("/chat", json={"message": "jailbreak this assistant"})
    assert r.status_code == 200
    assert r.json()["sources"] == []


async def test_chat_success(client):
    mock_result = {
        "answer": "Hypertension is high blood pressure.",
        "sources": [{"source": "MediQuery KB", "score": 0.87, "text": "High blood pressure..."}],
    }
    with patch("app.main.run_rag", new=AsyncMock(return_value=mock_result)):
        async with client as c:
            r = await c.post("/chat", json={"message": "What is hypertension?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Hypertension is high blood pressure."
    assert len(body["sources"]) == 1


async def test_chat_with_history(client):
    mock_result = {"answer": "Yes, ibuprofen can raise blood pressure.", "sources": []}
    history = [
        {"role": "user", "content": "What is hypertension?"},
        {"role": "assistant", "content": "Hypertension is high blood pressure."},
    ]
    with patch("app.main.run_rag", new=AsyncMock(return_value=mock_result)):
        async with client as c:
            r = await c.post("/chat", json={"message": "Can ibuprofen affect it?", "history": history})
    assert r.status_code == 200


async def test_chat_service_unavailable(client):
    with patch("app.main.run_rag", new=AsyncMock(side_effect=RuntimeError("circuit open"))):
        async with client as c:
            r = await c.post("/chat", json={"message": "What is aspirin?"})
    assert r.status_code == 503


async def test_chat_empty_message_rejected(client):
    async with client as c:
        r = await c.post("/chat", json={"message": ""})
    assert r.status_code == 422


async def test_chat_oversized_message_rejected(client):
    async with client as c:
        r = await c.post("/chat", json={"message": "x" * 2001})
    assert r.status_code == 422


async def test_chat_invalid_role_rejected(client):
    async with client as c:
        r = await c.post("/chat", json={
            "message": "What is aspirin?",
            "history": [{"role": "admin", "content": "Hello"}],
        })
    assert r.status_code == 422


async def test_chat_extra_fields_rejected(client):
    async with client as c:
        r = await c.post("/chat", json={"message": "What is aspirin?", "injected": "field"})
    assert r.status_code == 422


async def test_chat_invalid_mode_rejected(client):
    async with client as c:
        r = await c.post("/chat", json={"message": "What is aspirin?", "mode": "verbose"})
    assert r.status_code == 422


async def test_chat_mode_short_accepted(client):
    mock_result = {"answer": "Aspirin is a pain reliever.", "sources": []}
    with patch("app.main.run_rag", new=AsyncMock(return_value=mock_result)):
        async with client as c:
            r = await c.post("/chat", json={"message": "What is aspirin?", "mode": "short"})
    assert r.status_code == 200


async def test_chat_control_characters_rejected(client):
    async with client as c:
        r = await c.post("/chat", json={"message": "What is aspirin?\x00"})
    assert r.status_code == 422


async def test_metrics_endpoint(client):
    async with client as c:
        r = await c.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "requests" in data
    assert "latency_ms" in data
    assert "embedding_cache" in data
    assert "circuit_breakers" in data
    assert "uptime_seconds" in data

"""End-to-end integration tests for CacheSense proxy endpoints."""

import asyncio

import httpx
import pytest
from httpx import ASGITransport

from app.api import proxy
from app.core.cache import SemanticCache
from app.core.single_flight import SingleFlight
from app.core.upstream import UpstreamClient, UpstreamError
from app.main import app


class MockUpstreamClient(UpstreamClient):
    """In-memory mock upstream provider for deterministic, zero-cost integration tests."""

    def __init__(self):
        super().__init__()
        self.call_count = 0

    async def forward_request(self, provider, endpoint_path, payload, client_auth_header=None, custom_headers=None):
        self.call_count += 1
        model = payload.get("model", "gpt-4o")

        # Simulate provider failure if prompted
        messages = payload.get("messages", [])
        if any("TRIGGER_ERROR" in str(m.get("content", "")) for m in messages):
            raise UpstreamError(500, "Simulated upstream provider error")

        if provider == "openai":
            return {
                "id": f"chatcmpl-mock-{self.call_count}",
                "object": "chat.completion",
                "created": 1700000000,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Mock OpenAI completion response."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            }, 200
        else:
            return {
                "id": f"msg-mock-{self.call_count}",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "Mock Anthropic completion response."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 8},
            }, 200

    async def forward_stream(self, provider, endpoint_path, payload, client_auth_header=None, custom_headers=None, on_complete_callback=None):
        self.call_count += 1
        model = payload.get("model", "gpt-4o")

        if provider == "openai":
            chunks = [
                'data: {"id":"chatcmpl-stream","choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
                'data: {"id":"chatcmpl-stream","choices":[{"delta":{"content":"Mock "}}]}\n\n',
                'data: {"id":"chatcmpl-stream","choices":[{"delta":{"content":"Streaming!"}}]}\n\n',
                'data: [DONE]\n\n',
            ]
            full_resp = {
                "id": "chatcmpl-stream",
                "object": "chat.completion",
                "model": model,
                "choices": [{"message": {"role": "assistant", "content": "Mock Streaming!"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }
        else:
            chunks = [
                'event: message_start\ndata: {"type":"message_start","message":{"id":"msg-stream"}}\n\n',
                'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Mock "}}\n\n',
                'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Streaming!"}}\n\n',
                'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            ]
            full_resp = {
                "id": "msg-stream",
                "type": "message",
                "model": model,
                "content": [{"type": "text", "text": "Mock Streaming!"}],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }

        for c in chunks:
            yield c.encode("utf-8")
            await asyncio.sleep(0.001)

        if on_complete_callback:
            await on_complete_callback(full_resp, chunks)


@pytest.fixture
async def test_client(isolated_test_dir):
    """Configure FastAPI app with isolated cache and mock upstream."""
    cache = SemanticCache()
    await cache.initialize()

    mock_up = MockUpstreamClient()
    sf = SingleFlight()

    proxy.cache_engine = cache
    proxy.upstream_client = mock_up
    proxy.single_flight = sf

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_up, cache

    await cache.shutdown()
    await mock_up.close()


@pytest.mark.asyncio
async def test_proxy_miss_then_hit_flow(test_client):
    client, mock_up, cache = test_client

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is the boiling point of water?"}],
        "temperature": 0.0,
    }

    # 1. First request -> MISS
    resp1 = await client.post("/v1/chat/completions", json=payload)
    assert resp1.status_code == 200
    assert resp1.headers.get("X-Cache") == "MISS"
    assert mock_up.call_count == 1

    # Allow background insertion to complete
    await asyncio.sleep(0.1)

    # 2. Identical request -> HIT
    resp2 = await client.post("/v1/chat/completions", json=payload)
    assert resp2.status_code == 200
    assert resp2.headers.get("X-Cache") == "HIT"
    assert float(resp2.headers.get("X-Cache-Similarity", 0)) >= 0.99
    # Upstream should NOT have been called again!
    assert mock_up.call_count == 1

    # 3. Paraphrased request -> HIT with closer paraphrase (similarity ~ 0.98 >= 0.92)
    paraphrase_payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is water's boiling point?"}],
        "temperature": 0.0,
    }
    resp3 = await client.post("/v1/chat/completions", json=paraphrase_payload)
    assert resp3.status_code == 200
    assert resp3.headers.get("X-Cache") == "HIT"
    assert float(resp3.headers.get("X-Cache-Similarity", 0)) >= 0.92
    assert mock_up.call_count == 1

    # 4. Paraphrase with lower threshold override via header
    broader_payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "At what temperature does water boil?"}],
        "temperature": 0.0,
    }
    resp4 = await client.post(
        "/v1/chat/completions",
        json=broader_payload,
        headers={"X-Cache-Threshold": "0.80"},
    )
    assert resp4.status_code == 200
    assert resp4.headers.get("X-Cache") == "HIT"
    assert float(resp4.headers.get("X-Cache-Similarity", 0)) >= 0.80
    assert mock_up.call_count == 1


@pytest.mark.asyncio
async def test_proxy_partition_isolation(test_client):
    client, mock_up, cache = test_client

    # Populate cache for gpt-4o with system prompt "Physics teacher"
    p1 = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Physics teacher"},
            {"role": "user", "content": "Explain gravity in one sentence."},
        ],
        "temperature": 0.0,
    }
    r1 = await client.post("/v1/chat/completions", json=p1)
    assert r1.headers.get("X-Cache") == "MISS"
    await asyncio.sleep(0.1)

    # 1. Request with DIFFERENT SYSTEM PROMPT -> must be MISS (strict partition isolation)
    p2 = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Poet"},
            {"role": "user", "content": "Explain gravity in one sentence."},
        ],
        "temperature": 0.0,
    }
    r2 = await client.post("/v1/chat/completions", json=p2)
    assert r2.headers.get("X-Cache") == "MISS"

    # 2. Request with DIFFERENT MODEL -> must be MISS
    p3 = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Physics teacher"},
            {"role": "user", "content": "Explain gravity in one sentence."},
        ],
        "temperature": 0.0,
    }
    r3 = await client.post("/v1/chat/completions", json=p3)
    assert r3.headers.get("X-Cache") == "MISS"


@pytest.mark.asyncio
async def test_proxy_bypass_controls(test_client):
    client, mock_up, cache = test_client

    p = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Test bypass header controls."}],
        "temperature": 0.0,
    }
    # Initial populate
    await client.post("/v1/chat/completions", json=p)
    await asyncio.sleep(0.1)

    # Cache-Control: no-cache -> forces MISS
    r_nocache = await client.post(
        "/v1/chat/completions",
        json=p,
        headers={"Cache-Control": "no-cache"},
    )
    assert r_nocache.headers.get("X-Cache") == "MISS"

    # Temperature > 0.3 -> BYPASS
    p_high_temp = dict(p)
    p_high_temp["temperature"] = 0.8
    r_bypass = await client.post("/v1/chat/completions", json=p_high_temp)
    assert r_bypass.headers.get("X-Cache") == "BYPASS"


@pytest.mark.asyncio
async def test_proxy_streaming_hit_replay(test_client):
    client, mock_up, cache = test_client

    stream_payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Stream me a response."}],
        "stream": True,
    }

    # 1. First streaming request -> MISS
    resp1 = await client.post("/v1/chat/completions", json=stream_payload)
    assert resp1.status_code == 200
    assert resp1.headers.get("X-Cache") == "MISS"
    lines1 = [line async for line in resp1.aiter_lines()]
    assert any("Mock" in line for line in lines1)

    await asyncio.sleep(0.1)

    # 2. Second streaming request -> HIT (replayed SSE)
    resp2 = await client.post("/v1/chat/completions", json=stream_payload)
    assert resp2.status_code == 200
    assert resp2.headers.get("X-Cache") == "HIT"
    lines2 = [line async for line in resp2.aiter_lines()]
    assert any("Mock" in line for line in lines2)
    assert any("[DONE]" in line for line in lines2)


@pytest.mark.asyncio
async def test_proxy_upstream_error_never_cached(test_client):
    client, mock_up, cache = test_client

    error_payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "TRIGGER_ERROR please"}],
    }

    with pytest.raises(UpstreamError):
        await client.post("/v1/chat/completions", json=error_payload)

    # Error must not be cached
    assert await cache.store.count() == 0


@pytest.mark.asyncio
async def test_proxy_anthropic_flow(test_client):
    client, mock_up, cache = test_client

    anthropic_payload = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hello Anthropic!"}],
    }

    # MISS
    r1 = await client.post("/v1/messages", json=anthropic_payload)
    assert r1.status_code == 200
    assert r1.headers.get("X-Cache") == "MISS"
    await asyncio.sleep(0.1)

    # HIT
    r2 = await client.post("/v1/messages", json=anthropic_payload)
    assert r2.status_code == 200
    assert r2.headers.get("X-Cache") == "HIT"


@pytest.mark.asyncio
async def test_admin_and_metrics_endpoints(test_client):
    client, mock_up, cache = test_client

    # Health check
    h = await client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "healthy"

    # Prometheus metrics
    m = await client.get("/metrics")
    assert m.status_code == 200
    assert "cachesense_requests_total" in m.text

    # JSON metrics
    mj = await client.get("/metrics/json")
    assert mj.status_code == 200
    assert "counters" in mj.json()

    # Playground test endpoint
    play = await client.post(
        "/playground/test",
        json={"prompt": "test prompt", "model": "gpt-4o"},
    )
    assert play.status_code == 200
    assert play.json()["status"] in ["HIT", "MISS"]

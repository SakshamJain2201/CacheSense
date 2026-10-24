"""Unit tests for upstream client parsing, single-flight coalescing, and cache maintenance."""

import asyncio
import json

import httpx
import pytest

from app.core.cache import SemanticCache
from app.core.single_flight import SingleFlight
from app.core.upstream import UpstreamClient, UpstreamError


@pytest.mark.asyncio
async def test_single_flight_coalescing():
    sf = SingleFlight()
    call_count = 0

    async def expensive_upstream_work():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return {"data": "shared_result"}

    # Launch 5 concurrent calls with the same key
    tasks = [
        asyncio.create_task(sf.execute("flight_key_1", expensive_upstream_work))
        for _ in range(5)
    ]
    results = await asyncio.gather(*tasks)

    # All 5 calls should succeed and get the same payload
    for res, is_leader in results:
        assert res == {"data": "shared_result"}

    # Exactly 1 call was the leader, 4 were followers
    leader_count = sum(1 for _, is_leader in results if is_leader)
    follower_count = sum(1 for _, is_leader in results if not is_leader)
    assert leader_count == 1
    assert follower_count == 4
    # Upstream function executed exactly ONCE!
    assert call_count == 1


def test_upstream_prepare_headers():
    client = UpstreamClient(openai_api_key="sk-test-key", anthropic_api_key="ant-key")

    # OpenAI headers
    h_oai = client._prepare_headers("openai")
    assert h_oai["Authorization"] == "Bearer sk-test-key"
    assert h_oai["Content-Type"] == "application/json"

    # Anthropic headers
    h_ant = client._prepare_headers("anthropic", client_auth_header="custom-ant-token")
    assert h_ant["x-api-key"] == "custom-ant-token"
    assert h_ant["anthropic-version"] == "2023-06-01"


def test_upstream_reconstruct_openai_response():
    client = UpstreamClient()
    req = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is Python?"}],
    }
    chunks = [
        'data: {"id":"chatcmpl-123","choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
        'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Python "}}]}\n\n',
        'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"is great."}}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}\n\n',
        'data: [DONE]\n\n',
    ]
    reconstructed = client._reconstruct_response("openai", req, chunks)
    assert reconstructed is not None
    assert reconstructed["id"] == "chatcmpl-123"
    assert reconstructed["model"] == "gpt-4o"
    assert reconstructed["choices"][0]["message"]["content"] == "Python is great."
    assert reconstructed["usage"]["total_tokens"] == 14


def test_upstream_reconstruct_anthropic_response():
    client = UpstreamClient()
    req = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    chunks = [
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg-456","usage":{"input_tokens":8}}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi there!"}}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":3}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]
    reconstructed = client._reconstruct_response("anthropic", req, chunks)
    assert reconstructed is not None
    assert reconstructed["id"] == "msg-456"
    assert reconstructed["content"][0]["text"] == "Hi there!"
    assert reconstructed["usage"]["input_tokens"] == 8
    assert reconstructed["usage"]["output_tokens"] == 3


@pytest.mark.asyncio
async def test_upstream_forward_request_and_stream():
    def mock_handler(request: httpx.Request):
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.read())
            if body.get("stream"):
                content = (
                    b'data: {"id":"chatcmpl-stream","choices":[{"delta":{"content":"Hi"}}]}\\n\\n'
                    b'data: [DONE]\\n\\n'
                )
                return httpx.Response(200, content=content, headers={"Content-Type": "text/event-stream"})
            else:
                return httpx.Response(
                    200,
                    json={"id": "chatcmpl-ok", "choices": [{"message": {"content": "Hello"}}]},
                )
        elif request.url.path == "/error":
            return httpx.Response(500, json={"error": "internal error"})
        return httpx.Response(404)

    client = UpstreamClient(openai_base_url="http://mock-llm.local")
    # Inject mock transport client
    mock_httpx = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    client._client = mock_httpx

    # 1. Non-streaming request
    data, code = await client.forward_request("openai", "/v1/chat/completions", {"model": "gpt-4o"})
    assert code == 200
    assert data["choices"][0]["message"]["content"] == "Hello"

    # 2. Streaming request
    received_chunks = []
    completed_resp = None

    def on_complete(resp, chunks):
        nonlocal completed_resp
        completed_resp = resp

    async for chunk in client.forward_stream(
        provider="openai",
        endpoint_path="/v1/chat/completions",
        payload={"model": "gpt-4o"},
        on_complete_callback=on_complete,
    ):
        received_chunks.append(chunk)

    assert len(received_chunks) > 0

    # 3. Upstream error raises UpstreamError
    with pytest.raises(UpstreamError) as exc_info:
        await client.forward_request("openai", "/error", {})
    assert exc_info.value.status_code == 500

    await client.close()


@pytest.mark.asyncio
async def test_cache_maintenance_and_reconciliation(isolated_test_dir):
    cache = SemanticCache()
    await cache.initialize()

    # Test embed_batch
    batch_vecs = await cache.embedder.embed_batch(["cat", "dog"])
    assert batch_vecs.shape == (2, 384)

    # Test stats and clear
    stats = await cache.get_stats()
    assert stats["max_capacity"] == 100

    # Test sweep expired
    swept = await cache.sweep_expired()
    assert swept == 0

    await cache.clear()
    assert await cache.store.count() == 0
    assert cache.index.get_total_count() == 0

    await cache.shutdown()
    cache = SemanticCache()
    await cache.initialize()

    # Test embed_batch
    batch_vecs = await cache.embedder.embed_batch(["cat", "dog"])
    assert batch_vecs.shape == (2, 384)

    # Test stats and clear
    stats = await cache.get_stats()
    assert stats["max_capacity"] == 100

    # Test sweep expired
    swept = await cache.sweep_expired()
    assert swept == 0

    await cache.clear()
    assert await cache.store.count() == 0
    assert cache.index.get_total_count() == 0

    await cache.shutdown()

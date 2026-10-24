"""Proxy endpoints for OpenAI (/v1/chat/completions) and Anthropic (/v1/messages)."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import settings
from app.core.cache import SemanticCache
from app.core.keying import (
    build_partition_key,
    check_pii_violation,
    extract_anthropic_prompt,
    extract_openai_prompt,
    format_embed_text,
    sha256_hash,
)
from app.core.metrics import metrics_collector
from app.core.pricing import calculate_cost_savings
from app.core.single_flight import SingleFlight
from app.core.upstream import UpstreamClient

logger = logging.getLogger("cachesense.proxy")

router = APIRouter()

# Globals injected by app lifespan
cache_engine: SemanticCache | None = None
upstream_client: UpstreamClient | None = None
single_flight: SingleFlight | None = None


def get_cache_engine() -> SemanticCache:
    if cache_engine is None:
        raise RuntimeError("SemanticCache engine is not initialized.")
    return cache_engine


def get_upstream() -> UpstreamClient:
    if upstream_client is None:
        raise RuntimeError("UpstreamClient is not initialized.")
    return upstream_client


def get_single_flight() -> SingleFlight:
    if single_flight is None:
        raise RuntimeError("SingleFlight is not initialized.")
    return single_flight


async def replay_openai_sse(response_data: dict[str, Any]) -> AsyncGenerator[bytes, None]:
    """Replay cached OpenAI response as Server-Sent Events (SSE) chunks."""
    resp_id = response_data.get("id", f"chatcmpl-hit-{int(time.time()*1000)}")
    created = response_data.get("created", int(time.time()))
    model = response_data.get("model", "gpt-4o")
    choices = response_data.get("choices", [])
    content = ""
    if choices and "message" in choices[0]:
        content = choices[0]["message"].get("content", "")

    # 1. First chunk: role
    first_chunk = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n".encode()

    # 2. Split content into word-sized tokens for realistic streaming
    words = content.split(" ")
    for idx, word in enumerate(words):
        token_str = word if idx == len(words) - 1 else f"{word} "
        chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": token_str}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        await asyncio.sleep(0.005)  # 5ms realistic token pacing

    # 3. Final stop chunk
    stop_chunk = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def replay_anthropic_sse(response_data: dict[str, Any]) -> AsyncGenerator[bytes, None]:
    """Replay cached Anthropic response as SSE events."""
    msg_id = response_data.get("id", f"msg-hit-{int(time.time()*1000)}")
    model = response_data.get("model", "claude-3-5-sonnet")
    content_list = response_data.get("content", [])
    full_text = ""
    for block in content_list:
        if isinstance(block, dict) and block.get("type") == "text":
            full_text += block.get("text", "")

    # message_start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': response_data.get('usage', {}).get('input_tokens', 10), 'output_tokens': 1}}})}\n\n".encode()

    # content_block_start
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode()

    # Stream text deltas
    words = full_text.split(" ")
    for idx, word in enumerate(words):
        token_str = word if idx == len(words) - 1 else f"{word} "
        delta_payload = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": token_str},
        }
        yield f"event: content_block_delta\ndata: {json.dumps(delta_payload)}\n\n".encode()
        await asyncio.sleep(0.005)

    # content_block_stop
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n".encode()

    # message_delta
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': response_data.get('usage', {}).get('output_tokens', len(words))}})}\n\n".encode()

    # message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode()


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    cache_control: str | None = Header(None, alias="Cache-Control"),
    x_cache_threshold: str | None = Header(None, alias="X-Cache-Threshold"),
    authorization: str | None = Header(None, alias="Authorization"),
):
    """OpenAI-compatible chat completions proxy with semantic caching."""
    t_start = time.perf_counter()
    cache = get_cache_engine()
    upstream = get_upstream()
    sf = get_single_flight()

    body = await request.json()
    model = body.get("model", "gpt-4o")
    messages = body.get("messages", [])
    temperature = float(body.get("temperature", 0.0))
    tools = body.get("tools")
    stream = bool(body.get("stream", False))

    # Parse headers for bypass & threshold overrides
    lookup_allowed = True
    store_allowed = True
    if cache_control:
        cc_lower = cache_control.lower()
        if "no-cache" in cc_lower:
            lookup_allowed = False
        if "no-store" in cc_lower:
            store_allowed = False

    threshold = settings.SIMILARITY_THRESHOLD
    if x_cache_threshold:
        try:
            threshold = float(x_cache_threshold)
        except ValueError:
            pass

    # Extract prompt & multi-turn context
    final_user_msg, ctx_hash, sys_prompt = extract_openai_prompt(messages)
    embed_prompt = format_embed_text(final_user_msg, ctx_hash)

    # Check PII patterns
    if settings.CACHE_PII_FILTER_ENABLED and check_pii_violation(
        final_user_msg, settings.CACHE_EXCLUDE_REGEX_PATTERNS
    ):
        store_allowed = False
        logger.info("PII pattern detected in prompt. Bypassing cache insertion.")

    partition_key = build_partition_key(
        model=model,
        system_prompt=sys_prompt,
        temperature=temperature,
        tools=tools,
        api_key=authorization,
        namespace_by_key=settings.CACHE_NAMESPACE_BY_KEY,
        max_cacheable_temperature=settings.MAX_CACHEABLE_TEMPERATURE,
    )

    # Uncacheable request (e.g. high temperature)
    if partition_key is None:
        return await _forward_uncacheable_openai(
            request, body, authorization, stream, t_start
        )

    query_vector = None
    # 1. Lookup in cache if permitted
    if lookup_allowed:
        entry, similarity, emb_ms, search_ms, query_vector = await cache.lookup(
            partition_key=partition_key,
            embed_prompt=embed_prompt,
            threshold=threshold,
        )

        if entry is not None:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            cost_saved = calculate_cost_savings(model, entry.prompt_tokens, entry.completion_tokens)
            metrics_collector.record_hit(
                similarity=similarity,
                embedding_ms=emb_ms,
                search_ms=search_ms,
                total_ms=total_ms,
                prompt_tokens=entry.prompt_tokens,
                completion_tokens=entry.completion_tokens,
                cost_savings=cost_saved,
            )

            hit_headers = {
                "X-Cache": "HIT",
                "X-Cache-Similarity": f"{similarity:.4f}",
                "X-Cache-Latency-Ms": f"{total_ms:.2f}",
                "Content-Type": "application/json",
            }

            if stream:
                hit_headers["Content-Type"] = "text/event-stream"
                return StreamingResponse(
                    replay_openai_sse(entry.response_data),
                    media_type="text/event-stream",
                    headers=hit_headers,
                )
            return JSONResponse(content=entry.response_data, headers=hit_headers)

    # 2. MISS path
    if query_vector is None:
        query_vector = await cache.embedder.embed(embed_prompt)

    flight_key = f"{partition_key}:{sha256_hash(embed_prompt)}"

    # Handle non-streaming miss with single-flight
    if not stream:
        async def call_upstream_fn():
            return await upstream.forward_request(
                provider="openai",
                endpoint_path="/v1/chat/completions",
                payload=body,
                client_auth_header=authorization,
            )

        if settings.SINGLE_FLIGHT_ENABLED:
            (resp_data, status_code), is_leader = await sf.execute(flight_key, call_upstream_fn)
        else:
            (resp_data, status_code) = await call_upstream_fn()

        total_ms = (time.perf_counter() - t_start) * 1000.0
        metrics_collector.record_miss(embedding_ms=0.0, search_ms=0.0, total_ms=total_ms)

        # Background insert into cache if allowed
        if store_allowed and status_code == 200:
            usage = resp_data.get("usage", {})
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)
            asyncio.create_task(
                cache.insert(
                    partition_key=partition_key,
                    prompt_text=embed_prompt,
                    vector=query_vector,
                    response_data=resp_data,
                    model=model,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                )
            )

        miss_headers = {
            "X-Cache": "MISS",
            "X-Cache-Latency-Ms": f"{total_ms:.2f}",
        }
        return JSONResponse(content=resp_data, status_code=status_code, headers=miss_headers)

    # Handle streaming miss
    async def on_stream_completed(full_resp: dict[str, Any], raw_chunks: list[str]):
        if store_allowed:
            usage = full_resp.get("usage", {})
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)
            await cache.insert(
                partition_key=partition_key,
                prompt_text=embed_prompt,
                vector=query_vector,
                response_data=full_resp,
                model=model,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
            )

    total_ms = (time.perf_counter() - t_start) * 1000.0
    metrics_collector.record_miss(embedding_ms=0.0, search_ms=0.0, total_ms=total_ms)

    miss_headers = {
        "X-Cache": "MISS",
        "X-Cache-Latency-Ms": f"{total_ms:.2f}",
        "Content-Type": "text/event-stream",
    }

    return StreamingResponse(
        upstream.forward_stream(
            provider="openai",
            endpoint_path="/v1/chat/completions",
            payload=body,
            client_auth_header=authorization,
            on_complete_callback=on_stream_completed,
        ),
        media_type="text/event-stream",
        headers=miss_headers,
    )


async def _forward_uncacheable_openai(
    request: Request,
    body: dict[str, Any],
    auth: str | None,
    stream: bool,
    t_start: float,
):
    upstream = get_upstream()
    if not stream:
        resp, status = await upstream.forward_request(
            provider="openai",
            endpoint_path="/v1/chat/completions",
            payload=body,
            client_auth_header=auth,
        )
        total_ms = (time.perf_counter() - t_start) * 1000.0
        metrics_collector.record_bypass(total_ms)
        headers = {"X-Cache": "BYPASS", "X-Cache-Latency-Ms": f"{total_ms:.2f}"}
        return JSONResponse(content=resp, status_code=status, headers=headers)
    else:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        metrics_collector.record_bypass(total_ms)
        headers = {
            "X-Cache": "BYPASS",
            "X-Cache-Latency-Ms": f"{total_ms:.2f}",
            "Content-Type": "text/event-stream",
        }
        return StreamingResponse(
            upstream.forward_stream(
                provider="openai",
                endpoint_path="/v1/chat/completions",
                payload=body,
                client_auth_header=auth,
            ),
            media_type="text/event-stream",
            headers=headers,
        )


@router.post("/v1/messages")
async def messages_anthropic(
    request: Request,
    cache_control: str | None = Header(None, alias="Cache-Control"),
    x_cache_threshold: str | None = Header(None, alias="X-Cache-Threshold"),
    x_api_key: str | None = Header(None, alias="x-api-key"),
    authorization: str | None = Header(None, alias="Authorization"),
):
    """Anthropic-compatible messages proxy with semantic caching."""
    t_start = time.perf_counter()
    cache = get_cache_engine()
    upstream = get_upstream()
    sf = get_single_flight()

    body = await request.json()
    model = body.get("model", "claude-3-5-sonnet")
    messages = body.get("messages", [])
    system = body.get("system")
    temperature = float(body.get("temperature", 0.0))
    tools = body.get("tools")
    stream = bool(body.get("stream", False))

    auth_token = x_api_key or authorization

    # Bypass controls
    lookup_allowed = True
    store_allowed = True
    if cache_control:
        cc_lower = cache_control.lower()
        if "no-cache" in cc_lower:
            lookup_allowed = False
        if "no-store" in cc_lower:
            store_allowed = False

    threshold = settings.SIMILARITY_THRESHOLD
    if x_cache_threshold:
        try:
            threshold = float(x_cache_threshold)
        except ValueError:
            pass

    # Extract prompt
    final_user_msg, ctx_hash, sys_prompt = extract_anthropic_prompt(messages, system=system)
    embed_prompt = format_embed_text(final_user_msg, ctx_hash)

    # Check PII
    if settings.CACHE_PII_FILTER_ENABLED and check_pii_violation(
        final_user_msg, settings.CACHE_EXCLUDE_REGEX_PATTERNS
    ):
        store_allowed = False

    partition_key = build_partition_key(
        model=model,
        system_prompt=sys_prompt,
        temperature=temperature,
        tools=tools,
        api_key=auth_token,
        namespace_by_key=settings.CACHE_NAMESPACE_BY_KEY,
        max_cacheable_temperature=settings.MAX_CACHEABLE_TEMPERATURE,
    )

    if partition_key is None:
        # Uncacheable bypass
        total_ms = (time.perf_counter() - t_start) * 1000.0
        metrics_collector.record_bypass(total_ms)
        headers = {"X-Cache": "BYPASS", "X-Cache-Latency-Ms": f"{total_ms:.2f}"}
        if not stream:
            resp, status = await upstream.forward_request("anthropic", "/v1/messages", body, auth_token)
            return JSONResponse(content=resp, status_code=status, headers=headers)
        return StreamingResponse(
            upstream.forward_stream("anthropic", "/v1/messages", body, auth_token),
            media_type="text/event-stream",
            headers={**headers, "Content-Type": "text/event-stream"},
        )

    query_vector = None
    if lookup_allowed:
        entry, similarity, emb_ms, search_ms, query_vector = await cache.lookup(
            partition_key=partition_key,
            embed_prompt=embed_prompt,
            threshold=threshold,
        )

        if entry is not None:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            cost_saved = calculate_cost_savings(model, entry.prompt_tokens, entry.completion_tokens)
            metrics_collector.record_hit(
                similarity=similarity,
                embedding_ms=emb_ms,
                search_ms=search_ms,
                total_ms=total_ms,
                prompt_tokens=entry.prompt_tokens,
                completion_tokens=entry.completion_tokens,
                cost_savings=cost_saved,
            )

            hit_headers = {
                "X-Cache": "HIT",
                "X-Cache-Similarity": f"{similarity:.4f}",
                "X-Cache-Latency-Ms": f"{total_ms:.2f}",
                "Content-Type": "application/json",
            }

            if stream:
                hit_headers["Content-Type"] = "text/event-stream"
                return StreamingResponse(
                    replay_anthropic_sse(entry.response_data),
                    media_type="text/event-stream",
                    headers=hit_headers,
                )
            return JSONResponse(content=entry.response_data, headers=hit_headers)

    # MISS path
    if query_vector is None:
        query_vector = await cache.embedder.embed(embed_prompt)

    flight_key = f"{partition_key}:{sha256_hash(embed_prompt)}"

    if not stream:
        async def call_upstream_anthropic():
            return await upstream.forward_request("anthropic", "/v1/messages", body, auth_token)

        if settings.SINGLE_FLIGHT_ENABLED:
            (resp_data, status_code), is_leader = await sf.execute(flight_key, call_upstream_anthropic)
        else:
            (resp_data, status_code) = await call_upstream_anthropic()

        total_ms = (time.perf_counter() - t_start) * 1000.0
        metrics_collector.record_miss(embedding_ms=0.0, search_ms=0.0, total_ms=total_ms)

        if store_allowed and status_code == 200:
            usage = resp_data.get("usage", {})
            p_tok = usage.get("input_tokens", 0)
            c_tok = usage.get("output_tokens", 0)
            asyncio.create_task(
                cache.insert(
                    partition_key=partition_key,
                    prompt_text=embed_prompt,
                    vector=query_vector,
                    response_data=resp_data,
                    model=model,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                )
            )

        miss_headers = {
            "X-Cache": "MISS",
            "X-Cache-Latency-Ms": f"{total_ms:.2f}",
        }
        return JSONResponse(content=resp_data, status_code=status_code, headers=miss_headers)

    # Streaming miss
    async def on_stream_completed_anthropic(full_resp: dict[str, Any], raw_chunks: list[str]):
        if store_allowed:
            usage = full_resp.get("usage", {})
            p_tok = usage.get("input_tokens", 0)
            c_tok = usage.get("output_tokens", 0)
            await cache.insert(
                partition_key=partition_key,
                prompt_text=embed_prompt,
                vector=query_vector,
                response_data=full_resp,
                model=model,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
            )

    total_ms = (time.perf_counter() - t_start) * 1000.0
    metrics_collector.record_miss(embedding_ms=0.0, search_ms=0.0, total_ms=total_ms)

    miss_headers = {
        "X-Cache": "MISS",
        "X-Cache-Latency-Ms": f"{total_ms:.2f}",
        "Content-Type": "text/event-stream",
    }

    return StreamingResponse(
        upstream.forward_stream(
            provider="anthropic",
            endpoint_path="/v1/messages",
            payload=body,
            client_auth_header=auth_token,
            on_complete_callback=on_stream_completed_anthropic,
        ),
        media_type="text/event-stream",
        headers=miss_headers,
    )

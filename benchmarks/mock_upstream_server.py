"""High-fidelity mock upstream LLM server simulating OpenAI and Anthropic APIs with configurable latency."""

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="CacheSense Mock Upstream Provider")

# Configurable simulated upstream response latency (milliseconds)
SIMULATED_LATENCY_MS = float(os.getenv("MOCK_UPSTREAM_LATENCY_MS", "800"))
CHUNK_COUNT = 8


async def generate_mock_openai_stream(model: str, prompt_text: str) -> AsyncGenerator[bytes, None]:
    resp_id = f"chatcmpl-mock-{int(time.time()*1000)}"
    # Initial TTFT (Time to First Token)
    ttft_seconds = (SIMULATED_LATENCY_MS * 0.6) / 1000.0
    await asyncio.sleep(ttft_seconds)

    # Initial chunk
    c0 = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(c0)}\n\n".encode()

    # Tokens streaming
    tokens = ["This ", "is ", "a ", "realistic ", "mock ", "LLM ", "upstream ", "response."]
    inter_token_delay = ((SIMULATED_LATENCY_MS * 0.4) / len(tokens)) / 1000.0

    for token in tokens:
        await asyncio.sleep(inter_token_delay)
        c = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(c)}\n\n".encode()

    # Final chunk
    c_final = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 15, "completion_tokens": len(tokens), "total_tokens": 15 + len(tokens)},
    }
    yield f"data: {json.dumps(c_final)}\n\n".encode()
    yield b"data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def mock_chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "gpt-4o")
    stream = bool(body.get("stream", False))

    if stream:
        return StreamingResponse(
            generate_mock_openai_stream(model, "sample prompt"),
            media_type="text/event-stream",
        )

    # Non-streaming
    await asyncio.sleep(SIMULATED_LATENCY_MS / 1000.0)
    return JSONResponse({
        "id": f"chatcmpl-mock-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a realistic mock LLM upstream response.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 8,
            "total_tokens": 23,
        },
    })


@app.post("/v1/messages")
async def mock_anthropic_messages(request: Request):
    body = await request.json()
    model = body.get("model", "claude-3-5-sonnet")
    await asyncio.sleep(SIMULATED_LATENCY_MS / 1000.0)
    return JSONResponse({
        "id": f"msg-mock-{int(time.time()*1000)}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "This is a realistic mock Anthropic response."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 15, "output_tokens": 8},
    })


@app.get("/health")
async def mock_health():
    return {"status": "ready", "simulated_latency_ms": SIMULATED_LATENCY_MS}


if __name__ == "__main__":
    port = int(os.getenv("MOCK_UPSTREAM_PORT", "8080"))
    uvicorn.run("benchmarks.mock_upstream_server:app", host="0.0.0.0", port=port, reload=False)

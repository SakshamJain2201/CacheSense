"""Upstream LLM provider client supporting OpenAI and Anthropic formats."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, Callable
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger("cachesense.upstream")


class UpstreamError(Exception):
    def __init__(self, status_code: int, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"Upstream error ({status_code}): {message}")
        self.status_code = status_code
        self.message = message
        self.details = details or {}


class UpstreamClient:
    """
    Asynchronous client for forwarding requests to upstream LLM providers (OpenAI / Anthropic).
    Supports non-streaming and streaming SSE requests with real-time token buffering.
    """

    def __init__(
        self,
        openai_base_url: str | None = None,
        anthropic_base_url: str | None = None,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.openai_base_url = (openai_base_url or settings.UPSTREAM_OPENAI_BASE_URL).rstrip("/")
        self.anthropic_base_url = (anthropic_base_url or settings.UPSTREAM_ANTHROPIC_BASE_URL).rstrip("/")
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self.anthropic_api_key = anthropic_api_key or settings.ANTHROPIC_API_KEY
        self.timeout = timeout
        self.max_retries = max_retries

        # If mock mode is active, override base urls
        if settings.MOCK_UPSTREAM_ENABLED:
            mock_url = settings.MOCK_UPSTREAM_BASE_URL.rstrip("/")
            self.openai_base_url = mock_url
            self.anthropic_base_url = mock_url

        self._client: httpx.AsyncClient | None = None

    def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _prepare_headers(
        self,
        provider: str,
        client_auth_header: str | None = None,
        custom_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if provider == "openai":
            auth = client_auth_header or (f"Bearer {self.openai_api_key}" if self.openai_api_key else None)
            if auth:
                headers["Authorization"] = auth
        elif provider == "anthropic":
            auth = client_auth_header or self.anthropic_api_key
            if auth:
                # Support both "x-api-key" format and "Bearer ..." format
                if auth.startswith("Bearer "):
                    headers["x-api-key"] = auth[len("Bearer "):]
                else:
                    headers["x-api-key"] = auth
            headers["anthropic-version"] = "2023-06-01"

        if custom_headers:
            for k, v in custom_headers.items():
                if k.lower() not in {"content-length", "host"}:
                    headers[k] = v

        return headers

    async def forward_request(
        self,
        provider: str,
        endpoint_path: str,
        payload: dict[str, Any],
        client_auth_header: str | None = None,
        custom_headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], int]:
        """
        Non-streaming forward request to upstream provider with exponential backoff retries.
        """
        base_url = self.openai_base_url if provider == "openai" else self.anthropic_base_url
        url = f"{base_url}{endpoint_path}"
        headers = self._prepare_headers(provider, client_auth_header, custom_headers)
        client = self.get_client()

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code >= 400:
                    try:
                        err_data = response.json()
                    except Exception:
                        err_data = {"text": response.text}
                    raise UpstreamError(
                        status_code=response.status_code,
                        message=f"Provider {provider} returned status {response.status_code}",
                        details=err_data,
                    )
                return response.json(), response.status_code
            except (httpx.RequestError, UpstreamError) as exc:
                last_error = exc
                if isinstance(exc, UpstreamError) and exc.status_code < 500:
                    # Client errors (4xx) should not be retried
                    raise
                if attempt < self.max_retries:
                    await asyncio.sleep(0.2 * (2**attempt))
                else:
                    break

        if isinstance(last_error, UpstreamError):
            raise last_error
        raise UpstreamError(
            status_code=502,
            message=f"Failed to connect to upstream {provider}: {last_error!s}",
        )

    async def forward_stream(
        self,
        provider: str,
        endpoint_path: str,
        payload: dict[str, Any],
        client_auth_header: str | None = None,
        custom_headers: dict[str, str] | None = None,
        on_complete_callback: Callable[[dict[str, Any], list[str]], Any] | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Forward streaming SSE request from upstream to client while buffering response.
        When stream finishes, reconstructs full response & calls on_complete_callback.
        """
        base_url = self.openai_base_url if provider == "openai" else self.anthropic_base_url
        url = f"{base_url}{endpoint_path}"
        headers = self._prepare_headers(provider, client_auth_header, custom_headers)
        headers["Accept"] = "text/event-stream"
        client = self.get_client()

        # Enforce stream=True in payload
        stream_payload = dict(payload)
        stream_payload["stream"] = True

        req = client.build_request("POST", url, json=stream_payload, headers=headers)
        response = await client.send(req, stream=True)

        if response.status_code >= 400:
            content = await response.aread()
            try:
                err_data = json.loads(content.decode())
            except Exception:
                err_data = {"text": content.decode()}
            await response.aclose()
            raise UpstreamError(
                status_code=response.status_code,
                message=f"Provider stream error {response.status_code}",
                details=err_data,
            )

        buffered_chunks: list[str] = []
        is_completed_normally = False

        try:
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk_str = f"{line}\n\n"
                buffered_chunks.append(chunk_str)
                yield chunk_str.encode("utf-8")

                if line.strip() == "data: [DONE]":
                    is_completed_normally = True

            # If Anthropic or no explicit [DONE] marker, normal stream end
            if not is_completed_normally and buffered_chunks:
                is_completed_normally = True

        except Exception as exc:
            logger.error(f"Error while streaming from upstream {provider}: {exc}")
            raise
        finally:
            await response.aclose()

        if is_completed_normally and on_complete_callback and buffered_chunks:
            # Reconstruct response payload from buffered chunks
            full_response = self._reconstruct_response(provider, stream_payload, buffered_chunks)
            if full_response:
                try:
                    if asyncio.iscoroutinefunction(on_complete_callback):
                        await on_complete_callback(full_response, buffered_chunks)
                    else:
                        on_complete_callback(full_response, buffered_chunks)
                except Exception as cb_exc:
                    logger.error(f"Error executing on_complete_callback for stream: {cb_exc}")

    def _reconstruct_response(
        self,
        provider: str,
        request_payload: dict[str, Any],
        chunks: list[str],
    ) -> dict[str, Any] | None:
        """Reconstruct standard response payload and token usage from SSE chunks."""
        model = request_payload.get("model", "unknown")
        if provider == "openai":
            full_text = []
            role = "assistant"
            usage = None
            response_id = f"chatcmpl-cache-{int(time.time()*1000)}"

            for chunk_str in chunks:
                lines = chunk_str.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    data_json = line[len("data:"):].strip()
                    try:
                        data = json.loads(data_json)
                        if "id" in data:
                            response_id = data["id"]
                        if data.get("usage"):
                            usage = data["usage"]
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "role" in delta:
                                role = delta["role"]
                            if delta.get("content"):
                                full_text.append(delta["content"])
                    except Exception:
                        continue

            text_content = "".join(full_text)
            if not usage:
                # Estimate token usage if not provided in stream
                p_tokens = len(json.dumps(request_payload.get("messages", []))) // 4
                c_tokens = len(text_content) // 4
                usage = {
                    "prompt_tokens": max(1, p_tokens),
                    "completion_tokens": max(1, c_tokens),
                    "total_tokens": max(2, p_tokens + c_tokens),
                }

            return {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": role,
                            "content": text_content,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }

        elif provider == "anthropic":
            full_text = []
            usage = {"input_tokens": 0, "output_tokens": 0}
            msg_id = f"msg-cache-{int(time.time()*1000)}"

            for chunk_str in chunks:
                lines = chunk_str.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_json = line[len("data:"):].strip()
                    try:
                        data = json.loads(data_json)
                        ev_type = data.get("type", "")
                        if ev_type == "message_start":
                            msg = data.get("message", {})
                            msg_id = msg.get("id", msg_id)
                            if "usage" in msg:
                                usage["input_tokens"] = msg["usage"].get("input_tokens", 0)
                        elif ev_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                full_text.append(delta.get("text", ""))
                        elif ev_type == "message_delta":
                            if "usage" in data:
                                usage["output_tokens"] = data["usage"].get("output_tokens", 0)
                    except Exception:
                        continue

            text_content = "".join(full_text)
            if usage["input_tokens"] == 0:
                usage["input_tokens"] = len(json.dumps(request_payload.get("messages", []))) // 4
            if usage["output_tokens"] == 0:
                usage["output_tokens"] = len(text_content) // 4

            return {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": text_content}],
                "stop_reason": "end_turn",
                "usage": usage,
            }

        return None

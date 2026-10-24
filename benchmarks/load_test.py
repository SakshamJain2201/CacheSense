"""Async concurrent load generator for measuring CacheSense throughput and latency percentiles."""

import asyncio
import time
from typing import Any

import httpx
import numpy as np


async def send_single_request(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    model: str = "gpt-4o",
    stream: bool = False,
) -> tuple[float, str, int]:
    """Send single request and measure latency (ms), cache status (HIT/MISS/BYPASS), and status code."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": stream,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{base_url}/v1/chat/completions", json=payload, timeout=30.0)
        total_ms = (time.perf_counter() - t0) * 1000.0
        cache_status = resp.headers.get("X-Cache", "MISS")
        return total_ms, cache_status, resp.status_code
    except Exception:
        total_ms = (time.perf_counter() - t0) * 1000.0
        return total_ms, "ERROR", 500


async def run_concurrent_load(
    base_url: str,
    prompts: list[str],
    concurrency: int = 10,
    stream: bool = False,
) -> dict[str, Any]:
    """
    Run async load test against CacheSense gateway across specified concurrency.
    Returns latency percentiles (p50, p95, p99), throughput, and hit ratio.
    """
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def worker(prompt_text: str):
            async with semaphore:
                return await send_single_request(
                    client=client,
                    base_url=base_url,
                    prompt=prompt_text,
                    stream=stream,
                )

        t_start = time.perf_counter()
        tasks = [asyncio.create_task(worker(p)) for p in prompts]
        results = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - t_start

    latencies = [r[0] for r in results if r[2] == 200]
    hits = sum(1 for r in results if r[1] == "HIT")
    misses = sum(1 for r in results if r[1] == "MISS")
    errors = sum(1 for r in results if r[2] != 200)

    if not latencies:
        return {"error": "No successful requests"}

    lat_arr = np.array(latencies, dtype=np.float32)
    throughput = len(latencies) / total_duration if total_duration > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": len(prompts),
        "successful_requests": len(latencies),
        "errors": errors,
        "hits": hits,
        "misses": misses,
        "hit_ratio_percent": round((hits / len(latencies) * 100) if latencies else 0.0, 2),
        "duration_seconds": round(total_duration, 2),
        "throughput_rps": round(throughput, 2),
        "p50_latency_ms": round(float(np.percentile(lat_arr, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(lat_arr, 99)), 2),
        "avg_latency_ms": round(float(np.mean(lat_arr)), 2),
    }

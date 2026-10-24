"""Admin and observability endpoints (/health, /metrics, /metrics/json, /cache/stats, /dashboard)."""

import os
import time

from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.api.proxy import get_cache_engine
from app.config import settings
from app.core.keying import build_partition_key, format_embed_text
from app.core.metrics import metrics_collector

router = APIRouter()
START_TIME = time.time()


class PlaygroundRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o"
    temperature: float = 0.0
    system_prompt: str | None = None
    threshold: float | None = None


@router.get("/health")
async def health_check():
    """Liveness and readiness health probe."""
    cache = get_cache_engine()
    store_ok = True
    try:
        count = await cache.store.count()
    except Exception:
        store_ok = False
        count = -1

    status = "healthy" if store_ok else "degraded"
    return {
        "status": status,
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "store_connected": store_ok,
        "indexed_entries": count,
        "faiss_vectors": cache.index.get_total_count(),
        "model": settings.EMBEDDING_MODEL_NAME,
        "version": "0.1.0",
    }


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus text-format metrics exporter."""
    cache = get_cache_engine()
    count = await cache.store.count()
    data = metrics_collector.get_prometheus_metrics(cache_size=count)
    return Response(content=data, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/metrics/json")
async def json_metrics():
    """Real-time metrics in JSON format for the live dashboard."""
    cache = get_cache_engine()
    count = await cache.store.count()
    return metrics_collector.get_metrics_json(cache_size=count)


@router.get("/cache/stats")
async def cache_stats():
    """Detailed cache statistics and partition information."""
    cache = get_cache_engine()
    stats = await cache.get_stats()
    return stats


@router.delete("/cache")
async def purge_cache():
    """Administrative purge of all cached vectors and response records."""
    cache = get_cache_engine()
    await cache.clear()
    return {"status": "success", "message": "Cache successfully purged."}


@router.post("/playground/test")
async def playground_test(req: PlaygroundRequest):
    """
    Playground testing endpoint: simulate semantic lookup against the cache.
    Returns HIT/MISS status, similarity score, latency, and matched stored prompt.
    """
    t0 = time.perf_counter()
    cache = get_cache_engine()
    thresh = req.threshold if req.threshold is not None else settings.SIMILARITY_THRESHOLD

    partition_key = build_partition_key(
        model=req.model,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
        max_cacheable_temperature=settings.MAX_CACHEABLE_TEMPERATURE,
    )

    if partition_key is None:
        return {
            "status": "UNCACHEABLE",
            "reason": f"Temperature {req.temperature} exceeds maximum cacheable {settings.MAX_CACHEABLE_TEMPERATURE}",
            "similarity": 0.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "matched_prompt": None,
        }

    embed_text = format_embed_text(req.prompt, "ctx:empty")
    entry, similarity, emb_ms, search_ms, _ = await cache.lookup(
        partition_key=partition_key,
        embed_prompt=embed_text,
        threshold=thresh,
    )
    total_ms = (time.perf_counter() - t0) * 1000.0

    if entry is not None:
        return {
            "status": "HIT",
            "similarity": round(similarity, 4),
            "threshold": thresh,
            "latency_ms": round(total_ms, 2),
            "embedding_ms": round(emb_ms, 2),
            "search_ms": round(search_ms, 2),
            "matched_prompt": entry.prompt_text,
            "cached_response_preview": entry.response_data.get("choices", [{}])[0].get("message", {}).get("content", "")[:200]
            if "choices" in entry.response_data
            else str(entry.response_data)[:200],
        }

    return {
        "status": "MISS",
        "similarity": round(similarity, 4),
        "threshold": thresh,
        "latency_ms": round(total_ms, 2),
        "embedding_ms": round(emb_ms, 2),
        "search_ms": round(search_ms, 2),
        "matched_prompt": None,
    }


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the CacheSense real-time analytics dashboard."""
    dashboard_file = os.path.join("dashboard", "index.html")
    if os.path.exists(dashboard_file):
        with open(dashboard_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    # Fallback inline if file not yet written
    return HTMLResponse(content="<h1>CacheSense Dashboard Loading...</h1>")

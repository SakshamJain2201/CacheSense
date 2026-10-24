"""Main FastAPI application for CacheSense gateway."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import admin, proxy
from app.config import settings
from app.core.cache import SemanticCache
from app.core.single_flight import SingleFlight
from app.core.upstream import UpstreamClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cachesense.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan managing startup initialization and graceful shutdown."""
    logger.info("Starting CacheSense Gateway...")

    # 1. Initialize SemanticCache (store, model weights warmup, FAISS snapshot & reconciliation)
    cache = SemanticCache()
    await cache.initialize()
    proxy.cache_engine = cache

    # 2. Initialize Upstream Client
    upstream = UpstreamClient()
    proxy.upstream_client = upstream

    # 3. Initialize SingleFlight coalescer
    sf = SingleFlight()
    proxy.single_flight = sf

    logger.info(f"CacheSense Gateway successfully started on {settings.HOST}:{settings.PORT}")

    yield

    # Shutdown sequence
    logger.info("Stopping CacheSense Gateway...")
    await cache.shutdown()
    await upstream.close()
    logger.info("CacheSense Gateway successfully stopped.")


app = FastAPI(
    title="CacheSense",
    description="High-performance Semantic Caching Gateway for LLM APIs (OpenAI & Anthropic)",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for dashboard and playground interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(proxy.router)
app.include_router(admin.router)


@app.get("/")
async def root():
    return {
        "service": "CacheSense",
        "status": "online",
        "dashboard": "/dashboard",
        "metrics": "/metrics",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False)

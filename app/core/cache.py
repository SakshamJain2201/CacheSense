"""Semantic cache coordinator orchestrating embedding, FAISS indexing, SQLite storage, and eviction."""

import asyncio
import logging
import os
import time
from typing import Any

import numpy as np

from app.config import settings
from app.core.embedder import PromptEmbedder
from app.core.index import PartitionedFaissIndex
from app.core.keying import sha256_hash
from app.core.metrics import metrics_collector
from app.core.store import CacheEntry, ResponseStore, SQLiteResponseStore

logger = logging.getLogger("cachesense.cache")


class SemanticCache:
    """
    Central orchestration engine managing semantic lookups, vector indexing,
    persistence, lazy TTL expiry, LRU eviction, and snapshot reconciliation.
    """

    def __init__(
        self,
        embedder: PromptEmbedder | None = None,
        index: PartitionedFaissIndex | None = None,
        store: ResponseStore | None = None,
    ):
        self.embedder = embedder or PromptEmbedder(
            model_name=settings.EMBEDDING_MODEL_NAME,
            max_threads=settings.EMBEDDING_THREADS,
        )
        self.index = index or PartitionedFaissIndex(dimension=settings.EMBEDDING_DIM)
        self.store = store or SQLiteResponseStore(db_path=settings.SQLITE_DB_PATH)
        self._sweeper_task: asyncio.Task | None = None
        self._is_running = False

    async def initialize(self) -> None:
        """Initialize store, load models, restore FAISS snapshots, and reconcile."""
        logger.info("Initializing CacheSense semantic cache...")
        await self.store.initialize()

        # Load embedding model
        self.embedder.load_model()

        # Restore FAISS snapshot if available
        snapshot_dir = settings.FAISS_INDEX_PATH
        if os.path.exists(snapshot_dir):
            success = self.index.load_snapshot(snapshot_dir)
            if success:
                logger.info(f"Loaded FAISS snapshot from {snapshot_dir}")
            else:
                logger.info("No valid FAISS snapshot found. Starting with fresh index.")

        # Reconcile FAISS IDs against SQLite
        valid_store_ids = await self.store.get_all_ids()
        removed_orphans = self.index.reconcile_with_store(valid_store_ids)
        if removed_orphans > 0:
            logger.info(f"Reconciliation removed {removed_orphans} orphaned FAISS vectors.")

        # If store has entries missing from FAISS, re-index them
        faiss_ids = self.index.get_indexed_ids()
        missing_in_faiss = valid_store_ids - faiss_ids
        if missing_in_faiss:
            logger.info(f"Re-indexing {len(missing_in_faiss)} store entries into FAISS...")
            for eid in missing_in_faiss:
                entry = await self.store.get(eid)
                if entry and not entry.is_expired():
                    vec = await self.embedder.embed(entry.prompt_text)
                    self.index.add(entry.partition_key, eid, vec)

        self._is_running = True
        # Start periodic TTL sweeper
        self._sweeper_task = asyncio.create_task(self._periodic_sweeper_loop())
        logger.info(f"CacheSense ready. Active entries: {await self.store.count()}")

    async def shutdown(self) -> None:
        """Gracefully snapshot index and close connections on shutdown."""
        logger.info("Shutting down CacheSense semantic cache...")
        self._is_running = False
        if self._sweeper_task:
            self._sweeper_task.cancel()
            try:
                await self._sweeper_task
            except asyncio.CancelledError:
                pass

        # Persist FAISS index snapshot
        snapshot_dir = settings.FAISS_INDEX_PATH
        self.index.save_snapshot(snapshot_dir)
        logger.info(f"Saved FAISS index snapshot to {snapshot_dir}")

        self.embedder.shutdown()
        await self.store.close()

    async def lookup(
        self,
        partition_key: str,
        embed_prompt: str,
        threshold: float,
    ) -> tuple[CacheEntry | None, float, float, float, np.ndarray]:
        """
        Perform semantic search in the target partition.
        Returns (entry_or_None, similarity_score, embedding_ms, search_ms, query_vector).
        """
        # 1. Embed prompt
        t0 = time.perf_counter()
        query_vector = await self.embedder.embed(embed_prompt)
        embedding_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Search FAISS in partition
        t1 = time.perf_counter()
        matched_id, similarity = self.index.search(
            partition_key=partition_key,
            vector=query_vector,
            top_k=1,
        )
        search_ms = (time.perf_counter() - t1) * 1000.0

        # 3. Check threshold
        if matched_id >= 0 and similarity >= threshold:
            entry = await self.store.get(matched_id)
            if entry is not None:
                # Lazy TTL verification
                if entry.is_expired():
                    logger.debug(f"Entry {matched_id} expired lazily. Evicting.")
                    await self.delete_entry(matched_id)
                    return None, similarity, embedding_ms, search_ms, query_vector

                # Update LRU access timestamp asynchronously
                asyncio.create_task(self.store.update_access_time(matched_id))
                return entry, similarity, embedding_ms, search_ms, query_vector

        return None, similarity, embedding_ms, search_ms, query_vector

    async def insert(
        self,
        partition_key: str,
        prompt_text: str,
        vector: np.ndarray,
        response_data: dict[str, Any],
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        ttl_seconds: int | None = None,
    ) -> int:
        """Insert response into store and FAISS index with capacity LRU management."""
        ttl = ttl_seconds if ttl_seconds is not None else settings.DEFAULT_TTL_SECONDS
        
        # Enforce max cache entries capacity with LRU eviction
        current_count = await self.store.count()
        if current_count >= settings.MAX_CACHE_ENTRIES:
            evict_count = max(10, int(settings.MAX_CACHE_ENTRIES * 0.05))  # Evict 5%
            await self.evict_lru(limit=evict_count)

        entry = CacheEntry(
            partition_key=partition_key,
            prompt_hash=sha256_hash(prompt_text),
            prompt_text=prompt_text,
            response_data=response_data,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            ttl_seconds=ttl,
        )

        # 1. Insert in SQLite store to get unique integer ID
        entry_id = await self.store.insert(entry)

        # 2. Add to FAISS partition index
        self.index.add(partition_key=partition_key, entry_id=entry_id, vector=vector)
        return entry_id

    async def delete_entry(self, entry_id: int) -> None:
        """Consistently remove an entry from both FAISS and SQLite."""
        self.index.remove(entry_id)
        await self.store.delete(entry_id)
        metrics_collector.record_eviction(1)

    async def evict_lru(self, limit: int = 50) -> int:
        """Evict least recently used entries to free cache capacity."""
        candidates = await self.store.get_lru_candidates(limit=limit)
        if not candidates:
            return 0
        self.index.remove_many(candidates)
        await self.store.delete_many(candidates)
        metrics_collector.record_eviction(len(candidates))
        logger.info(f"LRU eviction purged {len(candidates)} entries.")
        return len(candidates)

    async def sweep_expired(self) -> int:
        """Periodic background sweeper to evict expired TTL entries."""
        now = time.time()
        expired_ids = await self.store.get_expired_candidates(now_ts=now, limit=500)
        if not expired_ids:
            return 0
        self.index.remove_many(expired_ids)
        await self.store.delete_many(expired_ids)
        metrics_collector.record_eviction(len(expired_ids))
        logger.info(f"TTL sweeper purged {len(expired_ids)} expired entries.")
        return len(expired_ids)

    async def clear(self) -> None:
        """Completely purge cache and vector index."""
        self.index.clear()
        await self.store.clear()

    async def get_stats(self) -> dict[str, Any]:
        """Summary statistics of store and FAISS."""
        count = await self.store.count()
        faiss_count = self.index.get_total_count()
        return {
            "total_store_entries": count,
            "total_faiss_vectors": faiss_count,
            "max_capacity": settings.MAX_CACHE_ENTRIES,
            "default_ttl_seconds": settings.DEFAULT_TTL_SECONDS,
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        }

    async def _periodic_sweeper_loop(self) -> None:
        """Background coroutine sweeping expired TTL entries periodically."""
        while self._is_running:
            try:
                await asyncio.sleep(settings.TTL_SWEEP_INTERVAL_SECONDS)
                await self.sweep_expired()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in TTL sweeper loop: {exc}")

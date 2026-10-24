"""ResponseStore interface and SQLite implementation with WAL mode."""

import asyncio
import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    id: int | None = None
    partition_key: str
    prompt_hash: str
    prompt_text: str
    response_data: dict[str, Any]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    created_at: float = Field(default_factory=time.time)
    last_accessed_at: float = Field(default_factory=time.time)
    ttl_seconds: int = 86400

    def is_expired(self, current_time: float | None = None) -> bool:
        now = current_time if current_time is not None else time.time()
        return (self.created_at + self.ttl_seconds) < now


class ResponseStore(ABC):
    """Abstract base class for CacheSense response storage."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize database, tables, and indices."""

    @abstractmethod
    async def get(self, entry_id: int) -> CacheEntry | None:
        """Fetch cache entry by primary ID."""

    @abstractmethod
    async def insert(self, entry: CacheEntry) -> int:
        """Insert new entry and return its assigned primary ID."""

    @abstractmethod
    async def update_access_time(self, entry_id: int, access_time: float | None = None) -> None:
        """Update last_accessed_at timestamp for LRU tracking."""

    @abstractmethod
    async def delete(self, entry_id: int) -> None:
        """Delete single entry by ID."""

    @abstractmethod
    async def delete_many(self, entry_ids: list[int]) -> None:
        """Delete multiple entries by ID."""

    @abstractmethod
    async def get_lru_candidates(self, limit: int) -> list[int]:
        """Fetch IDs of least recently accessed entries."""

    @abstractmethod
    async def get_expired_candidates(self, now_ts: float, limit: int = 500) -> list[int]:
        """Fetch IDs of expired entries based on TTL."""

    @abstractmethod
    async def get_all_ids(self) -> set[int]:
        """Retrieve all active entry IDs for reconciliation."""

    @abstractmethod
    async def count(self) -> int:
        """Total number of active records."""

    @abstractmethod
    async def clear(self) -> None:
        """Purge all stored entries."""

    @abstractmethod
    async def close(self) -> None:
        """Close storage connections."""


class SQLiteResponseStore(ResponseStore):
    """Production-grade SQLite ResponseStore operating in WAL mode."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Enable Write-Ahead Logging for high concurrency
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._conn = conn
        return self._conn

    async def initialize(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._init_db_sync)

    def _init_db_sync(self) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    partition_key TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    prompt_text TEXT NOT NULL,
                    response_data TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    ttl_seconds INTEGER NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_partition ON cache_entries(partition_key);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache_entries(created_at);")

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        return CacheEntry(
            id=row["id"],
            partition_key=row["partition_key"],
            prompt_hash=row["prompt_hash"],
            prompt_text=row["prompt_text"],
            response_data=json.loads(row["response_data"]),
            model=row["model"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            ttl_seconds=row["ttl_seconds"],
        )

    async def get(self, entry_id: int) -> CacheEntry | None:
        async with self._lock:
            return await asyncio.to_thread(self._get_sync, entry_id)

    def _get_sync(self, entry_id: int) -> CacheEntry | None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cache_entries WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        return self._row_to_entry(row) if row else None

    async def insert(self, entry: CacheEntry) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._insert_sync, entry)

    def _insert_sync(self, entry: CacheEntry) -> int:
        conn = self._get_connection()
        serialized_response = json.dumps(entry.response_data)
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cache_entries (
                    partition_key, prompt_hash, prompt_text, response_data,
                    model, prompt_tokens, completion_tokens, total_tokens,
                    created_at, last_accessed_at, ttl_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.partition_key,
                entry.prompt_hash,
                entry.prompt_text,
                serialized_response,
                entry.model,
                entry.prompt_tokens,
                entry.completion_tokens,
                entry.total_tokens,
                entry.created_at,
                entry.last_accessed_at,
                entry.ttl_seconds,
            ))
            entry_id = cursor.lastrowid
            if entry_id is None:
                raise RuntimeError("Failed to retrieve generated rowid from SQLite.")
            return entry_id

    async def update_access_time(self, entry_id: int, access_time: float | None = None) -> None:
        ts = access_time if access_time is not None else time.time()
        async with self._lock:
            await asyncio.to_thread(self._update_access_time_sync, entry_id, ts)

    def _update_access_time_sync(self, entry_id: int, ts: float) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute("UPDATE cache_entries SET last_accessed_at = ? WHERE id = ?", (ts, entry_id))

    async def delete(self, entry_id: int) -> None:
        async with self._lock:
            await asyncio.to_thread(self._delete_sync, entry_id)

    def _delete_sync(self, entry_id: int) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM cache_entries WHERE id = ?", (entry_id,))

    async def delete_many(self, entry_ids: list[int]) -> None:
        if not entry_ids:
            return
        async with self._lock:
            await asyncio.to_thread(self._delete_many_sync, entry_ids)

    def _delete_many_sync(self, entry_ids: list[int]) -> None:
        conn = self._get_connection()
        placeholders = ",".join("?" for _ in entry_ids)
        with conn:
            conn.execute(f"DELETE FROM cache_entries WHERE id IN ({placeholders})", entry_ids)

    async def get_lru_candidates(self, limit: int) -> list[int]:
        async with self._lock:
            return await asyncio.to_thread(self._get_lru_sync, limit)

    def _get_lru_sync(self, limit: int) -> list[int]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM cache_entries ORDER BY last_accessed_at ASC LIMIT ?", (limit,))
        return [row["id"] for row in cursor.fetchall()]

    async def get_expired_candidates(self, now_ts: float, limit: int = 500) -> list[int]:
        async with self._lock:
            return await asyncio.to_thread(self._get_expired_sync, now_ts, limit)

    def _get_expired_sync(self, now_ts: float, limit: int) -> list[int]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM cache_entries
            WHERE (created_at + ttl_seconds) < ?
            ORDER BY created_at ASC LIMIT ?
        """, (now_ts, limit))
        return [row["id"] for row in cursor.fetchall()]

    async def get_all_ids(self) -> set[int]:
        async with self._lock:
            return await asyncio.to_thread(self._get_all_ids_sync)

    def _get_all_ids_sync(self) -> set[int]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM cache_entries")
        return {row["id"] for row in cursor.fetchall()}

    async def count(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._count_sync)

    def _count_sync(self) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total FROM cache_entries")
        row = cursor.fetchone()
        return row["total"] if row else 0

    async def clear(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._clear_sync)

    def _clear_sync(self) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM cache_entries")

    async def close(self) -> None:
        async with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

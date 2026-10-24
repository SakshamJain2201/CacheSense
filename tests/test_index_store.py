"""Unit tests for FAISS Partitioned Vector Index and SQLite ResponseStore."""

import os
import time

import numpy as np
import pytest

from app.core.index import PartitionedFaissIndex
from app.core.store import CacheEntry, SQLiteResponseStore


@pytest.mark.asyncio
async def test_sqlite_store_crud(isolated_test_dir):
    db_path = os.path.join(isolated_test_dir, "test_store.db")
    store = SQLiteResponseStore(db_path)
    await store.initialize()

    entry = CacheEntry(
        partition_key="gpt-4o|test",
        prompt_hash="abcd1234",
        prompt_text="Hello world",
        response_data={"choices": [{"message": {"content": "Hi!"}}]},
        model="gpt-4o",
        prompt_tokens=5,
        completion_tokens=2,
        ttl_seconds=3600,
    )

    # Insert
    entry_id = await store.insert(entry)
    assert entry_id > 0
    assert await store.count() == 1

    # Get
    fetched = await store.get(entry_id)
    assert fetched is not None
    assert fetched.prompt_text == "Hello world"
    assert fetched.response_data["choices"][0]["message"]["content"] == "Hi!"
    assert fetched.is_expired() is False

    # Update access time (LRU)
    new_ts = time.time() + 10.0
    await store.update_access_time(entry_id, new_ts)
    updated = await store.get(entry_id)
    assert updated.last_accessed_at == new_ts

    # Delete
    await store.delete(entry_id)
    assert await store.count() == 0
    assert await store.get(entry_id) is None

    await store.close()


@pytest.mark.asyncio
async def test_sqlite_lru_and_ttl_expiry(isolated_test_dir):
    db_path = os.path.join(isolated_test_dir, "test_lru.db")
    store = SQLiteResponseStore(db_path)
    await store.initialize()

    now = time.time()
    # Insert 3 entries with different last_accessed_at and TTLs
    e1 = CacheEntry(
        partition_key="p1", prompt_hash="h1", prompt_text="t1",
        response_data={}, model="m", created_at=now - 200, last_accessed_at=now - 100, ttl_seconds=50
    )
    e2 = CacheEntry(
        partition_key="p1", prompt_hash="h2", prompt_text="t2",
        response_data={}, model="m", created_at=now, last_accessed_at=now - 50, ttl_seconds=3600
    )
    e3 = CacheEntry(
        partition_key="p1", prompt_hash="h3", prompt_text="t3",
        response_data={}, model="m", created_at=now, last_accessed_at=now, ttl_seconds=3600
    )

    id1 = await store.insert(e1)
    id2 = await store.insert(e2)
    id3 = await store.insert(e3)

    # LRU candidate check
    lru_ids = await store.get_lru_candidates(limit=2)
    assert lru_ids == [id1, id2]

    # Expired candidate check (e1 expired because now - 200 + 50 < now)
    expired = await store.get_expired_candidates(now_ts=now)
    assert id1 in expired
    assert id2 not in expired
    assert id3 not in expired

    await store.close()


def test_faiss_partition_isolation_and_search():
    index = PartitionedFaissIndex(dimension=4)

    # Generate two orthogonal unit vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    # Add v1 to partition_A with ID 101
    index.add(partition_key="partition_A", entry_id=101, vector=v1)
    # Add v2 to partition_B with ID 202
    index.add(partition_key="partition_B", entry_id=202, vector=v2)

    assert index.get_total_count() == 2

    # Query v1 in partition_A -> Should match ID 101 with cosine similarity ~ 1.0
    mid, score = index.search(partition_key="partition_A", vector=v1)
    assert mid == 101
    assert abs(score - 1.0) < 1e-4

    # Query v1 in partition_B -> Should NOT match (orthogonal vector in partition_B has score 0.0)
    mid_b, score_b = index.search(partition_key="partition_B", vector=v1)
    assert mid_b == 202
    assert abs(score_b - 0.0) < 1e-4

    # Query in non-existent partition -> returns (-1, 0.0)
    mid_c, score_c = index.search(partition_key="non_existent", vector=v1)
    assert mid_c == -1
    assert score_c == 0.0


def test_faiss_remove_and_eviction():
    index = PartitionedFaissIndex(dimension=4)
    v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    index.add(partition_key="part1", entry_id=1, vector=v)
    index.add(partition_key="part1", entry_id=2, vector=v)
    index.add(partition_key="part2", entry_id=3, vector=v)

    assert index.get_total_count() == 3

    # Remove single ID
    removed = index.remove(entry_id=1)
    assert removed is True
    assert index.get_total_count() == 2
    assert 1 not in index.get_indexed_ids()

    # Batch remove across partitions
    num_batch = index.remove_many([2, 3])
    assert num_batch == 2
    assert index.get_total_count() == 0


def test_faiss_snapshot_persistence_and_reload(isolated_test_dir):
    snapshot_dir = os.path.join(isolated_test_dir, "faiss_snap")
    index1 = PartitionedFaissIndex(dimension=4)
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    index1.add(partition_key="p1", entry_id=999, vector=v)

    # Save snapshot
    index1.save_snapshot(snapshot_dir)

    # Load into fresh index
    index2 = PartitionedFaissIndex(dimension=4)
    loaded = index2.load_snapshot(snapshot_dir)
    assert loaded is True
    assert index2.get_total_count() == 1
    assert 999 in index2.get_indexed_ids()

    mid, score = index2.search("p1", v)
    assert mid == 999
    assert abs(score - 1.0) < 1e-4


def test_faiss_reconciliation():
    index = PartitionedFaissIndex(dimension=4)
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    index.add("p", 1, v)
    index.add("p", 2, v)
    index.add("p", 3, v)

    # Suppose SQLite only has ID 2 and 3 (ID 1 was deleted out-of-band)
    valid_store_ids = {2, 3}
    orphans_removed = index.reconcile_with_store(valid_store_ids)
    assert orphans_removed == 1
    assert index.get_indexed_ids() == {2, 3}

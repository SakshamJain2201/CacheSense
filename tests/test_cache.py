"""Unit tests for SemanticCache coordinator."""

import asyncio

import pytest

from app.core.cache import SemanticCache


@pytest.mark.asyncio
async def test_semantic_cache_lookup_and_insert(isolated_test_dir):
    cache = SemanticCache()
    await cache.initialize()

    partition = "gpt-4o|sys:none|t0.00|tools:none|ns:global"
    prompt = "What is the capital of Japan?"
    vector = await cache.embedder.embed(prompt)
    response_payload = {
        "id": "chatcmpl-test-1",
        "choices": [{"message": {"role": "assistant", "content": "Tokyo"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }

    # 1. Lookup on empty cache -> Miss
    entry, sim, _, _, _ = await cache.lookup(partition, prompt, threshold=0.92)
    assert entry is None

    # 2. Insert into cache
    eid = await cache.insert(
        partition_key=partition,
        prompt_text=prompt,
        vector=vector,
        response_data=response_payload,
        model="gpt-4o",
        prompt_tokens=10,
        completion_tokens=2,
    )
    assert eid > 0

    # 3. Lookup identical prompt -> Hit
    entry_hit, sim_hit, _, _, _ = await cache.lookup(partition, prompt, threshold=0.92)
    assert entry_hit is not None
    assert sim_hit >= 0.99
    assert entry_hit.response_data["choices"][0]["message"]["content"] == "Tokyo"

    # 4. Lookup semantically equivalent paraphrase -> Hit
    paraphrase = "Tell me Japan's capital city"
    entry_para, sim_para, _, _, _ = await cache.lookup(partition, paraphrase, threshold=0.80)
    assert entry_para is not None
    assert sim_para >= 0.80

    await cache.shutdown()


@pytest.mark.asyncio
async def test_semantic_cache_capacity_lru_eviction(isolated_test_dir, monkeypatch):
    # Set capacity to 3
    from app.config import settings
    monkeypatch.setattr(settings, "MAX_CACHE_ENTRIES", 3)

    cache = SemanticCache()
    await cache.initialize()

    partition = "gpt-4o|test"
    dummy_resp = {"choices": [{"message": {"content": "ok"}}]}

    # Insert 3 entries
    for i in range(1, 4):
        v = await cache.embedder.embed(f"prompt {i}")
        await cache.insert(
            partition_key=partition,
            prompt_text=f"prompt {i}",
            vector=v,
            response_data=dummy_resp,
            model="gpt-4o",
        )
        await asyncio.sleep(0.01)

    assert await cache.store.count() == 3

    # Insert 4th entry -> Should trigger capacity eviction
    v4 = await cache.embedder.embed("prompt 4")
    await cache.insert(
        partition_key=partition,
        prompt_text="prompt 4",
        vector=v4,
        response_data=dummy_resp,
        model="gpt-4o",
    )

    # Count should stay at or below MAX_CACHE_ENTRIES
    count = await cache.store.count()
    assert count <= 3
    # FAISS and SQLite must match
    assert cache.index.get_total_count() == count

    await cache.shutdown()


@pytest.mark.asyncio
async def test_semantic_cache_ttl_expiry(isolated_test_dir):
    cache = SemanticCache()
    await cache.initialize()

    partition = "gpt-4o|ttl"
    v = await cache.embedder.embed("expire me")
    
    # Insert with 1 second TTL
    eid = await cache.insert(
        partition_key=partition,
        prompt_text="expire me",
        vector=v,
        response_data={"choices": [{"message": {"content": "expired"}}]},
        model="gpt-4o",
        ttl_seconds=1,
    )

    # Immediately lookup -> Hit
    entry, _, _, _, _ = await cache.lookup(partition, "expire me", threshold=0.92)
    assert entry is not None

    # Wait for TTL to pass
    await asyncio.sleep(1.2)

    # Lookup -> Lazy eviction removes entry and returns None
    entry_exp, _, _, _, _ = await cache.lookup(partition, "expire me", threshold=0.92)
    assert entry_exp is None
    assert await cache.store.get(eid) is None
    assert eid not in cache.index.get_indexed_ids()

    await cache.shutdown()

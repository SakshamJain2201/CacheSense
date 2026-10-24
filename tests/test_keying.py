"""Unit tests for cache partition key generation, prompt extraction, and PII filters."""

from app.core.keying import (
    build_partition_key,
    check_pii_violation,
    extract_anthropic_prompt,
    extract_openai_prompt,
    format_embed_text,
)


def test_build_partition_key_basic():
    key = build_partition_key(
        model="GPT-4O",
        system_prompt="You are an assistant.",
        temperature=0.0,
        tools=None,
    )
    assert key is not None
    assert key.startswith("gpt-4o|")
    assert "t0.00" in key
    assert "tools:none" in key
    assert "ns:global" in key


def test_build_partition_key_temperature_uncacheable():
    # Temperature > 0.3 should return None (uncacheable)
    key = build_partition_key(
        model="gpt-4o",
        temperature=0.7,
        max_cacheable_temperature=0.3,
    )
    assert key is None

    # Temperature <= 0.3 should be cacheable
    key_low = build_partition_key(
        model="gpt-4o",
        temperature=0.25,
        max_cacheable_temperature=0.3,
    )
    assert key_low is not None
    assert "t0.25" in key_low


def test_build_partition_key_system_prompt_isolation():
    key1 = build_partition_key(model="gpt-4o", system_prompt="System A")
    key2 = build_partition_key(model="gpt-4o", system_prompt="System B")
    assert key1 != key2


def test_build_partition_key_tools_isolation():
    tools1 = [{"name": "get_weather", "description": "Fetches weather"}]
    tools2 = [{"name": "search_db", "description": "Queries database"}]
    key1 = build_partition_key(model="gpt-4o", tools=tools1)
    key2 = build_partition_key(model="gpt-4o", tools=tools2)
    assert key1 != key2


def test_build_partition_key_namespace():
    key_global = build_partition_key(model="gpt-4o", namespace_by_key=False)
    key_user1 = build_partition_key(model="gpt-4o", api_key="sk-user-1", namespace_by_key=True)
    key_user2 = build_partition_key(model="gpt-4o", api_key="sk-user-2", namespace_by_key=True)

    assert "ns:global" in key_global
    assert "ns:key:" in key_user1
    assert "ns:key:" in key_user2
    assert key_user1 != key_user2


def test_extract_openai_prompt_single_turn():
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is Python?"},
    ]
    prompt, ctx_hash, sys_prompt = extract_openai_prompt(messages)
    assert prompt == "What is Python?"
    assert ctx_hash == "ctx:empty"
    assert sys_prompt == "Be concise."


def test_extract_openai_prompt_multi_turn():
    messages = [
        {"role": "user", "content": "Turn 1 question"},
        {"role": "assistant", "content": "Turn 1 response"},
        {"role": "user", "content": "Turn 2 question"},
    ]
    prompt, ctx_hash, sys_prompt = extract_openai_prompt(messages)
    assert prompt == "Turn 2 question"
    assert ctx_hash != "ctx:empty"
    assert ctx_hash.startswith("ctx:")

    # Formatted embed text must include context hash
    embed_text = format_embed_text(prompt, ctx_hash)
    assert f"[{ctx_hash}] Turn 2 question" == embed_text


def test_extract_anthropic_prompt():
    messages = [
        {"role": "user", "content": "Explain relativity."},
    ]
    prompt, ctx_hash, sys_prompt = extract_anthropic_prompt(messages, system="Physics tutor")
    assert prompt == "Explain relativity."
    assert sys_prompt == "Physics tutor"
    assert ctx_hash == "ctx:empty"


def test_check_pii_violation():
    patterns = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
        r"\bsk-[a-zA-Z0-9]{32,}\b",
    ]
    assert check_pii_violation("My email is user@example.com for login.", patterns) is True
    assert check_pii_violation("My token is sk-1234567890abcdef1234567890abcdef here.", patterns) is True
    assert check_pii_violation("What is the capital of France?", patterns) is False

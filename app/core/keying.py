"""Cache partition key and prompt extraction logic."""

import hashlib
import json
import re
from typing import Any


def sha256_hash(data: str) -> str:
    """Compute 16-character truncated hex SHA-256 digest."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def canonical_json_hash(obj: Any) -> str:
    """Produce deterministic SHA-256 hash for JSON-serializable structures."""
    if not obj:
        return "none"
    serialized = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return sha256_hash(serialized)


def build_partition_key(
    model: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    tools: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
    namespace_by_key: bool = False,
    max_cacheable_temperature: float = 0.3,
) -> str | None:
    """
    Construct partition key ensuring strict cache isolation.
    Returns None if request is uncacheable (e.g. temperature > max_cacheable_temperature).
    """
    if temperature > max_cacheable_temperature:
        return None

    norm_model = model.strip().lower()
    sys_hash = sha256_hash(system_prompt.strip()) if system_prompt and system_prompt.strip() else "none"
    temp_bucket = f"t{round(temperature, 2):.2f}"
    tools_hash = canonical_json_hash(tools)
    
    if namespace_by_key and api_key:
        ns = f"key:{sha256_hash(api_key)}"
    else:
        ns = "global"

    return f"{norm_model}|sys:{sys_hash}|{temp_bucket}|tools:{tools_hash}|ns:{ns}"


def extract_openai_prompt(
    messages: list[dict[str, Any]],
    system_override: str | None = None,
) -> tuple[str, str, str | None]:
    """
    Extract (final_user_message, context_hash, system_prompt) from OpenAI messages list.
    """
    system_parts: list[str] = []
    if system_override:
        system_parts.append(system_override)

    dialogue_turns: list[dict[str, str]] = []
    final_user_message = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Parse OpenAI multi-part text content
            text_blocks = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            content = " ".join(text_blocks)
        elif not isinstance(content, str):
            content = str(content)

        if role == "system":
            system_parts.append(content)
        else:
            dialogue_turns.append({"role": role, "content": content})

    if not dialogue_turns:
        return "", "ctx:empty", " ".join(system_parts) if system_parts else None

    # Find last user message
    last_user_idx = -1
    for i in range(len(dialogue_turns) - 1, -1, -1):
        if dialogue_turns[i]["role"] == "user":
            last_user_idx = i
            break

    if last_user_idx >= 0:
        final_user_message = dialogue_turns[last_user_idx]["content"]
        preceding_turns = dialogue_turns[:last_user_idx]
    else:
        # Fallback to last message
        final_user_message = dialogue_turns[-1]["content"]
        preceding_turns = dialogue_turns[:-1]

    if preceding_turns:
        ctx_repr = json.dumps(preceding_turns, sort_keys=True, separators=(",", ":"))
        context_hash = f"ctx:{sha256_hash(ctx_repr)}"
    else:
        context_hash = "ctx:empty"

    combined_system = " ".join(system_parts) if system_parts else None
    return final_user_message, context_hash, combined_system


def extract_anthropic_prompt(
    messages: list[dict[str, Any]],
    system: str | None = None,
) -> tuple[str, str, str | None]:
    """
    Extract (final_user_message, context_hash, system_prompt) from Anthropic messages list.
    """
    return extract_openai_prompt(messages, system_override=system)


def format_embed_text(final_user_message: str, context_hash: str) -> str:
    """
    Embed the final user message combined with the context hash for multi-turn chats.
    """
    cleaned_prompt = final_user_message.strip()
    if context_hash == "ctx:empty":
        return cleaned_prompt
    return f"[{context_hash}] {cleaned_prompt}"


def check_pii_violation(text: str, patterns: list[str]) -> bool:
    """Return True if text matches any PII/security regex patterns."""
    for pat in patterns:
        if re.search(pat, text):
            return True
    return False

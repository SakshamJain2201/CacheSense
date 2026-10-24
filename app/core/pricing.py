"""Per-model token pricing table for calculating dollar savings."""


# Pricing in USD per 1 Million tokens: (prompt_cost_per_1M, completion_cost_per_1M)
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1-preview": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    
    # Anthropic
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-5-haiku": (1.00, 5.00),
}

# Fallback default pricing if model is unrecognized
DEFAULT_PRICING_PER_1M: tuple[float, float] = (2.00, 10.00)


def get_model_pricing(model: str) -> tuple[float, float]:
    """Retrieve (prompt_price_per_1M, completion_price_per_1M) for model."""
    norm_model = model.strip().lower()
    for key, pricing in MODEL_PRICING_PER_1M.items():
        if norm_model.startswith(key):
            return pricing
    return DEFAULT_PRICING_PER_1M


def calculate_cost_savings(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate estimated dollar savings from a cache hit."""
    p_rate, c_rate = get_model_pricing(model)
    prompt_savings = (prompt_tokens / 1_000_000.0) * p_rate
    completion_savings = (completion_tokens / 1_000_000.0) * c_rate
    return round(prompt_savings + completion_savings, 6)

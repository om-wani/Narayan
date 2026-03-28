"""Request cost estimates."""

from __future__ import annotations

from decimal import Decimal


MODEL_PRICING_USD_PER_1M_TOKENS = {
    "gpt-4o-mini": {"prompt": Decimal("0.15"), "completion": Decimal("0.60")},
    "gpt-4.1-mini": {"prompt": Decimal("0.40"), "completion": Decimal("1.60")},
    "text-embedding-3-small": {"prompt": Decimal("0.02"), "completion": Decimal("0.00")},
}


def estimate_request_cost(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> float:
    pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(model, MODEL_PRICING_USD_PER_1M_TOKENS["gpt-4o-mini"])
    cost = (
        (Decimal(prompt_tokens) * pricing["prompt"]) / Decimal(1_000_000)
        + (Decimal(completion_tokens) * pricing["completion"]) / Decimal(1_000_000)
    )
    return float(cost)


def estimate_embedding_cost(token_count: int, model: str = "text-embedding-3-small") -> float:
    pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(model, MODEL_PRICING_USD_PER_1M_TOKENS["text-embedding-3-small"])
    cost = (Decimal(token_count) * pricing["prompt"]) / Decimal(1_000_000)
    return float(cost)

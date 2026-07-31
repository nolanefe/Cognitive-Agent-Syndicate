"""User-supplied pricing configuration and cost estimation."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from cognitive_agent_syndicate.benchmarking.schemas import CostEstimate, PricingConfig


class PricingLoadError(ValueError):
    """Raised when pricing configuration cannot be loaded."""


_MILLION = Decimal("1000000")


def load_pricing_config(path: Path) -> PricingConfig:
    """Load pricing configuration from a JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PricingLoadError(f"Cannot read pricing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PricingLoadError(f"Invalid JSON in pricing file: {path}") from exc

    try:
        return PricingConfig.model_validate(payload)
    except ValidationError as exc:
        raise PricingLoadError(f"Invalid pricing configuration: {exc}") from exc


def estimate_trial_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: PricingConfig | None,
) -> CostEstimate | None:
    """Calculate trial cost from token usage and user-supplied pricing."""
    if pricing is None:
        return None

    input_cost = (Decimal(prompt_tokens) / _MILLION) * pricing.input_usd_per_million_tokens
    output_cost = (Decimal(completion_tokens) / _MILLION) * pricing.output_usd_per_million_tokens
    total = input_cost + output_cost
    return CostEstimate(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=total,
        pricing=pricing,
    )

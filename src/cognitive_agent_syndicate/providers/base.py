"""Provider protocol for structured model generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from cognitive_agent_syndicate.schemas import UsageMetrics

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class GenerationResult(Generic[T]):
    """Validated structured response and usage metrics from a provider call."""

    response: T
    usage: UsageMetrics


class ModelProvider(Protocol):
    """Async contract for generating typed structured responses."""

    async def generate(
        self,
        *,
        system_instructions: str,
        user_content: str,
        response_type: type[T],
    ) -> GenerationResult[T]:
        """Generate and validate a structured response of the requested type."""
        ...

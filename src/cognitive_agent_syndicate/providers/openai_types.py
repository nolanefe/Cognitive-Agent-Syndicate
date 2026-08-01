"""OpenAI client protocol types without importing the optional SDK."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OpenAIResponsesResource(Protocol):
    """Typed subset of ``client.responses`` used by the provider."""

    async def parse(
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        text_format: type[Any],
        max_output_tokens: int,
        store: bool,
        background: bool,
    ) -> Any:
        """Parse a structured response."""
        ...


@runtime_checkable
class OpenAIResponsesClient(Protocol):
    """Minimal AsyncOpenAI surface required for structured generation."""

    @property
    def responses(self) -> OpenAIResponsesResource:
        """Responses API resource."""
        ...

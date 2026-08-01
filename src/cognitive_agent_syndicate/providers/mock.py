"""Deterministic offline model provider for development and tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from cognitive_agent_syndicate.providers.base import GenerationResult, T
from cognitive_agent_syndicate.schemas import UsageMetrics


class MockResponseNotConfiguredError(LookupError):
    """Raised when the mock provider has no configured response for a call."""

    def __init__(
        self,
        response_type: type[BaseModel],
        user_content: str,
    ) -> None:
        self.response_type = response_type
        self.user_content = user_content
        super().__init__(
            f"No mock response configured for {response_type.__name__!r} "
            f"with user content: {user_content!r}"
        )


class MockResponseSequenceExhaustedError(LookupError):
    """Raised when a configured mock response sequence has no remaining entries."""

    def __init__(
        self,
        response_type: type[BaseModel],
        user_content: str,
    ) -> None:
        self.response_type = response_type
        self.user_content = user_content
        super().__init__(
            f"Mock response sequence exhausted for {response_type.__name__!r} "
            f"with user content: {user_content!r}"
        )


@dataclass
class RecordedCall:
    system_instructions: str
    user_content: str
    response_type: type[BaseModel]


@dataclass
class MockModelProvider:
    """Offline provider that returns preconfigured Pydantic responses."""

    usage: UsageMetrics = field(
        default_factory=lambda: UsageMetrics(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            latency_ms=1.5,
        )
    )
    _responses: dict[tuple[type[BaseModel], str | None], BaseModel] = field(default_factory=dict)
    _response_sequences: dict[type[BaseModel], list[BaseModel]] = field(default_factory=dict)
    calls: list[RecordedCall] = field(default_factory=list)

    def configure_response(
        self,
        response_type: type[BaseModel],
        response: BaseModel,
        *,
        user_content: str | None = None,
    ) -> None:
        """Register a deterministic response for a response type and optional user content."""
        if not isinstance(response, response_type):
            raise TypeError(
                f"Configured response must be an instance of {response_type.__name__}, "
                f"got {type(response).__name__}"
            )
        copied = response_type.model_validate(response.model_dump())
        self._responses[(response_type, user_content)] = copied

    def configure_response_sequence(
        self,
        response_type: type[BaseModel],
        responses: list[BaseModel],
    ) -> None:
        """Register ordered responses returned on successive generate calls."""
        validated = [response_type.model_validate(item.model_dump()) for item in responses]
        self._response_sequences[response_type] = validated

    async def generate(
        self,
        *,
        system_instructions: str,
        user_content: str,
        response_type: type[T],
    ) -> GenerationResult[T]:
        self.calls.append(
            RecordedCall(
                system_instructions=system_instructions,
                user_content=user_content,
                response_type=response_type,
            )
        )

        response = self._lookup_response(response_type, user_content)
        validated = response_type.model_validate(response.model_dump())
        usage = UsageMetrics.model_validate(self.usage.model_dump())
        return GenerationResult(response=validated, usage=usage)

    def _lookup_response(
        self,
        response_type: type[BaseModel],
        user_content: str,
    ) -> BaseModel:
        if response_type in self._response_sequences:
            sequence = self._response_sequences[response_type]
            if not sequence:
                raise MockResponseSequenceExhaustedError(response_type, user_content)
            next_response = sequence.pop(0)
            return response_type.model_validate(next_response.model_dump())

        exact_key = (response_type, user_content)
        if exact_key in self._responses:
            stored = self._responses[exact_key]
            return response_type.model_validate(stored.model_dump())

        fallback_key = (response_type, None)
        if fallback_key in self._responses:
            stored = self._responses[fallback_key]
            return response_type.model_validate(stored.model_dump())

        raise MockResponseNotConfiguredError(response_type, user_content)

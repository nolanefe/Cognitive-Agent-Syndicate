"""Application-level live provider smoke validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cognitive_agent_syndicate.config import ProviderName, Settings
from cognitive_agent_syndicate.providers.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from cognitive_agent_syndicate.providers.factory import (
    create_model_provider,
    injectable_provider_kwargs,
)
from cognitive_agent_syndicate.schemas import SystemBrief

if TYPE_CHECKING:
    from cognitive_agent_syndicate.providers.openai_types import OpenAIResponsesClient


@dataclass(frozen=True)
class LiveSmokeResult:
    """Structured metadata from a single live provider smoke call."""

    success: bool
    model: str
    provider: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    failure_category: str | None
    failure_reason: str | None


def _map_smoke_failure(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, ProviderAuthenticationError):
        return "provider_authentication", "OpenAI authentication failed."
    if isinstance(exc, ProviderRateLimitError):
        return "provider_rate_limit", "OpenAI rate limit exceeded."
    if isinstance(exc, ProviderTimeoutError):
        return "provider_timeout", "OpenAI request timed out."
    if isinstance(exc, ProviderMalformedResponseError):
        return "malformed_structured_output", "OpenAI structured output failed validation."
    if isinstance(exc, ProviderConfigurationError):
        return "provider_configuration", "Provider configuration is invalid."
    if isinstance(exc, ProviderError):
        return "provider_connection", "OpenAI provider request failed."
    return "internal_error", "Live smoke request failed."


async def run_live_provider_smoke(
    settings: Settings,
    *,
    client: OpenAIResponsesClient | None = None,
) -> LiveSmokeResult:
    """Run one structured-output smoke call through the production provider path."""
    provider_kwargs = injectable_provider_kwargs(client=client)
    provider = create_model_provider(settings, **provider_kwargs)
    model = settings.model.strip()
    try:
        result = await provider.generate(
            system_instructions="Return a concise structured brief.",
            user_content="Create a one-line offline smoke-test brief.",
            response_type=SystemBrief,
        )
    except Exception as exc:
        category, reason = _map_smoke_failure(exc)
        return LiveSmokeResult(
            success=False,
            model=model,
            provider=ProviderName.OPENAI.value,
            latency_ms=0.0,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            failure_category=category,
            failure_reason=reason,
        )

    usage = result.usage
    return LiveSmokeResult(
        success=True,
        model=model,
        provider=ProviderName.OPENAI.value,
        latency_ms=usage.latency_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        failure_category=None,
        failure_reason=None,
    )

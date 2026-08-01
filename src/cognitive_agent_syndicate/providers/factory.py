"""Provider factory for explicit mock and OpenAI configuration."""

from __future__ import annotations

import importlib.util
from typing import Any, cast

from cognitive_agent_syndicate.config import ProviderName, Settings
from cognitive_agent_syndicate.orchestration.clock import MonotonicClock, default_monotonic_clock
from cognitive_agent_syndicate.providers.base import ModelProvider
from cognitive_agent_syndicate.providers.errors import ProviderConfigurationError
from cognitive_agent_syndicate.providers.mock import MockModelProvider
from cognitive_agent_syndicate.providers.openai_types import OpenAIResponsesClient

OPENAI_INSTALL_INSTRUCTION = 'Install with: pip install -e ".[openai]"'

_injected_openai_client: OpenAIResponsesClient | None = None


def set_openai_client_injection(client: OpenAIResponsesClient | None) -> None:
    """Replace OpenAI client construction for tests (not a user-facing CLI option)."""
    global _injected_openai_client
    _injected_openai_client = client


def create_model_provider(
    settings: Settings,
    *,
    client: OpenAIResponsesClient | None = None,
    clock: MonotonicClock | None = None,
) -> ModelProvider:
    """Return a configured model provider without pipeline side effects."""
    validate_provider_configuration(settings)

    if settings.provider == ProviderName.MOCK:
        return MockModelProvider()

    return _create_openai_provider(settings, client=client, clock=clock)


def validate_provider_configuration(settings: Settings) -> None:
    """Validate provider settings with static, secret-safe error messages."""
    if settings.provider not in {ProviderName.MOCK, ProviderName.OPENAI}:
        raise ProviderConfigurationError("Unknown provider. Expected one of: mock, openai.")

    if settings.provider == ProviderName.MOCK:
        return

    if settings.resolved_openai_api_key() is None:
        raise ProviderConfigurationError("OpenAI provider requires a non-empty API key.")

    model = settings.model.strip()
    if not model or model == "mock-model":
        raise ProviderConfigurationError("OpenAI provider requires an explicit model name.")

    _ensure_openai_sdk_installed()


def _create_openai_provider(
    settings: Settings,
    *,
    client: OpenAIResponsesClient | None,
    clock: MonotonicClock | None,
) -> ModelProvider:
    openai_provider = _import_openai_provider_module()
    api_key_secret = settings.resolved_openai_api_key()
    assert api_key_secret is not None

    resolved_client = client or _injected_openai_client
    if resolved_client is None:
        resolved_client = openai_provider.create_openai_client(
            api_key=api_key_secret.get_secret_value(),
            timeout=settings.request_timeout,
        )

    return cast(
        ModelProvider,
        openai_provider.OpenAIModelProvider(
            model=settings.model.strip(),
            client=resolved_client,
            max_output_tokens=settings.max_output_tokens,
            clock=clock or default_monotonic_clock,
        ),
    )


def _ensure_openai_sdk_installed() -> None:
    if importlib.util.find_spec("openai") is None:
        raise ProviderConfigurationError(
            f"OpenAI provider requires the optional openai dependency. {OPENAI_INSTALL_INSTRUCTION}"
        )


def _import_openai_provider_module() -> Any:
    _ensure_openai_sdk_installed()
    try:
        return importlib.import_module("cognitive_agent_syndicate.providers.openai_provider")
    except ImportError as exc:
        raise ProviderConfigurationError(
            f"OpenAI provider requires the optional openai dependency. {OPENAI_INSTALL_INSTRUCTION}"
        ) from exc


def injectable_provider_kwargs(
    *,
    client: OpenAIResponsesClient | None = None,
    clock: MonotonicClock | None = None,
) -> dict[str, Any]:
    """Return kwargs suitable for ``create_model_provider`` in tests."""
    kwargs: dict[str, Any] = {}
    if client is not None:
        kwargs["client"] = client
    if clock is not None:
        kwargs["clock"] = clock
    return kwargs

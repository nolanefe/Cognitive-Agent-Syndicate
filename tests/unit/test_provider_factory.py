"""Unit tests for the provider factory."""

from __future__ import annotations

import pytest

from cognitive_agent_syndicate.config import ProviderName, Settings, build_settings
from cognitive_agent_syndicate.providers.errors import ProviderConfigurationError
from cognitive_agent_syndicate.providers.factory import (
    create_model_provider,
    validate_provider_configuration,
)
from cognitive_agent_syndicate.providers.mock import MockModelProvider
from cognitive_agent_syndicate.providers.openai_provider import OpenAIModelProvider
from tests.fixtures.openai_provider_fixtures import FakeAsyncOpenAIClient, FakeResponsesResource


def test_factory_returns_mock_provider() -> None:
    provider = create_model_provider(build_settings(provider=ProviderName.MOCK.value))

    assert isinstance(provider, MockModelProvider)


def test_factory_returns_openai_provider_with_injected_client() -> None:
    client = FakeAsyncOpenAIClient(responses=FakeResponsesResource())
    provider = create_model_provider(
        build_settings(
            provider=ProviderName.OPENAI.value,
            model="gpt-4o-mini",
            openai_api_key="sk-test",
        ),
        client=client,
    )

    assert isinstance(provider, OpenAIModelProvider)
    assert provider.model == "gpt-4o-mini"


def test_factory_rejects_unknown_provider() -> None:
    settings = Settings.model_construct(provider="anthropic")
    with pytest.raises(ProviderConfigurationError, match="Unknown provider"):
        validate_provider_configuration(settings)


def test_factory_rejects_openai_without_key() -> None:
    settings = Settings.model_construct(provider=ProviderName.OPENAI, model="gpt-4o-mini")
    with pytest.raises(ProviderConfigurationError, match="requires a non-empty API key"):
        create_model_provider(settings)


def test_factory_rejects_openai_without_model() -> None:
    settings = Settings.model_construct(
        provider=ProviderName.OPENAI,
        model="mock-model",
        openai_api_key="sk-test",
    )
    with pytest.raises(ProviderConfigurationError, match="requires an explicit model"):
        create_model_provider(settings)


def test_validate_provider_configuration_accepts_mock_without_key() -> None:
    validate_provider_configuration(build_settings(provider=ProviderName.MOCK.value))

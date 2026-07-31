"""Unit tests for OpenAI provider configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cognitive_agent_syndicate.config import ProviderName, Settings, build_settings
from cognitive_agent_syndicate.providers.errors import ProviderConfigurationError
from cognitive_agent_syndicate.providers.factory import validate_provider_configuration


def test_mock_provider_requires_no_key() -> None:
    settings = build_settings(provider=ProviderName.MOCK.value)

    assert settings.resolved_openai_api_key() is None


def test_openai_provider_requires_key_via_safe_validation() -> None:
    settings = build_settings(provider=ProviderName.OPENAI.value, model="gpt-4o-mini")
    with pytest.raises(ProviderConfigurationError, match="requires a non-empty API key"):
        validate_provider_configuration(settings)


def test_openai_provider_requires_explicit_model_via_safe_validation() -> None:
    settings = build_settings(
        provider=ProviderName.OPENAI.value,
        openai_api_key="sk-test",
    )
    with pytest.raises(ProviderConfigurationError, match="requires an explicit model"):
        validate_provider_configuration(settings)


@pytest.mark.parametrize("api_key", ["", "   ", None])
def test_settings_normalize_empty_openai_api_key_to_none(api_key: str | None) -> None:
    settings = Settings(_env_file=None, openai_api_key=api_key)

    assert settings.openai_api_key is None


def test_openai_api_key_precedence_over_legacy_api_key() -> None:
    settings = build_settings(
        provider=ProviderName.MOCK.value,
        openai_api_key="sk-openai",
        api_key="sk-legacy",
    )

    resolved = settings.resolved_openai_api_key()
    assert resolved is not None
    assert resolved.get_secret_value() == "sk-openai"


def test_legacy_api_key_used_when_openai_key_missing() -> None:
    settings = build_settings(provider=ProviderName.MOCK.value, api_key="sk-legacy")

    resolved = settings.resolved_openai_api_key()
    assert resolved is not None
    assert resolved.get_secret_value() == "sk-legacy"


def test_settings_secret_keys_are_not_exposed_in_repr_or_str() -> None:
    secret = "sk-settings-secret-value"
    settings = build_settings(openai_api_key=secret, api_key="sk-other")

    rendered = repr(settings) + str(settings)
    assert secret not in rendered
    assert "sk-other" not in rendered


def test_request_timeout_bounds() -> None:
    with pytest.raises(ValidationError):
        build_settings(request_timeout=0.5)

    with pytest.raises(ValidationError):
        build_settings(request_timeout=700.0)


def test_max_output_tokens_bounds() -> None:
    with pytest.raises(ValidationError):
        build_settings(max_output_tokens=0)

    with pytest.raises(ValidationError):
        build_settings(max_output_tokens=200_000)

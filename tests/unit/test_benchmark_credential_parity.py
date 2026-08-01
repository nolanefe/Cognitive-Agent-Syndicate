"""Offline tests for benchmark vs standalone OpenAI credential parity."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cognitive_agent_syndicate.benchmark_cli import _build_benchmark_settings
from cognitive_agent_syndicate.config import ProviderName, apply_settings_overrides, build_settings
from cognitive_agent_syndicate.providers.errors import ProviderConfigurationError
from cognitive_agent_syndicate.providers.factory import create_model_provider

CORRECT_KEY = "sk-test-correct"
WRONG_KEY = "sk-test-wrong"


@pytest.fixture
def planted_dual_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", CORRECT_KEY)
    monkeypatch.setenv("API_KEY", WRONG_KEY)


@pytest.fixture
def planted_legacy_key_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY", CORRECT_KEY)


def _capture_create_openai_client() -> tuple[list[str], object]:
    captured: list[str] = []

    def fake_create_openai_client(*, api_key: str, timeout: float) -> object:
        captured.append(api_key)
        return object()

    return captured, fake_create_openai_client


def test_standalone_provider_path_uses_openai_api_key_precedence(
    planted_dual_keys: None,
) -> None:
    captured, fake_client = _capture_create_openai_client()
    base = build_settings()
    settings = apply_settings_overrides(
        base,
        provider=ProviderName.OPENAI.value,
        model="gpt-test-model",
    )

    with patch(
        "cognitive_agent_syndicate.providers.openai_provider.create_openai_client",
        fake_client,
    ):
        create_model_provider(settings)

    assert captured == [CORRECT_KEY]


def test_benchmark_generation_settings_use_openai_api_key_precedence(
    planted_dual_keys: None,
) -> None:
    captured, fake_client = _capture_create_openai_client()
    settings = _build_benchmark_settings(
        provider=ProviderName.OPENAI,
        model="gpt-test-model",
        live=True,
    )

    with patch(
        "cognitive_agent_syndicate.providers.openai_provider.create_openai_client",
        fake_client,
    ):
        create_model_provider(settings)

    assert captured == [CORRECT_KEY]
    assert settings.model == "gpt-test-model"


def test_benchmark_reviewer_settings_preserve_resolved_key_when_env_cleared(
    planted_dual_keys: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_settings = _build_benchmark_settings(
        provider=ProviderName.OPENAI,
        model="gpt-test-model",
        live=True,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    reviewer_settings = apply_settings_overrides(
        generation_settings,
        provider=ProviderName.OPENAI.value,
        model="gpt-reviewer-model",
    )

    captured, fake_client = _capture_create_openai_client()
    with patch(
        "cognitive_agent_syndicate.providers.openai_provider.create_openai_client",
        fake_client,
    ):
        create_model_provider(reviewer_settings)

    assert captured == [CORRECT_KEY]
    assert reviewer_settings.model == "gpt-reviewer-model"


def test_legacy_api_key_fallback_matches_across_paths(planted_legacy_key_only: None) -> None:
    standalone = apply_settings_overrides(
        build_settings(),
        provider=ProviderName.OPENAI.value,
        model="gpt-test-model",
    )
    benchmark = _build_benchmark_settings(
        provider=ProviderName.OPENAI,
        model="gpt-test-model",
        live=True,
    )

    assert standalone.resolved_openai_api_key() is not None
    assert benchmark.resolved_openai_api_key() is not None
    assert (
        standalone.resolved_openai_api_key().get_secret_value()
        == benchmark.resolved_openai_api_key().get_secret_value()
        == CORRECT_KEY
    )


def test_missing_credentials_raise_safe_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    settings = _build_benchmark_settings(
        provider=ProviderName.OPENAI,
        model="gpt-test-model",
        live=True,
    )

    with pytest.raises(ProviderConfigurationError, match="non-empty API key"):
        create_model_provider(settings)


def test_contract_trial_settings_copy_preserves_resolved_key(
    planted_dual_keys: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _build_benchmark_settings(
        provider=ProviderName.OPENAI,
        model="gpt-test-model",
        live=True,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    trial_settings = base.model_copy(
        update={
            "max_repair_attempts": 1,
            "artifact_output_dir": (
                "generated_artifacts/benchmarks/test/task/contract_with_repair/1"
            ),
        }
    )

    assert trial_settings.resolved_openai_api_key() is not None
    assert trial_settings.resolved_openai_api_key().get_secret_value() == CORRECT_KEY


def test_apply_settings_overrides_does_not_copy_secrets_into_repr(
    planted_dual_keys: None,
) -> None:
    settings = _build_benchmark_settings(
        provider=ProviderName.OPENAI,
        model="gpt-test-model",
        live=True,
    )

    rendered = repr(settings) + str(settings)
    assert CORRECT_KEY not in rendered
    assert WRONG_KEY not in rendered

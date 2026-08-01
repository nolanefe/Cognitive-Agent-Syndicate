"""Unit tests for live validation credential handling."""

from __future__ import annotations

import os

import pytest

from cognitive_agent_syndicate.live_validation.credentials import (
    prompt_for_openai_api_key,
    resolve_existing_api_key_from_env,
    restore_credential_environment,
    scoped_live_environment,
    snapshot_credential_environment,
)
from cognitive_agent_syndicate.providers.errors import ProviderConfigurationError


def test_prompt_for_openai_api_key_requires_value() -> None:
    with pytest.raises(ProviderConfigurationError):
        prompt_for_openai_api_key(prompt_fn=lambda _prompt: "   ")


def test_getpass_path_when_env_key_absent(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    captured: list[str] = []

    def fake_prompt(_prompt: str) -> str:
        captured.append(_prompt)
        return "sk-test-entered"

    with scoped_live_environment(prompt_fn=fake_prompt):
        assert captured == ["OpenAI API key: "]
        assert os.environ.get("OPENAI_API_KEY") == "sk-test-entered"
        assert os.environ.get("RUN_LIVE_BENCHMARKS") == "1"

    assert "OPENAI_API_KEY" not in os.environ
    assert "RUN_LIVE_BENCHMARKS" not in os.environ


def test_existing_env_key_path(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-existing")
    prompted = False

    def fake_prompt(_prompt: str) -> str:
        nonlocal prompted
        prompted = True
        return "should-not-be-used"

    with scoped_live_environment(prompt_fn=fake_prompt):
        assert os.environ["OPENAI_API_KEY"] == "sk-existing"
        assert prompted is False

    assert os.environ.get("OPENAI_API_KEY") == "sk-existing"


def test_api_key_precedence_openai_over_legacy(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-preferred")
    monkeypatch.setenv("API_KEY", "sk-legacy")
    resolved = resolve_existing_api_key_from_env()
    assert resolved is not None
    assert resolved.get_secret_value() == "sk-preferred"


def test_restore_prior_env_after_success(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-original")
    monkeypatch.setenv("RUN_LIVE_BENCHMARKS", "0")
    snapshot = snapshot_credential_environment()
    os.environ["OPENAI_API_KEY"] = "sk-temporary"
    os.environ["RUN_LIVE_BENCHMARKS"] = "1"
    restore_credential_environment(snapshot)
    assert os.environ["OPENAI_API_KEY"] == "sk-original"
    assert os.environ["RUN_LIVE_BENCHMARKS"] == "0"


def test_restore_after_smoke_failure_restores_absent_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    try:
        with scoped_live_environment(prompt_fn=lambda _prompt: "sk-temp"):
            assert os.environ.get("OPENAI_API_KEY") == "sk-temp"
            raise RuntimeError("simulated smoke failure")
    except RuntimeError:
        pass
    assert "OPENAI_API_KEY" not in os.environ

"""Unit tests for application configuration."""

import pytest
from pydantic import ValidationError

from cognitive_agent_syndicate.config import Settings


def test_settings_use_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.provider == "mock"
    assert settings.model == "mock-model"
    assert settings.temperature == 0.0
    assert settings.max_output_chars == 50_000
    assert settings.max_generated_files == 20
    assert settings.max_repair_attempts == 1
    assert settings.artifact_output_dir == "generated_artifacts"
    assert settings.api_key is None


@pytest.mark.parametrize("max_repair_attempts", [0, 1])
def test_settings_accept_valid_repair_limits(max_repair_attempts: int) -> None:
    settings = Settings(_env_file=None, max_repair_attempts=max_repair_attempts)

    assert settings.max_repair_attempts == max_repair_attempts


def test_settings_reject_repair_limit_above_one() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_repair_attempts=2)


def test_settings_reject_invalid_numeric_limits() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_output_chars=50)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_generated_files=0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, temperature=3.0)


def test_settings_reject_unsafe_artifact_output_dir() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, artifact_output_dir="/tmp/artifacts")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, artifact_output_dir="../artifacts")


@pytest.mark.parametrize("api_key", ["", "   ", None])
def test_settings_normalize_empty_api_key_to_none(api_key: str | None) -> None:
    settings = Settings(_env_file=None, api_key=api_key)

    assert settings.api_key is None


def test_settings_strip_provider_and_model_names() -> None:
    settings = Settings(_env_file=None, provider="  mock  ", model="  mock-model  ")

    assert settings.provider == "mock"
    assert settings.model == "mock-model"


@pytest.mark.parametrize("field_name", ["provider", "model"])
def test_settings_reject_whitespace_only_names(field_name: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field_name: "   "})

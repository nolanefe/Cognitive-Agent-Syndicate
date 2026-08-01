"""Tests for optional OpenAI dependency isolation."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cognitive_agent_syndicate.config import ProviderName, build_settings
from cognitive_agent_syndicate.providers.errors import ProviderConfigurationError
from cognitive_agent_syndicate.providers.factory import create_model_provider

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_BRIEF_PATH = REPO_ROOT / "examples" / "briefs" / "url_shortener.json"
_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def test_mock_provider_creation_does_not_import_openai(monkeypatch) -> None:
    original_import = builtins.__import__
    imported: list[str] = []

    def tracking_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        imported.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)
    provider = create_model_provider(build_settings(provider=ProviderName.MOCK.value))

    assert provider is not None
    assert "openai" not in imported


def test_cli_help_works_when_openai_import_is_blocked(monkeypatch) -> None:
    original_import = builtins.__import__

    def blocking_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "openai" or name.startswith("openai."):
            raise ImportError("openai blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    from cognitive_agent_syndicate.cli import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output


def test_default_mock_cli_works_when_openai_import_is_blocked(tmp_path, monkeypatch) -> None:
    original_import = builtins.__import__

    def blocking_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "openai" or name.startswith("openai."):
            raise ImportError("openai blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    artifact_dir = tmp_path / "generated"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    from cognitive_agent_syndicate.cli import app

    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--mock",
            "--artifact-dir",
            "generated",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Pipeline succeeded" in result.output


def test_openai_selection_without_sdk_shows_install_instruction(monkeypatch) -> None:
    def fake_find_spec(name: str, package: str | None = None):  # type: ignore[no-untyped-def]
        if name == "openai":
            return None
        return _ORIGINAL_FIND_SPEC(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    settings = build_settings(
        provider=ProviderName.OPENAI.value,
        model="gpt-4o-mini",
        openai_api_key="sk-test",
    )
    with pytest.raises(ProviderConfigurationError, match="pip install -e"):
        create_model_provider(settings)


def test_openai_cli_without_sdk_shows_install_instruction(monkeypatch) -> None:
    def fake_find_spec(name: str, package: str | None = None):  # type: ignore[no-untyped-def]
        if name == "openai":
            return None
        return _ORIGINAL_FIND_SPEC(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    from cognitive_agent_syndicate.cli import app

    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
        ],
        env={"OPENAI_API_KEY": "sk-test"},
    )
    assert result.exit_code != 0
    normalized = " ".join(result.output.split())
    assert "pip install -e" in normalized
    assert "openai" in normalized
    assert "Traceback" not in result.output

"""Tests for secret-safe configuration and error handling."""

from __future__ import annotations

import importlib.util
import json

import httpx
import pytest
from openai import AuthenticationError
from typer.testing import CliRunner

from cognitive_agent_syndicate.agents.architect import ArchitectAgent
from cognitive_agent_syndicate.agents.implementer import ImplementerAgent
from cognitive_agent_syndicate.agents.reviewer import ReviewerAgent
from cognitive_agent_syndicate.cli import app
from cognitive_agent_syndicate.config import ProviderName, Settings, build_settings
from cognitive_agent_syndicate.demo import canonical_url_shortener_brief
from cognitive_agent_syndicate.orchestration.pipeline import ContractDrivenPipeline
from cognitive_agent_syndicate.providers.errors import ProviderConfigurationError
from cognitive_agent_syndicate.providers.factory import create_model_provider
from cognitive_agent_syndicate.reporting.report_writer import build_run_report
from tests.fixtures.openai_provider_fixtures import FakeAsyncOpenAIClient, FakeResponsesResource

runner = CliRunner()
PLANTED_SECRET = "sk-planted-secret-not-for-output"
_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def test_provider_configuration_error_excludes_planted_secret() -> None:
    settings = Settings.model_construct(provider=ProviderName.OPENAI, model="gpt-4o-mini")
    try:
        create_model_provider(settings)
    except ProviderConfigurationError as exc:
        assert PLANTED_SECRET not in str(exc)
    else:
        raise AssertionError("Expected ProviderConfigurationError")


def test_cli_error_output_excludes_planted_secret_when_sdk_missing(monkeypatch) -> None:
    def fake_find_spec(name: str, package: str | None = None):  # type: ignore[no-untyped-def]
        if name == "openai":
            return None
        return _ORIGINAL_FIND_SPEC(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    result = runner.invoke(
        app,
        [
            "run",
            "examples/briefs/url_shortener.json",
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
        ],
        env={"OPENAI_API_KEY": PLANTED_SECRET},
    )
    assert result.exit_code != 0
    assert PLANTED_SECRET not in result.output


@pytest.mark.asyncio
async def test_failure_report_excludes_planted_secret(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    response = httpx.Response(
        status_code=401,
        headers={"x-request-id": "req_fail"},
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    fake = FakeResponsesResource(
        response=AuthenticationError("auth failed", response=response, body={}),
    )
    client = FakeAsyncOpenAIClient(responses=fake)
    settings = build_settings(
        provider=ProviderName.OPENAI.value,
        model="gpt-4o-mini",
        openai_api_key=PLANTED_SECRET,
        artifact_output_dir="artifacts",
    )
    provider = create_model_provider(settings, client=client)
    pipeline = ContractDrivenPipeline(
        architect=ArchitectAgent(provider),
        implementer=ImplementerAgent(provider),
        reviewer=ReviewerAgent(provider),
        settings=settings,
    )

    state = await pipeline.run(
        canonical_url_shortener_brief(),
        allowed_technologies=["python"],
        permitted_paths=["pyproject.toml", "src", "tests"],
        implementation_constraints=["No secrets."],
        required_project_files=["pyproject.toml"],
    )

    report_json = json.dumps(build_run_report(state, generated_files=[]).model_dump(mode="json"))
    assert PLANTED_SECRET not in report_json
    assert PLANTED_SECRET not in state.model_dump_json()

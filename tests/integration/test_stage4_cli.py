"""Integration tests for Stage 4 CLI provider selection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from cognitive_agent_syndicate.cli import app
from cognitive_agent_syndicate.config import ProviderName
from cognitive_agent_syndicate.demo import MockScenario, create_demo_provider
from cognitive_agent_syndicate.providers.factory import set_openai_client_injection
from cognitive_agent_syndicate.schemas import ArchitectureSpec, ArtifactBundle, ReviewReport
from tests.fixtures.openai_provider_fixtures import (
    FakeAsyncOpenAIClient,
    FakeResponsesResource,
    build_parsed_response,
    sample_usage,
)

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_BRIEF_PATH = REPO_ROOT / "examples" / "briefs" / "url_shortener.json"


def _success_responses() -> list[object]:
    mock = create_demo_provider(scenario=MockScenario.SUCCESS)
    return [
        build_parsed_response(
            parsed=mock._responses[(ArchitectureSpec, None)],
            usage=sample_usage(input_tokens=10, output_tokens=5),
        ),
        build_parsed_response(
            parsed=mock._responses[(ArtifactBundle, None)],
            usage=sample_usage(input_tokens=12, output_tokens=6),
        ),
        build_parsed_response(
            parsed=mock._responses[(ReviewReport, None)],
            usage=sample_usage(input_tokens=8, output_tokens=4),
        ),
    ]


def _sequenced_client(responses: list[object]) -> FakeAsyncOpenAIClient:
    call_index = {"value": 0}
    resource = FakeResponsesResource()

    async def rotating_parse(**kwargs: object):
        resource.calls.append(type("Call", (), {"kwargs": kwargs})())
        idx = call_index["value"]
        call_index["value"] += 1
        return responses[idx]

    resource.parse = rotating_parse  # type: ignore[method-assign]
    return FakeAsyncOpenAIClient(responses=resource)


def test_cli_default_mock_path_succeeds(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "generated"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--artifact-dir",
            "generated",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Pipeline succeeded" in result.output


def test_cli_explicit_mock_succeeds(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "generated"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--provider",
            "mock",
            "--artifact-dir",
            "generated",
        ],
    )

    assert result.exit_code == 0, result.output


def test_cli_openai_without_key_fails_before_request() -> None:
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
    )

    assert result.exit_code != 0
    assert "non-empty API key" in result.output


def test_cli_openai_requires_model() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--provider",
            "openai",
        ],
        env={"OPENAI_API_KEY": "sk-test"},
    )

    assert result.exit_code != 0
    assert "requires --model" in result.output


def test_cli_rejects_mock_scenario_with_openai() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--mock-scenario",
            "repair-success",
        ],
        env={"OPENAI_API_KEY": "sk-test"},
    )

    assert result.exit_code != 0
    assert "mock-scenario is valid only with the mock provider" in result.output


def test_cli_rejects_mock_and_openai_provider_together() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--mock",
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
        ],
        env={"OPENAI_API_KEY": "sk-test"},
    )

    assert result.exit_code != 0
    assert "--mock cannot be combined with --provider openai" in result.output


def test_cli_rejects_invalid_provider() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--provider",
            "anthropic",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid provider" in result.output


def test_cli_openai_whitespace_key_treated_as_missing() -> None:
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
        env={"OPENAI_API_KEY": "   "},
    )

    assert result.exit_code != 0
    assert "non-empty API key" in result.output


def test_cli_openai_success_with_injected_client(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "generated"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    fake_client = _sequenced_client(_success_responses())
    set_openai_client_injection(fake_client)

    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--provider",
            ProviderName.OPENAI.value,
            "--model",
            "test-model",
            "--artifact-dir",
            "generated",
        ],
        env={"OPENAI_API_KEY": "sk-test"},
    )

    assert result.exit_code == 0, result.output
    assert "Pipeline succeeded" in result.output
    assert "sk-test" not in result.output
    assert len(fake_client.responses.calls) == 3
    assert fake_client.responses.calls[0].kwargs["model"] == "test-model"
    assert fake_client.responses.calls[0].kwargs["store"] is False

    run_dirs = list(artifact_dir.iterdir())
    assert len(run_dirs) == 1
    report = json.loads((run_dirs[0] / "run-report.json").read_text(encoding="utf-8"))
    assert report["success"] is True
    assert "sk-test" not in json.dumps(report)


def test_cli_repeated_in_process_calls_remain_stable(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "generated"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    args = [
        "run",
        str(CANONICAL_BRIEF_PATH),
        "--mock",
        "--artifact-dir",
        "generated",
    ]
    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output


def test_missing_key_cli_subprocess_uses_sanitized_env() -> None:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cognitive_agent_syndicate",
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "non-empty API key" in result.stdout + result.stderr

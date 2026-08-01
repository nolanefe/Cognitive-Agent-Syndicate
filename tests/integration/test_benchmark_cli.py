"""Integration tests for benchmark CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import AuthenticationError
from typer.testing import CliRunner

from cognitive_agent_syndicate.cli import app
from tests.fixtures.openai_provider_fixtures import FakeAsyncOpenAIClient, FakeResponsesResource

runner = CliRunner()
CORRECT_KEY = "sk-test-correct"
WRONG_KEY = "sk-test-wrong"


def test_benchmark_plan_succeeds_offline() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "plan",
            "--dataset",
            "benchmarks/datasets/software_delivery_v1.json",
            "--modes",
            "single_agent,contract_no_repair,contract_with_repair",
            "--repetitions",
            "1",
            "--provider",
            "mock",
        ],
    )
    assert result.exit_code == 0
    assert "Total trials: 18" in result.stdout
    assert "Dataset: software_delivery v1" in result.stdout
    assert "vv1" not in result.stdout
    assert "Min provider calls: 48" in result.stdout
    assert "Max provider calls: 60" in result.stdout
    assert "no provider calls" in result.stdout.lower()


def test_mock_benchmark_run_succeeds(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(Path.cwd())
    output_dir = tmp_path / "benchmark_results"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--dataset",
            "benchmarks/datasets/software_delivery_v1.json",
            "--modes",
            "single_agent",
            "--repetitions",
            "1",
            "--provider",
            "mock",
            "--output-dir",
            str(output_dir),
            "--task-ids",
            "task-url-shortener",
            "--benchmark-id",
            "cli-mock-test",
        ],
    )
    assert result.exit_code == 0
    assert (output_dir / "cli-mock-test" / "summary.json").exists()
    assert "Mock benchmark results validate" in result.stdout


def test_invalid_dataset_fails() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "plan",
            "--dataset",
            "missing-dataset.json",
        ],
    )
    assert result.exit_code == 1


def test_unknown_option_exits_usage_code() -> None:
    result = runner.invoke(app, ["benchmark", "plan", "--not-a-real-flag"])
    assert result.exit_code == 2


def test_invalid_benchmark_id_exits_fatal() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--benchmark-id",
            "../bad",
            "--task-ids",
            "task-url-shortener",
            "--modes",
            "single_agent",
        ],
    )
    assert result.exit_code == 1


def test_all_success_mock_run_exits_zero(tmp_path) -> None:
    output_dir = tmp_path / "benchmark_results"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--dataset",
            "benchmarks/datasets/software_delivery_v1.json",
            "--modes",
            "single_agent",
            "--repetitions",
            "1",
            "--provider",
            "mock",
            "--output-dir",
            str(output_dir),
            "--task-ids",
            "task-url-shortener",
            "--benchmark-id",
            "cli-all-success",
        ],
    )
    assert result.exit_code == 0
    assert (output_dir / "cli-all-success" / "summary.json").exists()


def test_full_mock_run_exits_three_with_persisted_outputs(tmp_path) -> None:
    output_dir = tmp_path / "benchmark_results"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--dataset",
            "benchmarks/datasets/software_delivery_v1.json",
            "--modes",
            "single_agent,contract_no_repair,contract_with_repair",
            "--repetitions",
            "1",
            "--provider",
            "mock",
            "--output-dir",
            str(output_dir),
            "--benchmark-id",
            "cli-partial-failures",
        ],
    )
    assert result.exit_code == 3
    assert (output_dir / "cli-partial-failures" / "summary.json").exists()
    assert "Mock benchmark results validate" in result.stdout


def test_invalid_mode_fails() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "plan",
            "--modes",
            "not_valid",
        ],
    )
    assert result.exit_code == 1


def test_task_filtering_works() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "plan",
            "--task-ids",
            "task-url-shortener,task-feature-flag",
            "--repetitions",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "Total trials: 6" in result.stdout


def test_existing_output_rejected(tmp_path) -> None:
    output_dir = tmp_path / "benchmark_results"
    existing = output_dir / "dup-id"
    existing.mkdir(parents=True)
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--output-dir",
            str(output_dir),
            "--benchmark-id",
            "dup-id",
            "--task-ids",
            "task-url-shortener",
            "--modes",
            "single_agent",
        ],
    )
    assert result.exit_code != 0
    assert "already exists" in result.stdout.lower()


def test_openai_without_live_opt_in_rejected() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--provider",
            "openai",
            "--model",
            "gpt-test",
            "--task-ids",
            "task-url-shortener",
        ],
    )
    assert result.exit_code == 1


def test_confirm_live_insufficient_without_env(monkeypatch) -> None:
    monkeypatch.delenv("RUN_LIVE_BENCHMARKS", raising=False)
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--provider",
            "openai",
            "--model",
            "gpt-test",
            "--confirm-live",
            "--task-ids",
            "task-url-shortener",
        ],
    )
    assert result.exit_code == 1
    assert "RUN_LIVE_BENCHMARKS" in result.stdout


def test_inherited_env_cannot_activate_live_in_tests(monkeypatch) -> None:
    monkeypatch.setenv("RUN_LIVE_BENCHMARKS", "1")
    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--provider",
            "openai",
            "--model",
            "gpt-test",
            "--task-ids",
            "task-url-shortener",
        ],
    )
    assert result.exit_code == 1
    assert "confirm-live" in result.stdout.lower()


def test_fairness_same_gates_across_modes_in_plan() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "plan",
            "--modes",
            "single_agent,contract_no_repair,contract_with_repair",
        ],
    )
    assert result.exit_code == 0
    assert "single_agent" in result.stdout
    assert "contract_with_repair" in result.stdout


def test_openai_benchmark_provider_paths_use_resolved_openai_api_key(tmp_path) -> None:
    captured_keys: list[str] = []
    auth_error = AuthenticationError(
        "auth failed",
        response=httpx.Response(
            status_code=401,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        ),
        body={},
    )
    fake_client = FakeAsyncOpenAIClient(
        responses=FakeResponsesResource(response=auth_error),
    )

    def fake_create_openai_client(*, api_key: str, timeout: float) -> FakeAsyncOpenAIClient:
        captured_keys.append(api_key)
        return fake_client

    output_dir = tmp_path / "benchmark_results"
    env = {
        "OPENAI_API_KEY": CORRECT_KEY,
        "API_KEY": WRONG_KEY,
        "RUN_LIVE_BENCHMARKS": "1",
    }

    with patch(
        "cognitive_agent_syndicate.providers.openai_provider.create_openai_client",
        fake_create_openai_client,
    ):
        result = runner.invoke(
            app,
            [
                "benchmark",
                "run",
                "--dataset",
                "benchmarks/datasets/software_delivery_v1.json",
                "--modes",
                "single_agent",
                "--repetitions",
                "1",
                "--provider",
                "openai",
                "--model",
                "gpt-test-model",
                "--reviewer-model",
                "gpt-reviewer-model",
                "--confirm-live",
                "--output-dir",
                str(output_dir),
                "--task-ids",
                "task-url-shortener",
                "--benchmark-id",
                "cli-openai-credential-test",
            ],
            env=env,
        )

    assert captured_keys
    assert all(key == CORRECT_KEY for key in captured_keys)
    assert CORRECT_KEY not in result.output
    assert WRONG_KEY not in result.output

    config_path = output_dir / "cli-openai-credential-test" / "benchmark-config.json"
    assert config_path.exists()
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)
    assert config["generation_provider_label"] == "openai"
    assert config["model_label"] == "gpt-test-model"
    assert config["reviewer_provider_label"] == "openai"
    assert config["reviewer_model_label"] == "gpt-reviewer-model"
    assert config["is_mock"] is False
    assert CORRECT_KEY not in config_text
    assert WRONG_KEY not in config_text

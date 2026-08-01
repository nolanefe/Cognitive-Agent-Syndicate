"""Integration tests for validate-live CLI and orchestrator."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from cognitive_agent_syndicate.benchmarking.exit_codes import (
    EXIT_CANCELLED,
    EXIT_COMPLETED_WITH_FAILURES,
    EXIT_FATAL,
    EXIT_SUCCESS,
)
from cognitive_agent_syndicate.benchmarking.ids import validate_benchmark_id
from cognitive_agent_syndicate.benchmarking.mock_fixtures import create_benchmark_mock_provider
from cognitive_agent_syndicate.cli import app
from cognitive_agent_syndicate.live_validation.ids import generate_live_benchmark_id
from cognitive_agent_syndicate.live_validation.orchestrator import run_live_validation
from cognitive_agent_syndicate.live_validation.smoke import LiveSmokeResult
from cognitive_agent_syndicate.validate_live_cli import (
    ValidateLiveCancellation,
    build_validate_live_signal_handler,
)

runner = CliRunner()
CORRECT_KEY = "sk-test-correct"


def _successful_smoke(**overrides: object) -> LiveSmokeResult:
    return LiveSmokeResult(
        success=True,
        model="gpt-test",
        provider="openai",
        latency_ms=12.0,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        failure_category=None,
        failure_reason=None,
        **{k: v for k, v in overrides.items() if k in LiveSmokeResult.__dataclass_fields__},
    )


def _failed_smoke(
    category: str = "provider_authentication",
    reason: str = "OpenAI authentication failed.",
) -> LiveSmokeResult:
    return LiveSmokeResult(
        success=False,
        model="gpt-test",
        provider="openai",
        latency_ms=0.0,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        failure_category=category,
        failure_reason=reason,
    )


def _mock_provider_factory(task, mode):
    return create_benchmark_mock_provider(task, mode)


@pytest.mark.asyncio
async def test_orchestrator_one_task_three_modes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", CORRECT_KEY)
    output_dir = tmp_path / "benchmark_results"
    outcome = await run_live_validation(
        task_ids="task-url-shortener",
        modes="single_agent,contract_no_repair,contract_with_repair",
        repetitions=1,
        model="gpt-test",
        output_dir=str(output_dir),
        benchmark_id="live-offline-run",
        confirm_live=True,
        allow_dirty=True,
        smoke_runner=lambda _settings: _successful_smoke(),
        generation_provider_factory=_mock_provider_factory,
        reviewer_provider_factory=_mock_provider_factory,
    )
    assert outcome.exit_code in {EXIT_SUCCESS, EXIT_COMPLETED_WITH_FAILURES}
    assert outcome.run is not None
    assert len(outcome.run.trials) == 3
    assert outcome.results_path is not None
    assert (outcome.results_path / "summary.md").exists()
    assert (outcome.results_path / "live-validation.json").exists()
    assert CORRECT_KEY not in (outcome.results_path / "live-validation.json").read_text()
    assert outcome.handoff_text is not None
    assert "LIVE VALIDATION COMPLETE" in outcome.handoff_text
    assert "summary.md" in outcome.handoff_text


@pytest.mark.asyncio
async def test_smoke_failure_does_not_start_benchmark(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", CORRECT_KEY)
    output_dir = tmp_path / "benchmark_results"

    async def boom(_settings):
        return _failed_smoke()

    outcome = await run_live_validation(
        task_ids="task-url-shortener",
        modes="single_agent",
        repetitions=1,
        model="gpt-test",
        output_dir=str(output_dir),
        benchmark_id="live-smoke-fail",
        confirm_live=True,
        allow_dirty=True,
        smoke_runner=boom,
        generation_provider_factory=_mock_provider_factory,
        reviewer_provider_factory=_mock_provider_factory,
    )
    assert outcome.exit_code == EXIT_FATAL
    assert outcome.run is None
    assert outcome.handoff_text is not None
    assert "LIVE SMOKE FAILED" in outcome.handoff_text
    assert not (output_dir / "live-smoke-fail").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "reason"),
    [
        ("provider_authentication", "OpenAI authentication failed."),
        ("provider_timeout", "OpenAI request timed out."),
        ("provider_rate_limit", "OpenAI rate limit exceeded."),
    ],
)
async def test_smoke_failures_skip_benchmark_and_restore_credentials(
    tmp_path: Path,
    monkeypatch,
    category: str,
    reason: str,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    benchmark_calls: list[int] = []

    async def failing_smoke(_settings):
        return _failed_smoke(category=category, reason=reason)

    def benchmark_factory(task, mode):
        benchmark_calls.append(1)
        return create_benchmark_mock_provider(task, mode)

    outcome = await run_live_validation(
        task_ids="task-url-shortener",
        modes="single_agent",
        repetitions=1,
        model="gpt-test",
        output_dir=str(tmp_path / "benchmark_results"),
        benchmark_id=f"live-smoke-{category}",
        confirm_live=True,
        allow_dirty=True,
        smoke_runner=failing_smoke,
        prompt_fn=lambda _prompt: CORRECT_KEY,
        generation_provider_factory=benchmark_factory,
        reviewer_provider_factory=benchmark_factory,
    )
    assert outcome.exit_code == EXIT_FATAL
    assert outcome.run is None
    assert benchmark_calls == []
    assert outcome.handoff_text is not None
    assert "LIVE SMOKE FAILED" in outcome.handoff_text
    assert category in outcome.handoff_text
    assert CORRECT_KEY not in outcome.handoff_text
    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.asyncio
async def test_confirm_live_blocks_smoke_runner_before_invocation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", CORRECT_KEY)
    smoke_calls: list[int] = []

    async def smoke_spy(_settings):
        smoke_calls.append(1)
        return _successful_smoke()

    outcome = await run_live_validation(
        task_ids="task-url-shortener",
        modes="single_agent",
        repetitions=1,
        model="gpt-test",
        output_dir=str(tmp_path / "benchmark_results"),
        confirm_live=False,
        allow_dirty=True,
        smoke_runner=smoke_spy,
    )
    assert smoke_calls == []
    assert outcome.exit_code == EXIT_FATAL
    assert outcome.smoke is None


@pytest.mark.asyncio
async def test_benchmark_exception_exits_fatal_without_complete_handoff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", CORRECT_KEY)
    output_dir = tmp_path / "benchmark_results"

    async def smoke_ok(_settings):
        return _successful_smoke()

    with patch(
        "cognitive_agent_syndicate.live_validation.orchestrator.execute_benchmark",
        side_effect=RuntimeError("injected benchmark failure"),
    ):
        outcome = await run_live_validation(
            task_ids="task-url-shortener",
            modes="single_agent",
            repetitions=1,
            model="gpt-test",
            output_dir=str(output_dir),
            benchmark_id="live-benchmark-exception",
            confirm_live=True,
            allow_dirty=True,
            smoke_runner=smoke_ok,
            generation_provider_factory=_mock_provider_factory,
            reviewer_provider_factory=_mock_provider_factory,
        )

    assert outcome.exit_code == EXIT_FATAL
    assert outcome.handoff_text is not None
    assert "LIVE VALIDATION COMPLETE" not in outcome.handoff_text
    assert "injected benchmark failure" in outcome.handoff_text
    assert not (output_dir / "live-benchmark-exception" / "live-validation.json").exists()
    assert os.environ.get("OPENAI_API_KEY") == CORRECT_KEY


def test_cli_signal_handler_invoked_twice_preserves_cleanup_path() -> None:
    cancellation = ValidateLiveCancellation()
    handler = build_validate_live_signal_handler(cancellation)
    handler(signal.SIGINT, None)
    handler(signal.SIGINT, None)
    assert cancellation.event.is_set()


def test_cli_missing_model_exits_fatal() -> None:
    result = runner.invoke(
        app,
        [
            "validate-live",
            "--confirm-live",
            "--task-ids",
            "task-url-shortener",
        ],
    )
    assert result.exit_code == EXIT_FATAL
    assert "--model" in result.stdout


def test_cli_confirm_live_required_before_live_calls() -> None:
    result = runner.invoke(
        app,
        [
            "validate-live",
            "--model",
            "gpt-test",
            "--task-ids",
            "task-url-shortener",
        ],
    )
    assert result.exit_code == EXIT_FATAL
    assert "confirm-live" in result.stdout.lower()


def test_cli_getpass_when_env_key_absent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    output_dir = tmp_path / "benchmark_results"

    async def orchestrated(**kwargs: Any):
        return await run_live_validation(
            **kwargs,
            smoke_runner=lambda _settings: _successful_smoke(),
            generation_provider_factory=_mock_provider_factory,
            reviewer_provider_factory=_mock_provider_factory,
            prompt_fn=lambda _prompt: CORRECT_KEY,
        )

    with patch(
        "cognitive_agent_syndicate.validate_live_cli.run_live_validation",
        side_effect=orchestrated,
    ):
        result = runner.invoke(
            app,
            [
                "validate-live",
                "--confirm-live",
                "--allow-dirty",
                "--model",
                "gpt-test",
                "--task-ids",
                "task-url-shortener",
                "--modes",
                "single_agent",
                "--output-dir",
                str(output_dir),
                "--benchmark-id",
                "live-cli-getpass",
            ],
        )

    assert result.exit_code in {EXIT_SUCCESS, EXIT_COMPLETED_WITH_FAILURES}
    assert CORRECT_KEY not in result.output
    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.asyncio
async def test_benchmark_id_generated(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", CORRECT_KEY)
    output_dir = tmp_path / "benchmark_results"
    outcome = await run_live_validation(
        task_ids="task-url-shortener",
        modes="single_agent",
        repetitions=2,
        model="gpt-test",
        output_dir=str(output_dir),
        confirm_live=True,
        allow_dirty=True,
        smoke_runner=lambda _settings: _successful_smoke(),
        generation_provider_factory=_mock_provider_factory,
        reviewer_provider_factory=_mock_provider_factory,
        generate_benchmark_id=lambda task_ids, reps: generate_live_benchmark_id(task_ids, reps),
    )
    assert outcome.preflight is not None
    validate_benchmark_id(outcome.preflight.benchmark_id)


@pytest.mark.asyncio
async def test_cancelled_does_not_fabricate_trials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", CORRECT_KEY)
    output_dir = tmp_path / "benchmark_results"
    cancel_after = {"count": 0}

    def cancelled_check() -> bool:
        cancel_after["count"] += 1
        return cancel_after["count"] > 1

    outcome = await run_live_validation(
        task_ids="task-url-shortener",
        modes="single_agent,contract_no_repair",
        repetitions=1,
        model="gpt-test",
        output_dir=str(output_dir),
        benchmark_id="live-cancel-test",
        confirm_live=True,
        allow_dirty=True,
        smoke_runner=lambda _settings: _successful_smoke(),
        generation_provider_factory=_mock_provider_factory,
        reviewer_provider_factory=_mock_provider_factory,
        cancelled_check=cancelled_check,
    )
    assert outcome.run is not None
    assert len(outcome.run.trials) == 1
    assert outcome.exit_code == EXIT_CANCELLED


def test_live_validation_json_has_no_secrets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", CORRECT_KEY)

    async def _run() -> None:
        output_dir = tmp_path / "benchmark_results"
        outcome = await run_live_validation(
            task_ids="task-url-shortener",
            modes="single_agent",
            repetitions=1,
            model="gpt-test",
            output_dir=str(output_dir),
            benchmark_id="live-json-safe",
            confirm_live=True,
            allow_dirty=True,
            smoke_runner=lambda _settings: _successful_smoke(),
            generation_provider_factory=_mock_provider_factory,
            reviewer_provider_factory=_mock_provider_factory,
        )
        assert outcome.results_path is not None
        payload = json.loads((outcome.results_path / "live-validation.json").read_text())
        dumped = json.dumps(payload)
        assert CORRECT_KEY not in dumped

    __import__("asyncio").run(_run())

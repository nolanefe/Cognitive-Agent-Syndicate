"""Tests ensuring live provider tests remain opt-in only."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _minimal_subprocess_env(**overrides: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.update(overrides)
    return env


def test_default_pytest_deselects_live_tests() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/live/", "-q"],
        cwd=REPO_ROOT,
        env=_minimal_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert "deselected" in combined.lower()
    assert "Set RUN_LIVE_TESTS=1" not in combined


def test_default_pytest_does_not_run_live_test_even_with_run_live_tests_env() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/live/", "-q"],
        cwd=REPO_ROOT,
        env=_minimal_subprocess_env(
            RUN_LIVE_TESTS="1",
            OPENAI_API_KEY="sk-should-not-run-live",
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert "deselected" in combined.lower()
    assert "sk-should-not-run-live" not in combined
    assert "Set RUN_LIVE_TESTS=1" not in combined


def test_explicit_live_run_passes_run_live_tests_guard() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/live/test_live_env_preservation.py",
            "-m",
            "live",
            "-q",
            "-rs",
        ],
        cwd=REPO_ROOT,
        env=_minimal_subprocess_env(
            RUN_LIVE_TESTS="1",
            OPENAI_API_KEY="sk-fake-opt-in-key",
            OPENAI_LIVE_MODEL="gpt-fake-live-model",
            RUN_LIVE_BENCHMARKS="1",
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Set RUN_LIVE_TESTS=1" not in combined
    assert "SKIPPED" not in combined
    assert "1 passed" in combined


def test_autouse_fixture_clears_sensitive_env_for_non_live_tests() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_ambient_env_isolation_probe.py",
            "-q",
        ],
        cwd=REPO_ROOT,
        env=_minimal_subprocess_env(
            RUN_LIVE_TESTS="1",
            OPENAI_API_KEY="sk-planted",
            API_KEY="planted-api-key",
            RUN_LIVE_BENCHMARKS="1",
            OPENAI_LIVE_MODEL="gpt-planted",
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "1 passed" in combined


def test_live_benchmark_safety_unchanged_in_integration_tests(monkeypatch) -> None:
    from typer.testing import CliRunner

    from cognitive_agent_syndicate.cli import app

    runner = CliRunner()
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

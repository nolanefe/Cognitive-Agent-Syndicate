"""Tests ensuring live provider tests remain opt-in only."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_pytest_does_not_run_live_test_even_with_run_live_tests_env() -> None:
    env = {
        "RUN_LIVE_TESTS": "1",
        "OPENAI_API_KEY": "sk-should-not-run-live",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/live/test_openai_provider_live.py", "-q"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert "1 deselected" in combined or "deselected" in combined.lower()
    assert "sk-should-not-run-live" not in combined

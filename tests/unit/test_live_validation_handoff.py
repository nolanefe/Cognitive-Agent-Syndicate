"""Unit tests for atomic live-validation.json persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cognitive_agent_syndicate.live_validation.handoff import write_live_validation_json
from cognitive_agent_syndicate.live_validation.preflight import GitMetadata
from cognitive_agent_syndicate.live_validation.smoke import LiveSmokeResult

SECRET = "sk-test-secret-value"


def _smoke() -> LiveSmokeResult:
    return LiveSmokeResult(
        success=True,
        model="gpt-test",
        provider="openai",
        latency_ms=1.0,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        failure_category=None,
        failure_reason=None,
    )


def _git() -> GitMetadata:
    return GitMetadata(available=False, commit_sha=None, branch=None, working_tree_clean=None)


def test_atomic_write_creates_valid_final_file(tmp_path: Path) -> None:
    output_path = tmp_path / "live-validation.json"
    write_live_validation_json(
        output_path=output_path,
        benchmark_id="live-atomic",
        smoke=_smoke(),
        git=_git(),
        run=None,
        results_dir=None,
        final_status="smoke_only",
        benchmark_exit_status=None,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["benchmark_id"] == "live-atomic"
    assert not list(tmp_path.glob("*.tmp"))


def test_replace_failure_leaves_existing_final_file_intact(tmp_path: Path) -> None:
    output_path = tmp_path / "live-validation.json"
    original = '{"schema_version": "1.0", "benchmark_id": "keep-me"}\n'
    output_path.write_text(original, encoding="utf-8")

    with patch(
        "cognitive_agent_syndicate.live_validation.handoff.os.replace",
        side_effect=OSError("fail"),
    ):
        with pytest.raises(OSError, match="fail"):
            write_live_validation_json(
                output_path=output_path,
                benchmark_id="live-replace-fail",
                smoke=_smoke(),
                git=_git(),
                run=None,
                results_dir=None,
                final_status="failed",
                benchmark_exit_status=1,
            )

    assert output_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_write_cleans_up_temporary_file(tmp_path: Path) -> None:
    output_path = tmp_path / "live-validation.json"

    with patch(
        "cognitive_agent_syndicate.live_validation.handoff.os.fdopen",
        side_effect=OSError("write failed"),
    ):
        with pytest.raises(OSError, match="write failed"):
            write_live_validation_json(
                output_path=output_path,
                benchmark_id="live-temp-cleanup",
                smoke=_smoke(),
                git=_git(),
                run=None,
                results_dir=None,
                final_status="failed",
                benchmark_exit_status=1,
            )

    assert not output_path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_live_validation_json_excludes_secrets(tmp_path: Path) -> None:
    output_path = tmp_path / "live-validation.json"
    write_live_validation_json(
        output_path=output_path,
        benchmark_id="live-secret-safe",
        smoke=_smoke(),
        git=_git(),
        run=None,
        results_dir=None,
        final_status="smoke_only",
        benchmark_exit_status=None,
    )
    assert SECRET not in output_path.read_text(encoding="utf-8")

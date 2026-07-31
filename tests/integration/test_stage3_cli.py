"""CLI tests for Stage 3 mock scenarios."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from cognitive_agent_syndicate.cli import app
from cognitive_agent_syndicate.demo import canonical_url_shortener_brief

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_BRIEF_PATH = REPO_ROOT / "examples" / "briefs" / "url_shortener.json"


def test_cli_success_scenario(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "generated"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--mock",
            "--mock-scenario",
            "success",
            "--max-repair-attempts",
            "1",
            "--artifact-dir",
            "generated",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Pipeline succeeded" in result.output


def test_cli_repair_success_scenario(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "generated"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--mock",
            "--mock-scenario",
            "repair-success",
            "--max-repair-attempts",
            "1",
            "--artifact-dir",
            "generated",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Repair attempt succeeded" in result.output


def test_cli_repair_failure_scenario_exits_non_zero(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "generated"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--mock",
            "--mock-scenario",
            "repair-failure",
            "--max-repair-attempts",
            "1",
            "--artifact-dir",
            "generated",
        ],
    )

    assert result.exit_code != 0, result.output
    assert "Pipeline failed" in result.output
    run_dirs = list(artifact_dir.iterdir())
    assert len(run_dirs) == 1
    payload = json.loads((run_dirs[0] / "run-report.json").read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert payload["repair_attempted"] is True


def test_cli_max_repair_attempts_zero_produces_truthful_no_repair_failure(
    tmp_path, monkeypatch
) -> None:
    artifact_dir = tmp_path / "generated"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--mock",
            "--mock-scenario",
            "repair-success",
            "--max-repair-attempts",
            "0",
            "--artifact-dir",
            "generated",
        ],
    )

    assert result.exit_code != 0, result.output
    assert "Pipeline failed" in result.output
    assert "max_repair_attempts=0" in result.output
    run_dirs = list(artifact_dir.iterdir())
    assert len(run_dirs) == 1
    payload = json.loads((run_dirs[0] / "run-report.json").read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert payload["repair_attempted"] is False
    assert "max_repair_attempts=0" in (payload["failure_reason"] or "")


def test_cli_invalid_scenario_is_rejected() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(CANONICAL_BRIEF_PATH),
            "--mock",
            "--mock-scenario",
            "not-a-scenario",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid mock scenario" in result.output


def test_cli_repair_success_with_canonical_brief_file(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "generated"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    brief_path = tmp_path / "demo.json"
    brief_path.write_text(
        json.dumps(canonical_url_shortener_brief().model_dump(mode="json")),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "run",
            str(brief_path),
            "--mock",
            "--mock-scenario",
            "repair-success",
            "--artifact-dir",
            "generated",
        ],
    )

    assert result.exit_code == 0, result.output

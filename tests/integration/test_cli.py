"""Integration tests for the CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cognitive_agent_syndicate.cli import app
from cognitive_agent_syndicate.demo import canonical_url_shortener_brief

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_BRIEF_PATH = REPO_ROOT / "examples" / "briefs" / "url_shortener.json"


def test_cli_valid_mock_run_succeeds(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "generated"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

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
    assert "Artifacts written to:" in result.output
    run_dirs = list(artifact_dir.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "run-report.json").exists()


def test_cli_exact_demo_brief_succeeds(tmp_path, monkeypatch) -> None:
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
        ["run", str(brief_path), "--mock", "--artifact-dir", "generated"],
    )

    assert result.exit_code == 0, result.output


def test_cli_same_title_different_description_fails(tmp_path) -> None:
    brief = canonical_url_shortener_brief().model_dump(mode="json")
    brief["description"] = "A different description that should not match the demo."
    brief_path = tmp_path / "near-demo.json"
    brief_path.write_text(json.dumps(brief), encoding="utf-8")

    result = runner.invoke(app, ["run", str(brief_path), "--mock"])

    assert result.exit_code != 0
    assert "URL Shortener demo brief only" in result.output


def test_cli_altered_criterion_fails(tmp_path) -> None:
    brief = canonical_url_shortener_brief().model_dump(mode="json")
    brief["acceptance_criteria"][0]["description"] = "Changed criterion description."
    brief_path = tmp_path / "altered-criterion.json"
    brief_path.write_text(json.dumps(brief), encoding="utf-8")

    result = runner.invoke(app, ["run", str(brief_path), "--mock"])

    assert result.exit_code != 0
    assert "URL Shortener demo brief only" in result.output


def test_cli_unrelated_brief_fails(tmp_path) -> None:
    brief_path = tmp_path / "other.json"
    brief_path.write_text(
        json.dumps(
            {
                "title": "Todo App",
                "description": "Build a todo list.",
                "acceptance_criteria": [
                    {"id": "ac-1", "description": "Create todos.", "must_pass": True}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", str(brief_path), "--mock"])

    assert result.exit_code != 0
    assert "URL Shortener demo brief only" in result.output


def test_cli_invalid_brief_exits_non_zero(tmp_path) -> None:
    bad_brief = tmp_path / "bad.json"
    bad_brief.write_text(json.dumps({"title": "missing fields"}), encoding="utf-8")

    result = runner.invoke(
        app,
        ["run", str(bad_brief), "--mock"],
    )

    assert result.exit_code != 0
    assert "Invalid brief file" in result.output


def test_cli_default_mock_run_succeeds_without_mock_flag(tmp_path, monkeypatch) -> None:
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


def test_module_help_invocation() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cognitive_agent_syndicate", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "contract-driven" in result.stdout.lower() or "Usage" in result.stdout


def test_console_script_help_invocation() -> None:
    console_script = shutil.which("cognitive-agent-syndicate")
    if console_script is None:
        candidate = Path(sys.executable).parent / "cognitive-agent-syndicate"
        if not candidate.exists():
            pytest.skip("console script not installed")
        console_script = str(candidate)

    result = subprocess.run(
        [console_script, "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout or "contract-driven" in result.stdout.lower()


def test_subprocess_invalid_brief_exits_non_zero(tmp_path) -> None:
    bad_brief = tmp_path / "bad.json"
    bad_brief.write_text(json.dumps({"title": "missing fields"}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "cognitive_agent_syndicate", "run", str(bad_brief), "--mock"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode != 0
    assert "Invalid brief file" in result.stdout or "Invalid brief file" in result.stderr

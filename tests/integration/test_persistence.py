"""Integration tests for artifact persistence and reports."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cognitive_agent_syndicate.orchestration.state import PipelineStage, PipelineState
from cognitive_agent_syndicate.prompts import build_architect_system_instructions
from cognitive_agent_syndicate.reporting.artifacts import (
    ArtifactPersistenceError,
    persist_failure_report,
    persist_run_artifacts,
    persist_run_artifacts_legacy,
    validate_generated_path_hierarchy,
)
from cognitive_agent_syndicate.reporting.report_writer import (
    build_success_run_report_snapshot,
    write_run_reports,
)
from cognitive_agent_syndicate.schemas import AttemptOutcome, GeneratedFile, PipelineAttempt
from tests.fixtures.pipeline_fixtures import (
    sample_architecture,
    sample_brief,
    sample_bundle,
    sample_review_approved,
    sample_usage,
)


def _sample_successful_attempt() -> PipelineAttempt:
    return PipelineAttempt(
        attempt_number=1,
        artifacts=sample_bundle(),
        review=sample_review_approved(),
        gates_passed=True,
        reviewer_approved=True,
    )


def _persist_kwargs(
    *,
    artifact_root,
    run_id: str,
    state: PipelineState,
    bundle=None,
):
    attempt = state.attempts[0] if state.attempts else _sample_successful_attempt()
    return {
        "artifact_root": artifact_root,
        "run_id": run_id,
        "brief": sample_brief(),
        "architecture": sample_architecture(),
        "bundle": bundle or sample_bundle(),
        "review": sample_review_approved(),
        "state": state,
        "successful_attempt": attempt,
        "wall_clock_duration_ms": state.wall_clock_duration_ms or 1.0,
    }


def _sample_state(run_id: str = "persist-run", *, success: bool = True) -> PipelineState:
    attempt = _sample_successful_attempt()
    return PipelineState(
        run_id=run_id,
        brief=sample_brief(),
        architecture=sample_architecture(),
        artifacts=sample_bundle(),
        review=sample_review_approved(),
        usage=sample_usage(prompt=1, completion=1),
        stage=PipelineStage.COMPLETED if success else PipelineStage.FAILED,
        success=success,
        stages_completed=[
            PipelineStage.ARCHITECT,
            PipelineStage.IMPLEMENTER,
            PipelineStage.REVIEWER,
            PipelineStage.GATES,
        ],
        attempts=[attempt],
    )


def test_persistence_writes_expected_files(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    state = _sample_state("run-001")
    run_dir, files = persist_run_artifacts(
        **_persist_kwargs(artifact_root=artifact_root, run_id="run-001", state=state)
    )

    assert run_dir == artifact_root / "run-001"
    assert (run_dir / "brief.json").exists()
    assert (run_dir / "architecture.json").exists()
    assert (run_dir / "review.json").exists()
    assert (run_dir / "run-report.json").exists()
    assert (run_dir / "run-report.md").exists()
    assert (run_dir / "artifacts" / "src" / "demo" / "service.py").exists()
    assert len(files) == 4


def test_nested_safe_paths_persist_correctly(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    state = _sample_state("nested-run")
    run_dir, _files = persist_run_artifacts(
        **_persist_kwargs(artifact_root=artifact_root, run_id="nested-run", state=state)
    )

    nested = run_dir / "artifacts" / "src" / "demo" / "service.py"
    assert nested.read_text(encoding="utf-8").startswith("def run")


def test_path_escape_is_rejected(tmp_path) -> None:
    from cognitive_agent_syndicate.reporting.artifacts import resolve_artifact_target

    run_dir = tmp_path / "escape-run"
    run_dir.mkdir()
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    symlink = artifacts_dir / "linked.py"
    symlink.symlink_to(outside)

    with pytest.raises(ArtifactPersistenceError, match="Symlink"):
        resolve_artifact_target(run_dir, "linked.py")


def test_existing_final_directory_is_not_overwritten(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    existing = artifact_root / "existing-run"
    existing.mkdir(parents=True)
    state = _sample_state("existing-run")

    with pytest.raises(ArtifactPersistenceError, match="already exists"):
        persist_run_artifacts(
            **_persist_kwargs(artifact_root=artifact_root, run_id="existing-run", state=state)
        )


def test_path_hierarchy_conflicts_are_rejected() -> None:
    with pytest.raises(ArtifactPersistenceError, match="hierarchy collision"):
        validate_generated_path_hierarchy(["src", "src/demo/service.py"])


def test_forced_failure_leaves_no_partial_final_run(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    state = _sample_state("partial-run")

    with patch(
        "cognitive_agent_syndicate.reporting.artifacts.write_run_reports",
        side_effect=RuntimeError("simulated write failure"),
    ):
        with pytest.raises(RuntimeError, match="simulated write failure"):
            persist_run_artifacts(
                **_persist_kwargs(artifact_root=artifact_root, run_id="partial-run", state=state)
            )

    assert not (artifact_root / "partial-run").exists()
    staging_dirs = list(artifact_root.glob(".staging-*"))
    assert staging_dirs == []


def test_staging_directory_is_cleaned_on_failure(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    state = _sample_state("cleanup-run")

    with patch(
        "cognitive_agent_syndicate.reporting.artifacts._write_run_contents",
        side_effect=ArtifactPersistenceError("simulated content failure"),
    ):
        with pytest.raises(ArtifactPersistenceError, match="simulated content failure"):
            persist_run_artifacts(
                **_persist_kwargs(artifact_root=artifact_root, run_id="cleanup-run", state=state)
            )

    assert list(artifact_root.glob(".staging-*")) == []
    assert not (artifact_root / "cleanup-run").exists()


def test_failure_report_persists_minimal_directory(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    state = _sample_state("failed-run", success=False)
    state.failure_reason = "Reviewer status is rejected, not approved."
    state.stage = PipelineStage.FAILED
    state.stages_completed = [
        PipelineStage.ARCHITECT,
        PipelineStage.IMPLEMENTER,
        PipelineStage.REVIEWER,
        PipelineStage.GATES,
    ]

    run_dir = persist_failure_report(
        artifact_root=artifact_root,
        run_id="failed-run",
        state=state,
    )

    assert (run_dir / "run-report.json").exists()
    assert (run_dir / "run-report.md").exists()
    assert not (run_dir / "artifacts").exists()
    payload = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert payload["failure_reason"]
    assert payload["generated_files"] == []


def test_reports_do_not_contain_secrets_or_full_system_prompts(tmp_path) -> None:
    run_dir = tmp_path / "report-run"
    files = persist_run_artifacts_legacy(
        run_dir=run_dir,
        brief=sample_brief(),
        architecture=sample_architecture(),
        bundle=sample_bundle(),
        review=sample_review_approved(),
    )
    write_run_reports(run_dir=run_dir, state=_sample_state("report-run"), generated_files=files)

    report_json = (run_dir / "run-report.json").read_text(encoding="utf-8")
    report_md = (run_dir / "run-report.md").read_text(encoding="utf-8")
    system_prompt = build_architect_system_instructions()

    assert "api_key" not in report_json.lower()
    assert "sk-" not in report_json
    assert system_prompt not in report_json
    assert system_prompt not in report_md
    payload = json.loads(report_json)
    assert "limitations" in payload
    assert payload["limitations"]


def test_legacy_persistence_refuses_overwrite(tmp_path) -> None:
    run_dir = tmp_path / "existing-run"
    files = persist_run_artifacts_legacy(
        run_dir=run_dir,
        brief=sample_brief(),
        architecture=sample_architecture(),
        bundle=sample_bundle(),
        review=sample_review_approved(),
    )
    state = _sample_state("existing-run")
    write_run_reports(run_dir=run_dir, state=state, generated_files=files)

    with pytest.raises(FileExistsError):
        write_run_reports(run_dir=run_dir, state=state, generated_files=files)


def test_case_insensitive_hierarchy_collision_is_rejected(tmp_path) -> None:
    from cognitive_agent_syndicate.schemas import ArtifactBundle

    artifact_root = tmp_path / "artifacts"
    bundle = ArtifactBundle(
        files=[
            GeneratedFile(path="SRC", content="not a directory\n"),
            GeneratedFile(path="src/service.py", content="def run() -> None:\n    pass\n"),
        ]
    )
    state = _sample_state("case-collision-run")

    with pytest.raises(ArtifactPersistenceError, match="hierarchy collision"):
        persist_run_artifacts(
            **_persist_kwargs(
                artifact_root=artifact_root,
                run_id="case-collision-run",
                state=state,
                bundle=bundle,
            )
        )

    assert not (artifact_root / "case-collision-run").exists()


def test_hierarchy_collision_in_bundle_is_rejected(tmp_path) -> None:
    from cognitive_agent_syndicate.schemas import ArtifactBundle

    artifact_root = tmp_path / "artifacts"
    bundle = ArtifactBundle(
        files=[
            GeneratedFile(path="src", content="not a directory\n"),
            GeneratedFile(path="src/demo/service.py", content="def run() -> None:\n    pass\n"),
        ]
    )
    state = _sample_state("collision-run")

    with pytest.raises(ArtifactPersistenceError, match="hierarchy collision"):
        persist_run_artifacts(
            **_persist_kwargs(
                artifact_root=artifact_root,
                run_id="collision-run",
                state=state,
                bundle=bundle,
            )
        )

    assert not (artifact_root / "collision-run").exists()


def _patch_path_write_text_to_fail_on(filename: str):
    original = Path.write_text

    def selective_write_text(self, data, *args, **kwargs):
        if self.name == filename:
            raise OSError(f"simulated {filename} write failure")
        return original(self, data, *args, **kwargs)

    return patch.object(Path, "write_text", selective_write_text)


def test_json_report_write_failure_leaves_no_success_directory(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    state = _sample_state("json-fail-run")

    with _patch_path_write_text_to_fail_on("run-report.json"):
        with pytest.raises(OSError, match="run-report.json"):
            persist_run_artifacts(
                **_persist_kwargs(artifact_root=artifact_root, run_id="json-fail-run", state=state)
            )

    assert not (artifact_root / "json-fail-run").exists()
    assert list(artifact_root.glob(".staging-*")) == []


def test_markdown_report_write_failure_leaves_no_success_directory(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    state = _sample_state("md-fail-run")

    with _patch_path_write_text_to_fail_on("run-report.md"):
        with pytest.raises(OSError, match="run-report.md"):
            persist_run_artifacts(
                **_persist_kwargs(artifact_root=artifact_root, run_id="md-fail-run", state=state)
            )

    assert not (artifact_root / "md-fail-run").exists()
    assert list(artifact_root.glob(".staging-*")) == []


def test_successful_persist_atomically_contains_reports_and_artifacts(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    state = _sample_state("atomic-run")
    run_dir, files = persist_run_artifacts(
        **_persist_kwargs(artifact_root=artifact_root, run_id="atomic-run", state=state)
    )

    assert (run_dir / "brief.json").exists()
    assert (run_dir / "architecture.json").exists()
    assert (run_dir / "review.json").exists()
    assert (run_dir / "run-report.json").exists()
    assert (run_dir / "run-report.md").exists()
    assert (run_dir / "artifacts" / "pyproject.toml").exists()
    assert len(files) == 4


def test_success_report_manifest_matches_artifacts_directory(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    state = _sample_state("manifest-run")
    run_dir, _files = persist_run_artifacts(
        **_persist_kwargs(artifact_root=artifact_root, run_id="manifest-run", state=state)
    )

    payload = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    assert payload["success"] is True
    artifact_paths = sorted(
        str(path.relative_to(run_dir / "artifacts")).replace("\\", "/")
        for path in (run_dir / "artifacts").rglob("*")
        if path.is_file()
    )
    assert payload["generated_files"] == artifact_paths


def test_success_report_snapshot_marks_final_attempt_success_without_state_mutation() -> None:
    state = _sample_state("snapshot-run")
    state.success = False
    attempt = state.attempts[0]
    attempt.outcome = AttemptOutcome.FAILED

    report = build_success_run_report_snapshot(
        state=state,
        successful_attempt=attempt,
        generated_files=["pyproject.toml"],
        wall_clock_duration_ms=12.5,
    )

    assert state.success is False
    assert attempt.outcome.value == "failed"
    assert report.success is True
    assert report.attempts[0].outcome.value == "success"

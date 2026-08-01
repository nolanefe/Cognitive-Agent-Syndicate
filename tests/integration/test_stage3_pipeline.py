"""Integration tests for Stage 3 repair pipeline behavior."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from cognitive_agent_syndicate.agents.architect import ArchitectAgent
from cognitive_agent_syndicate.agents.implementer import ImplementerAgent
from cognitive_agent_syndicate.agents.reviewer import ReviewerAgent
from cognitive_agent_syndicate.config import build_settings
from cognitive_agent_syndicate.demo import (
    MockScenario,
    canonical_url_shortener_brief,
    create_demo_provider,
)
from cognitive_agent_syndicate.orchestration.pipeline import ContractDrivenPipeline
from cognitive_agent_syndicate.providers.mock import MockModelProvider
from cognitive_agent_syndicate.reporting.artifacts import ArtifactPersistenceError
from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    AttemptOutcome,
    GateRepairability,
    GateResult,
    GateStatus,
    ReviewReport,
    ReviewStatus,
)
from tests.fixtures.pipeline_fixtures import (
    sample_architecture,
    sample_brief,
    sample_bundle,
    sample_review_approved,
    sample_usage,
)


@pytest.fixture
def artifact_workspace(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    yield "artifacts"


def _run_kwargs() -> dict[str, object]:
    return {
        "allowed_technologies": ["python"],
        "permitted_paths": ["pyproject.toml", "src", "tests"],
        "implementation_constraints": ["offline only"],
        "required_project_files": ["pyproject.toml"],
    }


def _build_pipeline(
    provider: MockModelProvider,
    artifact_output_dir: str,
    *,
    run_id_factory=None,
    max_repair_attempts: int = 1,
    monotonic_clock=None,
) -> ContractDrivenPipeline:
    settings = build_settings(
        artifact_output_dir=artifact_output_dir,
        max_repair_attempts=max_repair_attempts,
    )
    return ContractDrivenPipeline(
        architect=ArchitectAgent(provider),
        implementer=ImplementerAgent(provider),
        reviewer=ReviewerAgent(provider),
        settings=settings,
        run_id_factory=run_id_factory,
        monotonic_clock=monotonic_clock,
    )


@pytest.mark.asyncio
async def test_immediate_success_uses_one_attempt(artifact_workspace) -> None:
    provider = create_demo_provider(scenario=MockScenario.SUCCESS)
    pipeline = _build_pipeline(provider, artifact_workspace, run_id_factory=lambda: "one-attempt")

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is True
    assert state.repair_attempted is False
    assert len(state.attempts) == 1
    assert state.attempts[0].attempt_number == 1
    assert state.attempts[0].outcome == AttemptOutcome.SUCCESS


@pytest.mark.asyncio
async def test_repair_success_after_initial_failure(artifact_workspace, tmp_path) -> None:
    provider = create_demo_provider(scenario=MockScenario.REPAIR_SUCCESS)
    pipeline = _build_pipeline(provider, artifact_workspace, run_id_factory=lambda: "repair-ok")

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is True
    assert state.repair_attempted is True
    assert len(state.attempts) == 2
    assert state.attempts[0].outcome == AttemptOutcome.FAILED
    assert state.attempts[1].outcome == AttemptOutcome.SUCCESS
    run_dir = tmp_path / "artifacts" / "repair-ok"
    assert (run_dir / "artifacts" / "pyproject.toml").exists()


@pytest.mark.asyncio
async def test_repair_failure_after_initial_failure(artifact_workspace, tmp_path) -> None:
    provider = create_demo_provider(scenario=MockScenario.REPAIR_FAILURE)
    pipeline = _build_pipeline(provider, artifact_workspace, run_id_factory=lambda: "repair-fail")

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is False
    assert state.repair_attempted is True
    assert len(state.attempts) == 2
    run_dir = tmp_path / "artifacts" / "repair-fail"
    assert run_dir.exists()
    assert not (run_dir / "artifacts").exists()
    payload = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    assert payload["repair_attempted"] is True
    assert payload["attempt_count"] == 2


@pytest.mark.asyncio
async def test_repair_provider_failure_produces_failed_state(artifact_workspace) -> None:
    provider = MockModelProvider(usage=sample_usage())
    provider.configure_response(ArchitectureSpec, sample_architecture())
    provider.configure_response_sequence(ArtifactBundle, [sample_bundle()])
    rejected = sample_review_approved().model_copy(update={"status": ReviewStatus.REJECTED})
    provider.configure_response(ReviewReport, rejected)
    pipeline = _build_pipeline(provider, artifact_workspace)

    state = await pipeline.run(sample_brief(), **_run_kwargs())

    assert state.success is False
    assert state.repair_attempted is True
    assert "Repair implementer failed" in (state.failure_reason or "")


@pytest.mark.asyncio
async def test_reviewer_failure_after_repair(artifact_workspace) -> None:
    provider = MockModelProvider(usage=sample_usage())
    provider.configure_response(ArchitectureSpec, sample_architecture())
    provider.configure_response_sequence(ArtifactBundle, [sample_bundle(), sample_bundle()])
    rejected = sample_review_approved().model_copy(update={"status": ReviewStatus.REJECTED})
    provider.configure_response_sequence(ReviewReport, [rejected, rejected])
    pipeline = _build_pipeline(provider, artifact_workspace)

    state = await pipeline.run(sample_brief(), **_run_kwargs())

    assert state.success is False
    assert state.repair_attempted is True
    assert len(state.attempts) == 2


@pytest.mark.asyncio
async def test_usage_aggregated_across_repair_attempts(artifact_workspace) -> None:
    provider = create_demo_provider(
        scenario=MockScenario.REPAIR_SUCCESS,
        per_stage_usage=sample_usage(prompt=10, completion=5, latency=1.0),
    )
    pipeline = _build_pipeline(provider, artifact_workspace)

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is True
    assert state.usage.prompt_tokens == 50
    assert state.usage.completion_tokens == 25
    assert state.usage.total_tokens == 75
    assert state.usage.latency_ms == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_initial_failed_bundle_not_persisted(artifact_workspace, tmp_path) -> None:
    provider = create_demo_provider(scenario=MockScenario.REPAIR_SUCCESS)
    pipeline = _build_pipeline(provider, artifact_workspace, run_id_factory=lambda: "final-only")

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    run_dir = tmp_path / "artifacts" / "final-only"
    report = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    assert state.success is True
    assert "pyproject.toml" in report["generated_files"]


@pytest.mark.asyncio
async def test_attempt_history_in_report(artifact_workspace, tmp_path) -> None:
    provider = create_demo_provider(scenario=MockScenario.REPAIR_SUCCESS)
    pipeline = _build_pipeline(provider, artifact_workspace, run_id_factory=lambda: "history-run")

    await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    payload = json.loads(
        (tmp_path / "artifacts" / "history-run" / "run-report.json").read_text(encoding="utf-8")
    )
    assert len(payload["attempts"]) == 2
    assert payload["attempts"][0]["outcome"] == "failed"
    assert payload["attempts"][1]["outcome"] == "success"


@pytest.mark.asyncio
async def test_deterministic_clock_records_wall_clock_duration(artifact_workspace) -> None:
    class SteppingClock:
        def __init__(self) -> None:
            self._value = 0.0

        def __call__(self) -> float:
            current = self._value
            self._value += 0.01
            return current

    clock = SteppingClock()
    provider = create_demo_provider(scenario=MockScenario.SUCCESS)
    pipeline = _build_pipeline(provider, artifact_workspace, monotonic_clock=clock)

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.wall_clock_duration_ms == pytest.approx(30.0, rel=0.01)
    assert state.attempts[0].duration_ms == pytest.approx(10.0, rel=0.01)


@pytest.mark.asyncio
async def test_max_repair_attempts_zero_skips_repair(artifact_workspace) -> None:
    provider = create_demo_provider(scenario=MockScenario.REPAIR_SUCCESS)
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "no-repair",
        max_repair_attempts=0,
    )

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is False
    assert state.repair_attempted is False
    assert len(state.attempts) == 1
    assert state.failure_reason is not None
    assert "max_repair_attempts=0" in state.failure_reason


@pytest.mark.asyncio
async def test_immediate_success_persistence_failure_is_truthful(
    artifact_workspace, tmp_path
) -> None:
    provider = create_demo_provider(scenario=MockScenario.SUCCESS)
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "persist-fail-immediate",
    )

    with patch(
        "cognitive_agent_syndicate.reporting.artifacts._write_run_contents",
        side_effect=ArtifactPersistenceError("simulated persistence failure"),
    ):
        state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is False
    assert state.final_artifacts is None
    assert len(state.attempts) == 1
    assert state.attempts[0].outcome != AttemptOutcome.SUCCESS
    assert state.attempts[0].outcome == AttemptOutcome.FAILED
    assert "persistence failed" in (state.attempts[0].failure_reason or "").lower()
    assert "persistence failed" in (state.failure_reason or "").lower()
    run_dir = tmp_path / "artifacts" / "persist-fail-immediate"
    if run_dir.exists():
        assert not (run_dir / "artifacts").exists()
    assert list((tmp_path / "artifacts").glob(".staging-*")) == []


@pytest.mark.asyncio
async def test_repair_success_persistence_failure_is_truthful(artifact_workspace, tmp_path) -> None:
    provider = create_demo_provider(scenario=MockScenario.REPAIR_SUCCESS)
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "persist-fail-repair",
    )

    with patch(
        "cognitive_agent_syndicate.reporting.artifacts._write_run_contents",
        side_effect=ArtifactPersistenceError("simulated persistence failure"),
    ):
        state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is False
    assert state.final_artifacts is None
    assert len(state.attempts) == 2
    assert state.attempts[1].outcome == AttemptOutcome.FAILED
    assert state.attempts[1].outcome != AttemptOutcome.SUCCESS
    run_dir = tmp_path / "artifacts" / "persist-fail-repair"
    if run_dir.exists():
        assert not (run_dir / "artifacts").exists()
    assert list((tmp_path / "artifacts").glob(".staging-*")) == []


@pytest.mark.asyncio
async def test_initial_failed_bundle_differs_from_repaired_final_bundle(
    artifact_workspace,
    tmp_path,
) -> None:
    provider = create_demo_provider(scenario=MockScenario.REPAIR_SUCCESS)
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "bundle-diff",
    )

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is True
    assert state.attempts[0].artifacts is not None
    assert state.attempts[1].artifacts is not None
    initial_paths = {file.path for file in state.attempts[0].artifacts.files}
    final_paths = {file.path for file in state.attempts[1].artifacts.files}
    assert initial_paths != final_paths
    assert "pyproject.toml" not in initial_paths
    assert "pyproject.toml" in final_paths
    run_dir = tmp_path / "artifacts" / "bundle-diff"
    assert (run_dir / "artifacts" / "pyproject.toml").exists()
    report = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    assert "pyproject.toml" in report["generated_files"]
    assert "pyproject.toml" not in {file.path for file in state.attempts[0].artifacts.files}


@pytest.mark.asyncio
async def test_usage_reconciles_across_architect_and_attempts(artifact_workspace) -> None:
    per_stage = sample_usage(prompt=10, completion=5, latency=1.0)
    provider = create_demo_provider(
        scenario=MockScenario.REPAIR_SUCCESS,
        per_stage_usage=per_stage,
    )
    pipeline = _build_pipeline(provider, artifact_workspace)

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    attempt_prompt = state.attempts[0].usage.prompt_tokens + state.attempts[1].usage.prompt_tokens
    attempt_completion = (
        state.attempts[0].usage.completion_tokens + state.attempts[1].usage.completion_tokens
    )
    attempt_latency = state.attempts[0].usage.latency_ms + state.attempts[1].usage.latency_ms
    assert state.architect_usage.prompt_tokens + attempt_prompt == state.usage.prompt_tokens
    assert (
        state.architect_usage.completion_tokens + attempt_completion
        == state.usage.completion_tokens
    )
    assert state.architect_usage.latency_ms + attempt_latency == pytest.approx(
        state.usage.latency_ms
    )


@pytest.mark.asyncio
async def test_mixed_non_repairable_gate_failure_prevents_repair(artifact_workspace) -> None:
    provider = MockModelProvider(usage=sample_usage())
    provider.configure_response(ArchitectureSpec, sample_architecture())
    provider.configure_response(ArtifactBundle, sample_bundle())
    rejected = sample_review_approved().model_copy(update={"status": ReviewStatus.REJECTED})
    provider.configure_response(ReviewReport, rejected)

    from cognitive_agent_syndicate.validation.gates import GateRunner

    original_run = GateRunner.run

    def mixed_gate_run(runner_self, **kwargs):
        results = original_run(runner_self, **kwargs)
        return [
            *results,
            GateResult(
                gate_id="paths_comply_with_permitted_prefixes",
                gate_name="paths_comply_with_permitted_prefixes",
                status=GateStatus.FAILED,
                message="Path outside permitted prefixes.",
                duration_ms=1.0,
                repairable=GateRepairability.NON_REPAIRABLE,
            ),
            GateResult(
                gate_id="python_syntax",
                gate_name="python_syntax",
                status=GateStatus.FAILED,
                message="Syntax issue.",
                duration_ms=1.0,
                repairable=GateRepairability.REPAIRABLE,
            ),
        ]

    pipeline = _build_pipeline(provider, artifact_workspace)
    with patch.object(GateRunner, "run", mixed_gate_run):
        state = await pipeline.run(sample_brief(), **_run_kwargs())

    assert state.success is False
    assert state.repair_attempted is False
    assert state.failure_reason is not None
    assert "non-repairable" in state.failure_reason.lower()


@pytest.mark.asyncio
async def test_regressing_clock_produces_non_negative_timings(artifact_workspace, tmp_path) -> None:
    class RegressingClock:
        def __init__(self) -> None:
            self._tick = 0

        def __call__(self) -> float:
            value = max(0.0, 1.0 - self._tick * 0.2)
            self._tick += 1
            return value

    provider = create_demo_provider(scenario=MockScenario.SUCCESS)
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "regressing-clock",
        monotonic_clock=RegressingClock(),
    )

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.wall_clock_duration_ms >= 0.0
    assert state.attempts[0].ended_at_ms >= state.attempts[0].started_at_ms
    assert state.attempts[0].duration_ms >= 0.0
    payload = json.loads(
        (tmp_path / "artifacts" / "regressing-clock" / "run-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["wall_clock_duration_ms"] >= 0.0
    assert payload["attempts"][0]["duration_ms"] >= 0.0


def _patch_path_write_text_to_fail_on(filename: str):
    original = Path.write_text

    def selective_write_text(self, data, *args, **kwargs):
        if self.name == filename:
            raise OSError(f"simulated {filename} write failure")
        return original(self, data, *args, **kwargs)

    return patch.object(Path, "write_text", selective_write_text)


@pytest.mark.asyncio
async def test_pipeline_json_report_write_failure_is_truthful(artifact_workspace, tmp_path) -> None:
    provider = create_demo_provider(scenario=MockScenario.SUCCESS)
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "json-report-fail",
    )

    with _patch_path_write_text_to_fail_on("run-report.json"):
        state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is False
    assert state.final_artifacts is None
    assert state.attempts[0].outcome != AttemptOutcome.SUCCESS
    assert not (tmp_path / "artifacts" / "json-report-fail").exists()
    assert list((tmp_path / "artifacts").glob(".staging-*")) == []


@pytest.mark.asyncio
async def test_pipeline_markdown_report_write_failure_is_truthful(
    artifact_workspace, tmp_path
) -> None:
    provider = create_demo_provider(scenario=MockScenario.SUCCESS)
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "md-report-fail",
    )

    with _patch_path_write_text_to_fail_on("run-report.md"):
        state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is False
    assert state.final_artifacts is None
    assert state.attempts[0].outcome != AttemptOutcome.SUCCESS
    assert not (tmp_path / "artifacts" / "md-report-fail").exists()
    assert list((tmp_path / "artifacts").glob(".staging-*")) == []


@pytest.mark.asyncio
async def test_immediate_success_run_is_atomic_with_reports_and_artifacts(
    artifact_workspace, tmp_path
) -> None:
    provider = create_demo_provider(scenario=MockScenario.SUCCESS)
    pipeline = _build_pipeline(provider, artifact_workspace, run_id_factory=lambda: "atomic-ok")

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    run_dir = tmp_path / "artifacts" / "atomic-ok"
    assert state.success is True
    assert (run_dir / "run-report.json").exists()
    assert (run_dir / "run-report.md").exists()
    assert (run_dir / "artifacts" / "pyproject.toml").exists()
    payload = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert payload["attempts"][-1]["outcome"] == "success"


@pytest.mark.asyncio
async def test_repair_success_persists_only_final_bundle_atomically(
    artifact_workspace, tmp_path
) -> None:
    provider = create_demo_provider(scenario=MockScenario.REPAIR_SUCCESS)
    pipeline = _build_pipeline(provider, artifact_workspace, run_id_factory=lambda: "repair-atomic")

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    run_dir = tmp_path / "artifacts" / "repair-atomic"
    assert state.success is True
    assert (run_dir / "run-report.json").exists()
    assert (run_dir / "run-report.md").exists()
    assert (run_dir / "artifacts" / "pyproject.toml").exists()
    payload = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    artifact_paths = sorted(
        str(path.relative_to(run_dir / "artifacts")).replace("\\", "/")
        for path in (run_dir / "artifacts").rglob("*")
        if path.is_file()
    )
    assert payload["generated_files"] == artifact_paths
    assert payload["success"] is True
    assert payload["attempts"][0]["outcome"] == "failed"
    assert payload["attempts"][1]["outcome"] == "success"


@pytest.mark.asyncio
async def test_pipeline_run_directory_collision_is_rejected(artifact_workspace, tmp_path) -> None:
    provider = create_demo_provider(scenario=MockScenario.SUCCESS)
    pipeline = _build_pipeline(provider, artifact_workspace, run_id_factory=lambda: "collision-run")
    existing = tmp_path / "artifacts" / "collision-run"
    existing.mkdir(parents=True)

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is False
    assert state.final_artifacts is None
    assert state.attempts[0].outcome == AttemptOutcome.FAILED
    assert "persistence failed" in (state.failure_reason or "").lower()
    assert not (existing / "run-report.json").exists()

"""Integration tests for the contract-driven pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from cognitive_agent_syndicate.agents.architect import ArchitectAgent
from cognitive_agent_syndicate.agents.implementer import ImplementerAgent
from cognitive_agent_syndicate.agents.reviewer import ReviewerAgent
from cognitive_agent_syndicate.config import build_settings
from cognitive_agent_syndicate.demo import canonical_url_shortener_brief, create_demo_provider
from cognitive_agent_syndicate.orchestration.pipeline import ContractDrivenPipeline
from cognitive_agent_syndicate.orchestration.state import PipelineStage
from cognitive_agent_syndicate.providers.mock import MockModelProvider
from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
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


def _build_pipeline(
    provider: MockModelProvider,
    artifact_output_dir: str,
    *,
    run_id_factory=None,
    max_repair_attempts: int = 0,
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
    )


def _run_kwargs() -> dict[str, object]:
    return {
        "allowed_technologies": ["python"],
        "permitted_paths": ["pyproject.toml", "src", "tests"],
        "implementation_constraints": ["offline only"],
        "required_project_files": ["pyproject.toml"],
    }


@pytest.mark.asyncio
async def test_successful_full_mock_run(artifact_workspace) -> None:
    provider = create_demo_provider()
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "demo-run-001",
    )

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is True
    assert state.stage == PipelineStage.COMPLETED
    assert state.run_id == "demo-run-001"
    assert state.architecture is not None
    assert state.artifacts is not None
    assert state.review is not None
    assert state.review.status == ReviewStatus.APPROVED
    assert state.usage.total_tokens == 450
    assert state.usage.prompt_tokens + state.usage.completion_tokens == state.usage.total_tokens
    assert state.artifact_directory is not None


@pytest.mark.asyncio
async def test_architect_failure_stops_pipeline(artifact_workspace, tmp_path) -> None:
    provider = MockModelProvider()
    pipeline = _build_pipeline(provider, artifact_workspace, run_id_factory=lambda: "arch-fail")

    state = await pipeline.run(sample_brief(), **_run_kwargs())

    assert state.success is False
    assert state.stage == PipelineStage.FAILED
    assert state.architecture is None
    assert state.artifacts is None
    assert state.review is None
    assert "No mock response configured" in (state.failure_reason or "")
    assert state.artifact_directory is not None
    run_dir = Path(state.artifact_directory)
    payload = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert payload["failure_reason"]
    assert payload["stages_completed"] == []


@pytest.mark.asyncio
async def test_implementer_failure_stops_before_reviewer(artifact_workspace) -> None:
    provider = MockModelProvider()
    provider.configure_response(ArchitectureSpec, sample_architecture())
    pipeline = _build_pipeline(provider, artifact_workspace)

    state = await pipeline.run(sample_brief(), **_run_kwargs())

    assert state.success is False
    assert state.architecture is not None
    assert state.artifacts is None
    assert state.review is None
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_reviewer_failure_produces_failed_state(artifact_workspace) -> None:
    provider = MockModelProvider()
    provider.configure_response(ArchitectureSpec, sample_architecture())
    provider.configure_response(ArtifactBundle, sample_bundle())
    pipeline = _build_pipeline(provider, artifact_workspace)

    state = await pipeline.run(sample_brief(), **_run_kwargs())

    assert state.success is False
    assert state.review is None
    assert state.artifacts is not None
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_usage_is_aggregated_correctly(artifact_workspace) -> None:
    provider = MockModelProvider(usage=sample_usage(prompt=10, completion=5, latency=1.0))
    provider.configure_response(ArchitectureSpec, sample_architecture())
    provider.configure_response(ArtifactBundle, sample_bundle())
    provider.configure_response(ReviewReport, sample_review_approved())
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "usage-run",
    )

    state = await pipeline.run(sample_brief(), **_run_kwargs())

    assert state.usage.prompt_tokens == 30
    assert state.usage.completion_tokens == 15
    assert state.usage.total_tokens == 45
    assert state.usage.latency_ms == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_deterministic_run_id_support(artifact_workspace, tmp_path) -> None:
    provider = create_demo_provider()
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "fixed-run-id",
    )

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.run_id == "fixed-run-id"
    run_dir = tmp_path / "artifacts" / "fixed-run-id"
    assert run_dir.exists()


@pytest.mark.asyncio
async def test_gate_failure_persists_failure_report(artifact_workspace, tmp_path) -> None:
    provider = MockModelProvider(usage=sample_usage())
    provider.configure_response(ArchitectureSpec, sample_architecture())
    provider.configure_response(ArtifactBundle, sample_bundle())
    rejected = sample_review_approved()
    rejected = rejected.model_copy(update={"status": ReviewStatus.REJECTED})
    provider.configure_response(ReviewReport, rejected)
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "gate-fail",
    )

    state = await pipeline.run(sample_brief(), **_run_kwargs())

    assert state.success is False
    assert state.artifact_directory is not None
    run_dir = tmp_path / "artifacts" / "gate-fail"
    assert run_dir.exists()
    assert not (run_dir / "artifacts").exists()
    payload = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert "not approved" in payload["failure_reason"]
    assert "reviewer" in payload["stages_completed"]
    assert "gates" in payload["stages_completed"]


@pytest.mark.asyncio
async def test_agent_failure_persists_useful_failure_report(artifact_workspace, tmp_path) -> None:
    provider = MockModelProvider()
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "agent-fail",
    )

    state = await pipeline.run(sample_brief(), **_run_kwargs())

    assert state.success is False
    run_dir = tmp_path / "artifacts" / "agent-fail"
    assert run_dir.exists()
    payload = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert payload["failure_reason"]
    assert "Traceback" not in payload["failure_reason"]


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="Symlinks not supported")
async def test_symlinked_artifact_root_is_rejected(artifact_workspace, tmp_path) -> None:
    real_root = tmp_path / "real-artifacts"
    real_root.mkdir()
    symlink_root = tmp_path / "linked-artifacts"
    symlink_root.symlink_to(real_root)

    provider = create_demo_provider()
    pipeline = _build_pipeline(
        provider,
        str(symlink_root.relative_to(tmp_path)),
        run_id_factory=lambda: "symlink-run",
    )

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is False
    assert state.failure_reason is not None
    assert "symlink" in state.failure_reason.lower()
    assert state.artifact_directory is None
    assert state.stages_completed == []


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="Symlinks not supported")
async def test_symlinked_intermediate_parent_is_rejected(artifact_workspace, tmp_path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent)
    artifact_root = linked_parent / "artifacts"

    provider = create_demo_provider()
    pipeline = _build_pipeline(
        provider,
        str(artifact_root.relative_to(tmp_path)),
        run_id_factory=lambda: "symlink-parent-run",
    )

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is False
    assert state.failure_reason is not None
    assert "symlink" in state.failure_reason.lower()
    assert state.artifact_directory is None
    assert state.stages_completed == []


@pytest.mark.asyncio
async def test_pipeline_does_not_execute_generated_code(artifact_workspace, monkeypatch) -> None:
    import builtins
    import importlib
    import subprocess
    import traceback

    blocked: list[str] = []
    original_import = builtins.__import__
    original_import_module = importlib.import_module
    original_subprocess_run = subprocess.run
    generated_roots = {"url_shortener"}

    def _invoked_from_pipeline() -> bool:
        return any(
            "cognitive_agent_syndicate" in frame.filename and "/tests/" not in frame.filename
            for frame in traceback.extract_stack()
        )

    def fake_exec(_code, _globals=None, _locals=None):  # type: ignore[no-untyped-def]
        if _invoked_from_pipeline():
            blocked.append("exec")
            raise AssertionError("Generated code must not be executed")

    def fake_eval(_code, _globals=None, _locals=None):  # type: ignore[no-untyped-def]
        if _invoked_from_pipeline():
            blocked.append("eval")
            raise AssertionError("Generated code must not be evaluated")

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        if _invoked_from_pipeline():
            blocked.append("subprocess.run")
            raise AssertionError("Generated code must not be executed via subprocess")
        return original_subprocess_run(*args, **kwargs)

    def fake_import_module(name, package=None):  # type: ignore[no-untyped-def]
        root = name.partition(".")[0]
        if root in generated_roots:
            blocked.append("importlib.import_module")
            raise AssertionError("Generated modules must not be imported")
        return original_import_module(name, package)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        root = name.partition(".")[0]
        if root in generated_roots:
            blocked.append(f"__import__:{name}")
            raise AssertionError("Generated modules must not be imported")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.exec", fake_exec)
    monkeypatch.setattr("builtins.eval", fake_eval)
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("importlib.import_module", fake_import_module)
    monkeypatch.setattr("builtins.__import__", fake_import)

    provider = create_demo_provider()
    pipeline = _build_pipeline(
        provider,
        artifact_workspace,
        run_id_factory=lambda: "no-exec",
    )

    state = await pipeline.run(canonical_url_shortener_brief(), **_run_kwargs())

    assert state.success is True
    assert blocked == []

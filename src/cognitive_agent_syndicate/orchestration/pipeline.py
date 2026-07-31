"""Contract-driven multi-agent pipeline orchestration."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

from cognitive_agent_syndicate.agents.architect import ArchitectAgent
from cognitive_agent_syndicate.agents.implementer import ImplementerAgent
from cognitive_agent_syndicate.agents.reviewer import ReviewerAgent
from cognitive_agent_syndicate.config import Settings
from cognitive_agent_syndicate.orchestration.state import PipelineStage, PipelineState
from cognitive_agent_syndicate.paths import SymlinkArtifactRootError, reject_symlink_artifact_root
from cognitive_agent_syndicate.reporting.artifacts import (
    ArtifactPersistenceError,
    persist_failure_report,
    persist_run_artifacts,
)
from cognitive_agent_syndicate.schemas import ReviewStatus, SystemBrief, UsageMetrics
from cognitive_agent_syndicate.validation.gates import GateRunner

RunIdFactory = Callable[[], str]


class ContractDrivenPipeline:
    """Offline contract-driven architect → implementer → reviewer pipeline."""

    def __init__(
        self,
        *,
        architect: ArchitectAgent,
        implementer: ImplementerAgent,
        reviewer: ReviewerAgent,
        settings: Settings,
        gate_runner: GateRunner | None = None,
        run_id_factory: RunIdFactory | None = None,
    ) -> None:
        self._architect = architect
        self._implementer = implementer
        self._reviewer = reviewer
        self._settings = settings
        self._gate_runner = gate_runner or GateRunner()
        self._run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)

    async def run(
        self,
        brief: SystemBrief,
        *,
        allowed_technologies: list[str],
        permitted_paths: list[str],
        implementation_constraints: list[str],
        required_project_files: list[str] | None = None,
    ) -> PipelineState:
        run_id = self._run_id_factory()
        state = PipelineState(run_id=run_id, brief=brief, stage=PipelineStage.INIT)
        required_files = required_project_files or []
        artifact_root = Path(self._settings.artifact_output_dir)

        try:
            reject_symlink_artifact_root(artifact_root)
        except SymlinkArtifactRootError as exc:
            state.stage = PipelineStage.FAILED
            state.success = False
            state.failure_reason = _sanitize_failure_reason(exc)
            return state

        try:
            state.stage = PipelineStage.ARCHITECT
            architect_result = await self._architect.run(brief)
            state.architecture = architect_result.response
            state.usage = architect_result.usage
            state.stages_completed.append(PipelineStage.ARCHITECT)

            state.stage = PipelineStage.IMPLEMENTER
            implementer_result = await self._implementer.run(
                brief=brief,
                architecture=state.architecture,
                allowed_technologies=allowed_technologies,
                permitted_paths=permitted_paths,
                implementation_constraints=implementation_constraints,
            )
            state.artifacts = implementer_result.response
            state.usage = _aggregate_usage(state.usage, implementer_result.usage)
            state.stages_completed.append(PipelineStage.IMPLEMENTER)

            state.stage = PipelineStage.REVIEWER
            reviewer_result = await self._reviewer.run(
                brief=brief,
                architecture=state.architecture,
                bundle=state.artifacts,
            )
            state.review = reviewer_result.response
            state.usage = _aggregate_usage(state.usage, reviewer_result.usage)
            state.stages_completed.append(PipelineStage.REVIEWER)

            state.stage = PipelineStage.GATES
            assert state.architecture is not None
            assert state.artifacts is not None
            assert state.review is not None

            gate_results = self._gate_runner.run(
                brief=brief,
                architecture=state.architecture,
                bundle=state.artifacts,
                review=state.review,
                settings=self._settings,
                permitted_paths=permitted_paths,
                required_project_files=required_files,
            )
            state.gate_results = gate_results
            state.stages_completed.append(PipelineStage.GATES)

            gates_passed = GateRunner.all_required_passed(gate_results)
            reviewer_approved = state.review.status == ReviewStatus.APPROVED
            if not gates_passed or not reviewer_approved:
                reasons: list[str] = []
                if not gates_passed:
                    reasons.append("One or more deterministic gates failed.")
                if not reviewer_approved:
                    reasons.append(f"Reviewer status is {state.review.status.value}, not approved.")
                state.failure_reason = " ".join(reasons)
                state.stage = PipelineStage.FAILED
                state.success = False
                report_dir = _persist_failure_report_safely(
                    artifact_root=artifact_root,
                    run_id=run_id,
                    state=state,
                )
                state.artifact_directory = str(report_dir) if report_dir is not None else None
                return state

            state.stage = PipelineStage.PERSISTENCE
            run_dir, generated_files = persist_run_artifacts(
                artifact_root=artifact_root,
                run_id=run_id,
                brief=brief,
                architecture=state.architecture,
                bundle=state.artifacts,
                review=state.review,
                state=state,
            )
            state.artifact_directory = str(run_dir)
            state.stages_completed.append(PipelineStage.PERSISTENCE)
            state.stage = PipelineStage.COMPLETED
            state.success = True
            return state
        except Exception as exc:
            state.stage = PipelineStage.FAILED
            state.success = False
            state.failure_reason = _sanitize_failure_reason(exc)
            report_dir = _persist_failure_report_safely(
                artifact_root=artifact_root,
                run_id=run_id,
                state=state,
            )
            state.artifact_directory = str(report_dir) if report_dir is not None else None
            return state


def _aggregate_usage(existing: UsageMetrics, latest: UsageMetrics) -> UsageMetrics:
    prompt_tokens = existing.prompt_tokens + latest.prompt_tokens
    completion_tokens = existing.completion_tokens + latest.completion_tokens
    total_tokens = prompt_tokens + completion_tokens
    latency_ms = existing.latency_ms + latest.latency_ms
    return UsageMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
    )


def _sanitize_failure_reason(exc: BaseException) -> str:
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return message.splitlines()[0][:2000]


def _persist_failure_report_safely(
    *,
    artifact_root: Path,
    run_id: str,
    state: PipelineState,
) -> Path | None:
    try:
        return persist_failure_report(
            artifact_root=artifact_root,
            run_id=run_id,
            state=state,
        )
    except (ArtifactPersistenceError, OSError, SymlinkArtifactRootError):
        return None

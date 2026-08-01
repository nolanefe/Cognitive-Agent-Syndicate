"""Contract-driven multi-agent pipeline orchestration."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from cognitive_agent_syndicate.agents.architect import ArchitectAgent
from cognitive_agent_syndicate.agents.implementer import ImplementerAgent
from cognitive_agent_syndicate.agents.reviewer import ReviewerAgent
from cognitive_agent_syndicate.config import Settings
from cognitive_agent_syndicate.orchestration.clock import MonotonicClock, default_monotonic_clock
from cognitive_agent_syndicate.orchestration.failures import (
    PipelineFailureCategory,
    categorize_pipeline_exception,
    infer_evaluable_failure_category,
)
from cognitive_agent_syndicate.orchestration.state import PipelineStage, PipelineState
from cognitive_agent_syndicate.paths import SymlinkArtifactRootError, reject_symlink_artifact_root
from cognitive_agent_syndicate.reporting.artifacts import (
    ArtifactPersistenceError,
    persist_failure_report,
    persist_run_artifacts,
)
from cognitive_agent_syndicate.schemas import (
    AttemptOutcome,
    PipelineAttempt,
    ReviewStatus,
    SystemBrief,
    UsageMetrics,
)
from cognitive_agent_syndicate.validation.gates import GateRunner
from cognitive_agent_syndicate.validation.repair import build_repair_request
from cognitive_agent_syndicate.validation.repair_eligibility import (
    RepairIneligibilityReason,
    evaluate_repair_eligibility,
)

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
        monotonic_clock: MonotonicClock | None = None,
    ) -> None:
        self._architect = architect
        self._implementer = implementer
        self._reviewer = reviewer
        self._settings = settings
        self._gate_runner = gate_runner or GateRunner()
        self._run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)
        self._clock = monotonic_clock or default_monotonic_clock

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
        pipeline_start = self._clock()
        state = PipelineState(
            run_id=run_id,
            brief=brief,
            stage=PipelineStage.INIT,
            pipeline_started_at_ms=pipeline_start * 1000.0,
        )
        required_files = required_project_files or []
        artifact_root = Path(self._settings.artifact_output_dir)
        provider_failure = False

        try:
            reject_symlink_artifact_root(artifact_root)
        except SymlinkArtifactRootError as exc:
            return _finalize_failed_state(
                state=state,
                clock=self._clock,
                pipeline_start=pipeline_start,
                failure_reason=_sanitize_failure_reason(exc),
                failure_category=PipelineFailureCategory.PERSISTENCE_FAILED,
                artifact_root=artifact_root,
                run_id=run_id,
            )

        try:
            state.stage = PipelineStage.ARCHITECT
            architect_result = await self._architect.run(brief)
            state.architecture = architect_result.response
            state.architect_usage = architect_result.usage
            state.usage = architect_result.usage
            state.stages_completed.append(PipelineStage.ARCHITECT)

            attempt1 = _start_attempt(
                attempt_number=1,
                clock=self._clock,
                pipeline_start=pipeline_start,
            )
            state.stage = PipelineStage.IMPLEMENTER
            implementer_result = await self._implementer.run(
                brief=brief,
                architecture=state.architecture,
                allowed_technologies=allowed_technologies,
                permitted_paths=permitted_paths,
                implementation_constraints=implementation_constraints,
            )
            attempt1.artifacts = implementer_result.response
            state.artifacts = attempt1.artifacts
            attempt1.usage = _aggregate_usage(attempt1.usage, implementer_result.usage)
            state.usage = _aggregate_usage(state.usage, implementer_result.usage)
            state.stages_completed.append(PipelineStage.IMPLEMENTER)

            state.stage = PipelineStage.REVIEWER
            reviewer_result = await self._reviewer.run(
                brief=brief,
                architecture=state.architecture,
                bundle=attempt1.artifacts,
            )
            attempt1.review = reviewer_result.response
            state.review = attempt1.review
            attempt1.usage = _aggregate_usage(attempt1.usage, reviewer_result.usage)
            state.usage = _aggregate_usage(state.usage, reviewer_result.usage)
            state.stages_completed.append(PipelineStage.REVIEWER)

            state.stage = PipelineStage.GATES
            assert state.architecture is not None
            assert attempt1.artifacts is not None
            assert attempt1.review is not None

            attempt1.gate_results = self._gate_runner.run(
                brief=brief,
                architecture=state.architecture,
                bundle=attempt1.artifacts,
                review=attempt1.review,
                settings=self._settings,
                permitted_paths=permitted_paths,
                required_project_files=required_files,
            )
            state.gate_results = attempt1.gate_results
            state.stages_completed.append(PipelineStage.GATES)

            attempt1.gates_passed = GateRunner.all_required_passed(attempt1.gate_results)
            attempt1.reviewer_approved = attempt1.review.status == ReviewStatus.APPROVED
            _finalize_attempt(attempt1, clock=self._clock, pipeline_start=pipeline_start)

            if attempt1.gates_passed and attempt1.reviewer_approved:
                state.attempts.append(attempt1)
                return _persist_success(
                    state=state,
                    attempt=attempt1,
                    artifact_root=artifact_root,
                    run_id=run_id,
                    brief=brief,
                    clock=self._clock,
                    pipeline_start=pipeline_start,
                )

            attempt1.outcome = AttemptOutcome.FAILED
            attempt1.failure_reason = _attempt_failure_reason(attempt1)
            state.attempts.append(attempt1)

            eligibility = evaluate_repair_eligibility(
                settings=self._settings,
                stage=state.stage,
                artifacts=attempt1.artifacts,
                review=attempt1.review,
                gate_results=attempt1.gate_results,
                repair_attempted=state.repair_attempted,
                provider_failure=provider_failure,
            )

            if not eligibility.eligible:
                state.failure_reason = _repair_ineligibility_failure_reason(
                    attempt=attempt1,
                    eligibility_reason=eligibility.reason,
                    trigger=eligibility.trigger,
                )
                return _finalize_failed_state(
                    state=state,
                    clock=self._clock,
                    pipeline_start=pipeline_start,
                    failure_reason=state.failure_reason or "Initial attempt failed.",
                    failure_category=_category_for_attempt(attempt1),
                    artifact_root=artifact_root,
                    run_id=run_id,
                )

            state.repair_trigger = eligibility.trigger
            state.repair_attempted = True
            state.stage = PipelineStage.REPAIR

            repair_request = build_repair_request(
                brief=brief,
                architecture=state.architecture,
                current_bundle=attempt1.artifacts,
                gate_results=attempt1.gate_results,
                review=attempt1.review,
                allowed_technologies=allowed_technologies,
                permitted_paths=permitted_paths,
                implementation_constraints=implementation_constraints,
            )

            attempt2 = _start_attempt(
                attempt_number=2,
                clock=self._clock,
                pipeline_start=pipeline_start,
            )
            try:
                repair_result = await self._implementer.repair(repair_request)
                attempt2.artifacts = repair_result.response
                attempt2.usage = _aggregate_usage(attempt2.usage, repair_result.usage)
                state.usage = _aggregate_usage(state.usage, repair_result.usage)
            except Exception as exc:
                provider_failure = True
                attempt2.failure_reason = _sanitize_failure_reason(exc)
                attempt2.outcome = AttemptOutcome.FAILED
                _finalize_attempt(attempt2, clock=self._clock, pipeline_start=pipeline_start)
                state.attempts.append(attempt2)
                return _finalize_failed_state(
                    state=state,
                    clock=self._clock,
                    pipeline_start=pipeline_start,
                    failure_reason=f"Repair implementer failed: {attempt2.failure_reason}",
                    failure_category=categorize_pipeline_exception(exc),
                    artifact_root=artifact_root,
                    run_id=run_id,
                )

            state.stage = PipelineStage.REVIEWER
            try:
                reviewer_result2 = await self._reviewer.run(
                    brief=brief,
                    architecture=state.architecture,
                    bundle=attempt2.artifacts,
                )
                attempt2.review = reviewer_result2.response
                attempt2.usage = _aggregate_usage(attempt2.usage, reviewer_result2.usage)
                state.usage = _aggregate_usage(state.usage, reviewer_result2.usage)
            except Exception as exc:
                attempt2.failure_reason = _sanitize_failure_reason(exc)
                attempt2.outcome = AttemptOutcome.FAILED
                _finalize_attempt(attempt2, clock=self._clock, pipeline_start=pipeline_start)
                state.attempts.append(attempt2)
                return _finalize_failed_state(
                    state=state,
                    clock=self._clock,
                    pipeline_start=pipeline_start,
                    failure_reason=f"Repair reviewer failed: {attempt2.failure_reason}",
                    failure_category=categorize_pipeline_exception(exc),
                    artifact_root=artifact_root,
                    run_id=run_id,
                )

            state.stage = PipelineStage.GATES
            assert attempt2.artifacts is not None
            assert attempt2.review is not None
            attempt2.gate_results = self._gate_runner.run(
                brief=brief,
                architecture=state.architecture,
                bundle=attempt2.artifacts,
                review=attempt2.review,
                settings=self._settings,
                permitted_paths=permitted_paths,
                required_project_files=required_files,
            )
            state.gate_results = attempt2.gate_results
            state.review = attempt2.review
            state.artifacts = attempt2.artifacts

            attempt2.gates_passed = GateRunner.all_required_passed(attempt2.gate_results)
            attempt2.reviewer_approved = attempt2.review.status == ReviewStatus.APPROVED
            _finalize_attempt(attempt2, clock=self._clock, pipeline_start=pipeline_start)

            if attempt2.gates_passed and attempt2.reviewer_approved:
                state.attempts.append(attempt2)
                return _persist_success(
                    state=state,
                    attempt=attempt2,
                    artifact_root=artifact_root,
                    run_id=run_id,
                    brief=brief,
                    clock=self._clock,
                    pipeline_start=pipeline_start,
                )

            attempt2.outcome = AttemptOutcome.FAILED
            attempt2.failure_reason = _attempt_failure_reason(attempt2)
            state.attempts.append(attempt2)
            return _finalize_failed_state(
                state=state,
                clock=self._clock,
                pipeline_start=pipeline_start,
                failure_reason=attempt2.failure_reason or "Repair attempt failed.",
                failure_category=_category_for_attempt(attempt2),
                artifact_root=artifact_root,
                run_id=run_id,
            )

        except Exception as exc:
            return _finalize_failed_state(
                state=state,
                clock=self._clock,
                pipeline_start=pipeline_start,
                failure_reason=_sanitize_failure_reason(exc),
                failure_category=categorize_pipeline_exception(exc),
                artifact_root=artifact_root,
                run_id=run_id,
            )


def _start_attempt(
    *,
    attempt_number: int,
    clock: MonotonicClock,
    pipeline_start: float,
) -> PipelineAttempt:
    now_ms = max(0.0, (clock() - pipeline_start) * 1000.0)
    return PipelineAttempt(
        attempt_number=attempt_number,
        started_at_ms=now_ms,
    )


def _finalize_attempt(
    attempt: PipelineAttempt,
    *,
    clock: MonotonicClock,
    pipeline_start: float,
) -> None:
    ended_at_ms = (clock() - pipeline_start) * 1000.0
    attempt.ended_at_ms = max(attempt.started_at_ms, ended_at_ms)
    attempt.duration_ms = max(0.0, attempt.ended_at_ms - attempt.started_at_ms)
    attempt.reviewer_status = attempt.review.status if attempt.review is not None else None


def _attempt_failure_reason(attempt: PipelineAttempt) -> str:
    reasons: list[str] = []
    if not attempt.gates_passed:
        reasons.append("One or more deterministic gates failed.")
    if not attempt.reviewer_approved:
        status = attempt.review.status.value if attempt.review is not None else "unknown"
        reasons.append(f"Reviewer status is {status}, not approved.")
    return " ".join(reasons)


def _repair_ineligibility_failure_reason(
    *,
    attempt: PipelineAttempt,
    eligibility_reason: RepairIneligibilityReason | None,
    trigger: str | None,
) -> str:
    base_reason = attempt.failure_reason or "Initial attempt failed."
    if eligibility_reason == RepairIneligibilityReason.MAX_REPAIR_ATTEMPTS_ZERO:
        return f"{base_reason} Repair was disabled (max_repair_attempts=0)."
    if eligibility_reason == RepairIneligibilityReason.NON_REPAIRABLE_GATE_FAILURE:
        detail = trigger or "non-repairable gate failure"
        return f"{base_reason} Repair blocked: {detail}."
    return base_reason


def _persist_success(
    *,
    state: PipelineState,
    attempt: PipelineAttempt,
    artifact_root: Path,
    run_id: str,
    brief: SystemBrief,
    clock: MonotonicClock,
    pipeline_start: float,
) -> PipelineState:
    assert state.architecture is not None
    assert attempt.artifacts is not None
    assert attempt.review is not None

    state.stage = PipelineStage.PERSISTENCE
    wall_clock_duration_ms = max(0.0, (clock() - pipeline_start) * 1000.0)

    try:
        final_dir, _generated_files = persist_run_artifacts(
            artifact_root=artifact_root,
            run_id=run_id,
            brief=brief,
            architecture=state.architecture,
            bundle=attempt.artifacts,
            review=attempt.review,
            state=state,
            successful_attempt=attempt,
            wall_clock_duration_ms=wall_clock_duration_ms,
        )
    except (
        ArtifactPersistenceError,
        OSError,
        SymlinkArtifactRootError,
        ValidationError,
        FileExistsError,
    ) as exc:
        attempt.outcome = AttemptOutcome.FAILED
        attempt.failure_reason = f"Artifact persistence failed: {_sanitize_failure_reason(exc)}"
        state.final_artifacts = None
        state.success = False
        return _finalize_failed_state(
            state=state,
            clock=clock,
            pipeline_start=pipeline_start,
            failure_reason=f"Artifact persistence failed: {_sanitize_failure_reason(exc)}",
            failure_category=PipelineFailureCategory.PERSISTENCE_FAILED,
            artifact_root=artifact_root,
            run_id=run_id,
        )

    attempt.outcome = AttemptOutcome.SUCCESS
    state.final_artifacts = attempt.artifacts
    state.artifacts = attempt.artifacts
    state.review = attempt.review
    state.gate_results = attempt.gate_results
    state.artifact_directory = str(final_dir)
    state.stages_completed.append(PipelineStage.PERSISTENCE)
    state.stage = PipelineStage.COMPLETED
    state.success = True
    state.pipeline_ended_at_ms = clock() * 1000.0
    state.wall_clock_duration_ms = wall_clock_duration_ms
    return state


def _category_for_attempt(attempt: PipelineAttempt) -> PipelineFailureCategory:
    reviewer_status = attempt.review.status if attempt.review is not None else None
    return infer_evaluable_failure_category(
        reviewer_status=reviewer_status,
        gate_results=attempt.gate_results,
    )


def _finalize_failed_state(
    *,
    state: PipelineState,
    clock: MonotonicClock,
    pipeline_start: float,
    failure_reason: str,
    artifact_root: Path,
    run_id: str,
    failure_category: PipelineFailureCategory | None = None,
) -> PipelineState:
    state.stage = PipelineStage.FAILED
    state.success = False
    state.failure_reason = failure_reason
    state.failure_category = failure_category
    state.final_artifacts = None
    state.pipeline_ended_at_ms = clock() * 1000.0
    state.wall_clock_duration_ms = max(0.0, (clock() - pipeline_start) * 1000.0)
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
    first_line = message.splitlines()[0]
    if len(first_line) > 500:
        return f"{first_line[:497]}..."
    return first_line[:2000]


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
    except (ArtifactPersistenceError, OSError, SymlinkArtifactRootError, ValidationError):
        return None

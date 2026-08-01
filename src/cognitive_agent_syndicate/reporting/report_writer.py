"""Run report generation for pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

from cognitive_agent_syndicate.orchestration.state import PipelineStage, PipelineState
from cognitive_agent_syndicate.schemas import (
    AttemptOutcome,
    AttemptSummary,
    GateResult,
    PipelineAttempt,
    ReviewReport,
    ReviewStatus,
    RunReport,
    UsageMetrics,
)

REPORT_LIMITATIONS = [
    (
        "Generated code was parsed and statically inspected but never executed "
        "during this pipeline run."
    ),
    "Gate evaluation is deterministic and does not run tests or scripts.",
    (
        "Forbidden-content detection is a limited static policy check, "
        "not a complete security scanner."
    ),
]


def build_run_report(state: PipelineState, generated_files: list[str]) -> RunReport:
    final_review = state.review
    reviewer_status = final_review.status if final_review is not None else None
    manifest = sorted(generated_files)
    attempt_summaries = [
        AttemptSummary(
            attempt_number=attempt.attempt_number,
            outcome=attempt.outcome,
            reviewer_status=attempt.reviewer_status,
            gates_passed=attempt.gates_passed,
            reviewer_approved=attempt.reviewer_approved,
            failure_reason=attempt.failure_reason,
            usage=attempt.usage,
            duration_ms=max(0.0, attempt.duration_ms),
        )
        for attempt in state.attempts
    ]
    final_gates = state.gate_results
    if state.attempts:
        final_gates = state.attempts[-1].gate_results

    return RunReport(
        run_id=state.run_id,
        brief_title=state.brief.title,
        gates=final_gates,
        usage=state.usage,
        success=state.success,
        artifact_count=len(manifest),
        stages_completed=[stage.value for stage in state.stages_completed],
        reviewer_status=reviewer_status,
        failure_reason=state.failure_reason,
        generated_files=manifest,
        limitations=list(REPORT_LIMITATIONS),
        repair_attempted=state.repair_attempted,
        attempt_count=state.attempt_count or max(1, len(state.attempts)),
        attempts=attempt_summaries,
        repair_trigger=state.repair_trigger,
        wall_clock_duration_ms=max(0.0, state.wall_clock_duration_ms),
        provider_latency_ms=state.usage.latency_ms,
    )


def _single_agent_failure_reason(*, review: ReviewReport, gates_passed: bool) -> str:
    reasons: list[str] = []
    if not gates_passed:
        reasons.append("One or more deterministic gates failed.")
    if review.status != ReviewStatus.APPROVED:
        reasons.append(f"Reviewer status is {review.status.value}, not approved.")
    return " ".join(reasons) if reasons else "Trial did not meet success criteria."


def build_single_agent_run_report(
    *,
    run_id: str,
    brief_title: str,
    gate_results: list[GateResult],
    usage: UsageMetrics,
    success: bool,
    generated_files: list[str],
    review: ReviewReport,
    wall_clock_duration_ms: float,
    gates_passed: bool,
) -> RunReport:
    """Build a run report for the single-agent benchmark baseline."""
    reviewer_approved = review.status == ReviewStatus.APPROVED
    resolved_wall_clock = max(0.0, wall_clock_duration_ms)
    failure_reason = (
        None
        if success
        else _single_agent_failure_reason(
            review=review,
            gates_passed=gates_passed,
        )
    )
    attempt = AttemptSummary(
        attempt_number=1,
        outcome=AttemptOutcome.SUCCESS if success else AttemptOutcome.FAILED,
        reviewer_status=review.status,
        gates_passed=gates_passed,
        reviewer_approved=reviewer_approved,
        failure_reason=failure_reason,
        usage=usage,
        duration_ms=resolved_wall_clock,
    )
    return RunReport(
        run_id=run_id,
        brief_title=brief_title,
        gates=gate_results,
        usage=usage,
        success=success,
        artifact_count=len(generated_files),
        stages_completed=[
            PipelineStage.SINGLE_AGENT_GENERATION.value,
            PipelineStage.REVIEWER.value,
            PipelineStage.GATES.value,
            PipelineStage.COMPLETED.value,
        ],
        reviewer_status=review.status,
        failure_reason=failure_reason,
        generated_files=sorted(generated_files),
        limitations=list(REPORT_LIMITATIONS),
        repair_attempted=False,
        attempt_count=1,
        attempts=[attempt],
        wall_clock_duration_ms=resolved_wall_clock,
        provider_latency_ms=usage.latency_ms,
    )


def build_success_run_report_snapshot(
    *,
    state: PipelineState,
    successful_attempt: PipelineAttempt,
    generated_files: list[str],
    wall_clock_duration_ms: float,
) -> RunReport:
    """Build a success run-report snapshot without mutating pipeline state."""
    manifest = sorted(generated_files)
    attempt_summaries = [
        AttemptSummary(
            attempt_number=attempt.attempt_number,
            outcome=(
                AttemptOutcome.SUCCESS
                if attempt.attempt_number == successful_attempt.attempt_number
                else attempt.outcome
            ),
            reviewer_status=attempt.reviewer_status,
            gates_passed=attempt.gates_passed,
            reviewer_approved=attempt.reviewer_approved,
            failure_reason=attempt.failure_reason,
            usage=attempt.usage,
            duration_ms=max(0.0, attempt.duration_ms),
        )
        for attempt in state.attempts
    ]
    final_review = (
        successful_attempt.review if successful_attempt.review is not None else state.review
    )
    reviewer_status = final_review.status if final_review is not None else None
    stages_completed = [
        *state.stages_completed,
        PipelineStage.PERSISTENCE,
        PipelineStage.COMPLETED,
    ]

    return RunReport(
        run_id=state.run_id,
        brief_title=state.brief.title,
        gates=successful_attempt.gate_results,
        usage=state.usage,
        success=True,
        artifact_count=len(manifest),
        stages_completed=[stage.value for stage in stages_completed],
        reviewer_status=reviewer_status,
        failure_reason=None,
        generated_files=manifest,
        limitations=list(REPORT_LIMITATIONS),
        repair_attempted=state.repair_attempted,
        attempt_count=state.attempt_count or max(1, len(state.attempts)),
        attempts=attempt_summaries,
        repair_trigger=state.repair_trigger,
        wall_clock_duration_ms=max(0.0, wall_clock_duration_ms),
        provider_latency_ms=state.usage.latency_ms,
    )


def write_run_reports(
    *,
    run_dir: Path,
    state: PipelineState,
    generated_files: list[str],
    report: RunReport | None = None,
    report_stage: PipelineStage | None = None,
) -> None:
    resolved_report = report or build_run_report(state, generated_files)
    current_stage = report_stage or state.stage
    json_path = run_dir / "run-report.json"
    markdown_path = run_dir / "run-report.md"

    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("Run report files already exist.")

    json_path.write_text(
        json.dumps(resolved_report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_run_report_markdown(resolved_report, current_stage=current_stage),
        encoding="utf-8",
    )


def render_run_report_markdown(report: RunReport, *, current_stage: PipelineStage) -> str:
    reviewer_decision = report.reviewer_status.value if report.reviewer_status else "n/a"
    gate_lines = [
        f"- **{gate.gate_name}**: {gate.status.value} — {gate.message}" for gate in report.gates
    ]
    file_lines = [f"- `{path}`" for path in report.generated_files]
    limitation_lines = [f"- {item}" for item in report.limitations]
    stages = ", ".join(report.stages_completed) if report.stages_completed else "none"
    outcome = "success" if report.success else "failure"

    attempt_rows = [
        "| Attempt | Outcome | Reviewer | Gates | Duration (ms) | Provider tokens |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for attempt in report.attempts:
        reviewer = attempt.reviewer_status.value if attempt.reviewer_status else "n/a"
        gates = "passed" if attempt.gates_passed else "failed"
        attempt_rows.append(
            f"| {attempt.attempt_number} | {attempt.outcome.value} | {reviewer} | "
            f"{gates} | {attempt.duration_ms:.1f} | {attempt.usage.total_tokens} |"
        )

    is_single_agent_baseline = (
        PipelineStage.SINGLE_AGENT_GENERATION.value in report.stages_completed
    )
    attempt_scope_note = (
        "Single-agent attempt provider tokens include the baseline generation and reviewer calls."
        if is_single_agent_baseline
        else (
            "Contract attempt provider tokens include implementer, reviewer, and repair provider "
            "calls for that attempt only. Architecture generation tokens are included in usage "
            "totals but excluded from attempt rows."
        )
    )

    sections = [
        "# Pipeline Run Report",
        "",
        f"- **Run ID**: `{report.run_id}`",
        f"- **Brief title**: {report.brief_title}",
        f"- **Stages completed**: {stages}",
        f"- **Current stage**: {current_stage.value}",
        f"- **Reviewer decision**: {reviewer_decision}",
        f"- **Outcome**: {outcome}",
        f"- **Repair attempted**: {'yes' if report.repair_attempted else 'no'}",
        f"- **Attempt count**: {report.attempt_count}",
    ]

    if report.repair_trigger:
        sections.extend(["", f"- **Repair trigger**: {report.repair_trigger}"])

    if report.failure_reason:
        sections.extend(["", f"- **Failure reason**: {report.failure_reason}"])

    sections.extend(
        [
            "",
            "## Attempt summary",
            "",
            *(attempt_rows if report.attempts else ["- none"]),
            "",
            f"> {attempt_scope_note}" if report.attempts else "",
            "",
            "## Deterministic gate results (final attempt)",
            "",
            *(gate_lines or ["- none"]),
            "",
            "## Generated file manifest",
            "",
            *(file_lines or ["- none"]),
            "",
            "## Usage totals",
            "",
            f"- Prompt tokens: {report.usage.prompt_tokens}",
            f"- Completion tokens: {report.usage.completion_tokens}",
            f"- Total tokens: {report.usage.total_tokens}",
            (
                "- Provider latency (ms, summed across all provider calls): "
                f"{report.provider_latency_ms}"
            ),
            f"- Wall-clock duration (ms, end-to-end trial): {report.wall_clock_duration_ms}",
            "",
            "## Limitations",
            "",
            *limitation_lines,
            "",
        ]
    )
    return "\n".join(sections)

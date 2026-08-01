"""Failure categorization and trial evaluation helpers."""

from __future__ import annotations

from cognitive_agent_syndicate.benchmarking.schemas import TrialFailureCategory, TrialStatus
from cognitive_agent_syndicate.orchestration.failures import (
    EVALUABLE_PIPELINE_FAILURE_CATEGORIES,
    FATAL_PIPELINE_FAILURE_CATEGORIES,
    PipelineFailureCategory,
    categorize_pipeline_exception,
    infer_evaluable_failure_category,
)
from cognitive_agent_syndicate.schemas import GateResult, GateStatus, ReviewStatus


def categorize_exception(exc: BaseException) -> TrialFailureCategory:
    """Map an exception to a safe benchmark failure category."""
    return pipeline_to_trial_category(categorize_pipeline_exception(exc))


def pipeline_to_trial_category(category: PipelineFailureCategory) -> TrialFailureCategory:
    """Convert a pipeline failure category to the benchmark equivalent."""
    return TrialFailureCategory(category.value)


def gate_passed(gate_results: list[GateResult], gate_id: str) -> bool | None:
    """Return whether a specific gate passed, or None if absent."""
    for gate in gate_results:
        if gate.gate_id == gate_id:
            return gate.status in {GateStatus.PASSED, GateStatus.SKIPPED}
    return None


def count_acceptance_criteria_from_review(
    gate_results: list[GateResult],
    *,
    total_criteria: int,
    review_passed_count: int | None = None,
) -> tuple[int, int]:
    """Return (passed, total) acceptance criteria counts."""
    if review_passed_count is not None:
        return review_passed_count, total_criteria
    for gate in gate_results:
        if gate.gate_id == "acceptance_criteria_represented_in_review":
            if gate.status == GateStatus.PASSED:
                return total_criteria, total_criteria
            return 0, total_criteria
    return 0, total_criteria


def classify_evaluable_failure(
    *,
    reviewer_status: ReviewStatus | None,
    gate_results: list[GateResult],
) -> TrialFailureCategory:
    """Classify an evaluable completed trial failure."""
    return pipeline_to_trial_category(
        infer_evaluable_failure_category(
            reviewer_status=reviewer_status,
            gate_results=gate_results,
        )
    )


def trial_status_from_failure_category(
    category: TrialFailureCategory | None,
    *,
    success: bool,
) -> TrialStatus:
    """Map a failure category and success flag to a benchmark trial status."""
    if success:
        return TrialStatus.COMPLETED
    if category is None:
        return TrialStatus.FAILED
    if category.value in {item.value for item in EVALUABLE_PIPELINE_FAILURE_CATEGORIES}:
        return TrialStatus.COMPLETED
    return TrialStatus.FAILED


def trial_status_from_pipeline_category(
    category: PipelineFailureCategory | None,
    *,
    success: bool,
) -> TrialStatus:
    """Map pipeline failure category to benchmark trial status."""
    if success:
        return TrialStatus.COMPLETED
    if category is None:
        return TrialStatus.FAILED
    if category in EVALUABLE_PIPELINE_FAILURE_CATEGORIES:
        return TrialStatus.COMPLETED
    if category in FATAL_PIPELINE_FAILURE_CATEGORIES:
        return TrialStatus.FAILED
    return TrialStatus.FAILED


def sanitize_failure_reason(message: str) -> str:
    """Return a concise, safe failure reason."""
    first_line = message.strip().splitlines()[0] if message.strip() else "Unknown error"
    if len(first_line) > 500:
        return f"{first_line[:497]}..."
    return first_line[:2000]

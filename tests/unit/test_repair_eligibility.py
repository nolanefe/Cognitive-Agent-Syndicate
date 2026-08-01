"""Unit tests for repair eligibility decisions."""

from cognitive_agent_syndicate.config import build_settings
from cognitive_agent_syndicate.orchestration.state import PipelineStage
from cognitive_agent_syndicate.schemas import (
    GateRepairability,
    GateResult,
    GateStatus,
    ReviewStatus,
)
from cognitive_agent_syndicate.validation.repair_eligibility import evaluate_repair_eligibility
from tests.fixtures.pipeline_fixtures import sample_bundle, sample_review_approved


def _gate(
    gate_id: str,
    *,
    status: GateStatus = GateStatus.FAILED,
    repairable: GateRepairability = GateRepairability.REPAIRABLE,
) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        gate_name=gate_id,
        status=status,
        message=f"{gate_id} outcome",
        duration_ms=1.0,
        repairable=repairable,
    )


def test_reviewer_rejection_is_repairable() -> None:
    review = sample_review_approved().model_copy(update={"status": ReviewStatus.REJECTED})
    result = evaluate_repair_eligibility(
        settings=build_settings(max_repair_attempts=1),
        stage=PipelineStage.GATES,
        artifacts=sample_bundle(),
        review=review,
        gate_results=[_gate("acceptance_criteria_represented_in_review", status=GateStatus.PASSED)],
        repair_attempted=False,
        provider_failure=False,
    )

    assert result.eligible is True
    assert result.trigger is not None


def test_repairable_gate_failure_permits_repair() -> None:
    review = sample_review_approved()
    result = evaluate_repair_eligibility(
        settings=build_settings(max_repair_attempts=1),
        stage=PipelineStage.GATES,
        artifacts=sample_bundle(),
        review=review,
        gate_results=[_gate("python_syntax")],
        repair_attempted=False,
        provider_failure=False,
    )

    assert result.eligible is True
    assert "python_syntax" in (result.trigger or "")


def test_non_repairable_path_failure_prevents_repair() -> None:
    review = sample_review_approved().model_copy(update={"status": ReviewStatus.REJECTED})
    result = evaluate_repair_eligibility(
        settings=build_settings(max_repair_attempts=1),
        stage=PipelineStage.GATES,
        artifacts=sample_bundle(),
        review=review,
        gate_results=[
            _gate(
                "paths_comply_with_permitted_prefixes",
                repairable=GateRepairability.NON_REPAIRABLE,
            )
        ],
        repair_attempted=False,
        provider_failure=False,
    )

    assert result.eligible is False


def test_provider_failure_prevents_repair() -> None:
    result = evaluate_repair_eligibility(
        settings=build_settings(max_repair_attempts=1),
        stage=PipelineStage.FAILED,
        artifacts=None,
        review=None,
        gate_results=[],
        repair_attempted=False,
        provider_failure=True,
    )

    assert result.eligible is False


def test_max_repair_attempts_zero_prevents_repair() -> None:
    review = sample_review_approved().model_copy(update={"status": ReviewStatus.REJECTED})
    result = evaluate_repair_eligibility(
        settings=build_settings(max_repair_attempts=0),
        stage=PipelineStage.GATES,
        artifacts=sample_bundle(),
        review=review,
        gate_results=[_gate("python_syntax")],
        repair_attempted=False,
        provider_failure=False,
    )

    assert result.eligible is False


def test_second_repair_is_impossible() -> None:
    review = sample_review_approved().model_copy(update={"status": ReviewStatus.REJECTED})
    result = evaluate_repair_eligibility(
        settings=build_settings(max_repair_attempts=1),
        stage=PipelineStage.GATES,
        artifacts=sample_bundle(),
        review=review,
        gate_results=[_gate("python_syntax")],
        repair_attempted=True,
        provider_failure=False,
    )

    assert result.eligible is False

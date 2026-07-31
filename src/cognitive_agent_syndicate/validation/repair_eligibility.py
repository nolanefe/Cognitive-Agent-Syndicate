"""Deterministic repair eligibility decisions for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from cognitive_agent_syndicate.config import Settings
from cognitive_agent_syndicate.orchestration.state import PipelineStage
from cognitive_agent_syndicate.schemas import (
    ArtifactBundle,
    GateRepairability,
    GateResult,
    GateStatus,
    ReviewReport,
    ReviewStatus,
)
from cognitive_agent_syndicate.validation.gates import GateRunner


class RepairIneligibilityReason(StrEnum):
    MAX_REPAIR_ATTEMPTS_ZERO = "max_repair_attempts_is_zero"
    REPAIR_ALREADY_ATTEMPTED = "repair_already_attempted"
    ARCHITECT_NOT_COMPLETED = "architect_not_completed"
    IMPLEMENTER_NOT_COMPLETED = "implementer_not_completed"
    NO_ARTIFACT_BUNDLE = "no_artifact_bundle"
    PROVIDER_FAILURE = "provider_failure"
    NON_REPAIRABLE_GATE_FAILURE = "non_repairable_gate_failure"
    NOT_REVIEWER_OR_GATE_FAILURE = "not_reviewer_or_repairable_gate_failure"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class RepairEligibility:
    eligible: bool
    reason: RepairIneligibilityReason | None = None
    trigger: str | None = None


def evaluate_repair_eligibility(
    *,
    settings: Settings,
    stage: PipelineStage,
    artifacts: ArtifactBundle | None,
    review: ReviewReport | None,
    gate_results: list[GateResult],
    repair_attempted: bool,
    provider_failure: bool,
) -> RepairEligibility:
    """Return whether a single bounded repair attempt may proceed."""
    if settings.max_repair_attempts == 0:
        return RepairEligibility(False, RepairIneligibilityReason.MAX_REPAIR_ATTEMPTS_ZERO)

    if repair_attempted:
        return RepairEligibility(False, RepairIneligibilityReason.REPAIR_ALREADY_ATTEMPTED)

    if provider_failure:
        return RepairEligibility(False, RepairIneligibilityReason.PROVIDER_FAILURE)

    if stage == PipelineStage.FAILED and artifacts is None:
        return RepairEligibility(False, RepairIneligibilityReason.PROVIDER_FAILURE)

    if artifacts is None:
        return RepairEligibility(False, RepairIneligibilityReason.NO_ARTIFACT_BUNDLE)

    if review is None:
        return RepairEligibility(False, RepairIneligibilityReason.PROVIDER_FAILURE)

    if GateRunner.has_non_repairable_failure(gate_results):
        return RepairEligibility(
            False,
            RepairIneligibilityReason.NON_REPAIRABLE_GATE_FAILURE,
            trigger=_non_repairable_trigger(gate_results),
        )

    reviewer_approved = review.status == ReviewStatus.APPROVED
    gates_passed = GateRunner.all_required_passed(gate_results)

    if reviewer_approved and gates_passed:
        return RepairEligibility(
            False,
            RepairIneligibilityReason.NOT_REVIEWER_OR_GATE_FAILURE,
        )

    failed_gates = GateRunner.failed_required(gate_results)
    repairable_gate_failures = [
        gate for gate in failed_gates if gate.repairable == GateRepairability.REPAIRABLE
    ]

    if not reviewer_approved or repairable_gate_failures:
        trigger_parts: list[str] = []
        if not reviewer_approved:
            trigger_parts.append(f"reviewer status {review.status.value}")
        if repairable_gate_failures:
            gate_names = ", ".join(gate.gate_id for gate in repairable_gate_failures)
            trigger_parts.append(f"failed gates: {gate_names}")
        return RepairEligibility(
            True,
            trigger="; ".join(trigger_parts),
        )

    return RepairEligibility(
        False,
        RepairIneligibilityReason.NOT_REVIEWER_OR_GATE_FAILURE,
    )


def _non_repairable_trigger(gate_results: list[GateResult]) -> str:
    failed = [
        gate.gate_id
        for gate in gate_results
        if gate.required
        and gate.status == GateStatus.FAILED
        and gate.repairable == GateRepairability.NON_REPAIRABLE
    ]
    return f"non-repairable gate failures: {', '.join(failed)}"


def build_permitted_file_changes(
    *,
    current_bundle: ArtifactBundle,
    permitted_paths: list[str],
) -> list[str]:
    """List paths that may be added, replaced, or removed during repair."""
    existing = sorted({file.path for file in current_bundle.files})
    permitted = sorted(permitted_paths)
    return sorted(set(existing + permitted))

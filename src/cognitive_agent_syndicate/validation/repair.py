"""Build bounded repair requests from pipeline failures."""

from __future__ import annotations

from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    GateResult,
    GateStatus,
    RepairInstruction,
    RepairRequest,
    ReviewFinding,
    ReviewReport,
    ReviewSeverity,
    ReviewStatus,
    SystemBrief,
)
from cognitive_agent_syndicate.validation.gates import GateRunner
from cognitive_agent_syndicate.validation.repair_eligibility import build_permitted_file_changes


def build_repair_request(
    *,
    brief: SystemBrief,
    architecture: ArchitectureSpec,
    current_bundle: ArtifactBundle,
    gate_results: list[GateResult],
    review: ReviewReport,
    allowed_technologies: list[str],
    permitted_paths: list[str],
    implementation_constraints: list[str],
) -> RepairRequest:
    """Construct a RepairRequest from exact deterministic failures."""
    gate_failures = GateRunner.failed_required(gate_results)
    reviewer_findings = _select_reviewer_findings(review)
    instructions = _build_repair_instructions(gate_failures, review)
    permitted_file_changes = build_permitted_file_changes(
        current_bundle=current_bundle,
        permitted_paths=permitted_paths,
    )

    return RepairRequest(
        brief=brief,
        architecture=architecture,
        current_bundle=current_bundle,
        gate_failures=gate_failures,
        reviewer_findings=reviewer_findings,
        allowed_technologies=sorted(allowed_technologies)[:20],
        permitted_paths=sorted(permitted_paths)[:20],
        implementation_constraints=sorted(implementation_constraints)[:20],
        permitted_file_changes=permitted_file_changes,
        repair_instructions=instructions,
    )


def _select_reviewer_findings(review: ReviewReport) -> list[ReviewFinding]:
    if review.status == ReviewStatus.APPROVED:
        return []
    selected = [
        finding
        for finding in review.findings
        if finding.severity in {ReviewSeverity.ERROR, ReviewSeverity.CRITICAL}
        or (finding.category.value == "acceptance_criterion" and finding.passed is False)
    ]
    if selected:
        return selected[:100]
    return list(review.findings)[:100]


def _build_repair_instructions(
    gate_failures: list[GateResult],
    review: ReviewReport,
) -> list[RepairInstruction]:
    instructions: list[RepairInstruction] = []
    for gate in gate_failures:
        if gate.status != GateStatus.FAILED:
            continue
        instructions.append(
            RepairInstruction(
                source="gate",
                gate_id=gate.gate_id,
                message=gate.message,
            )
        )
    if review.status != ReviewStatus.APPROVED:
        for finding in _select_reviewer_findings(review):
            instructions.append(
                RepairInstruction(
                    source="reviewer",
                    message=finding.message,
                    suggestion=finding.suggestion,
                )
            )
        for repair_hint in review.recommended_repairs[:10]:
            instructions.append(
                RepairInstruction(
                    source="reviewer",
                    message=repair_hint,
                )
            )
    return instructions[:50]

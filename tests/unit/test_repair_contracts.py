"""Unit tests for Stage 3 repair contracts and prompts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cognitive_agent_syndicate.prompts import (
    build_implementer_repair_system_instructions,
    build_implementer_repair_user_content,
)
from cognitive_agent_syndicate.schemas import (
    GateRepairability,
    GateResult,
    GateStatus,
    RepairInstruction,
    RepairRequest,
)
from cognitive_agent_syndicate.validation.repair import build_repair_request
from tests.fixtures.pipeline_fixtures import (
    sample_architecture,
    sample_brief,
    sample_bundle,
    sample_review_approved,
)


def _failed_gate(
    gate_id: str, *, repairable: GateRepairability = GateRepairability.REPAIRABLE
) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        gate_name=gate_id,
        status=GateStatus.FAILED,
        message=f"{gate_id} failed",
        duration_ms=1.0,
        repairable=repairable,
    )


def test_repair_request_constructs_with_required_fields() -> None:
    rejected = sample_review_approved().model_copy(update={"status": "rejected"})
    request = build_repair_request(
        brief=sample_brief(),
        architecture=sample_architecture(),
        current_bundle=sample_bundle(),
        gate_results=[_failed_gate("required_common_project_files")],
        review=rejected,
        allowed_technologies=["python"],
        permitted_paths=["src", "tests"],
        implementation_constraints=["offline only"],
    )

    assert isinstance(request, RepairRequest)
    assert request.brief.title
    assert request.current_bundle.files
    assert request.gate_failures
    assert request.permitted_file_changes


def test_repair_request_excludes_secrets_and_hidden_prompts() -> None:
    rejected = sample_review_approved().model_copy(update={"status": "rejected"})
    request = build_repair_request(
        brief=sample_brief(),
        architecture=sample_architecture(),
        current_bundle=sample_bundle(),
        gate_results=[_failed_gate("python_syntax")],
        review=rejected,
        allowed_technologies=["python"],
        permitted_paths=["src"],
        implementation_constraints=["safe"],
    )
    payload = request.model_dump(mode="json")
    serialized = json.dumps(payload)

    assert "api_key" not in serialized.lower()
    assert "sk-" not in serialized
    assert "traceback" not in serialized.lower()
    assert "system_instructions" not in serialized.lower()
    assert "stack" not in serialized.lower()


def test_repair_user_content_includes_gate_and_reviewer_failures() -> None:
    rejected = sample_review_approved().model_copy(update={"status": "rejected"})
    request = build_repair_request(
        brief=sample_brief(),
        architecture=sample_architecture(),
        current_bundle=sample_bundle(),
        gate_results=[_failed_gate("required_common_project_files")],
        review=rejected,
        allowed_technologies=["python"],
        permitted_paths=["pyproject.toml", "src", "tests"],
        implementation_constraints=["no secrets"],
    )
    content = build_implementer_repair_user_content(request)

    assert "required_common_project_files" in content
    assert "permitted_paths" in content
    assert "implementation_constraints" in content
    assert "one repair attempt" in content.lower()


def test_repair_prompts_are_deterministic() -> None:
    rejected = sample_review_approved().model_copy(update={"status": "rejected"})
    request = build_repair_request(
        brief=sample_brief(),
        architecture=sample_architecture(),
        current_bundle=sample_bundle(),
        gate_results=[_failed_gate("python_syntax")],
        review=rejected,
        allowed_technologies=["python"],
        permitted_paths=["src"],
        implementation_constraints=["safe"],
    )

    assert (
        build_implementer_repair_system_instructions()
        == build_implementer_repair_system_instructions()
    )
    assert build_implementer_repair_user_content(request) == build_implementer_repair_user_content(
        request
    )


def test_repair_system_instructions_define_boundaries() -> None:
    text = build_implementer_repair_system_instructions()
    assert "one repair attempt" in text.lower()
    assert "permitted" in text.lower()
    assert "secrets" in text.lower()


def test_repair_instruction_requires_bounded_message() -> None:
    with pytest.raises(ValidationError):
        RepairInstruction(source="gate", message="")

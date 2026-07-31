"""Unit tests for pipeline contract schemas."""

import json

import pytest
from pydantic import ValidationError

from cognitive_agent_syndicate.schemas import (
    AcceptanceCriterion,
    ArchitectureSpec,
    ArtifactBundle,
    ComponentSpec,
    GateResult,
    GateStatus,
    GeneratedFile,
    ReviewCategory,
    ReviewFinding,
    ReviewReport,
    ReviewSeverity,
    ReviewStatus,
    RunReport,
    SystemBrief,
    UsageMetrics,
)


def _sample_criterion() -> AcceptanceCriterion:
    return AcceptanceCriterion(id="ac-1", description="Expose a health endpoint.")


def _sample_brief() -> SystemBrief:
    return SystemBrief(
        title="RAG Pipeline",
        description="Build a low-latency retrieval pipeline.",
        acceptance_criteria=[_sample_criterion()],
    )


def _sample_architecture() -> ArchitectureSpec:
    return ArchitectureSpec(
        summary="Event-driven ingestion service.",
        assumptions=["Documents arrive asynchronously."],
        components=[
            ComponentSpec(
                name="ingestor",
                description="Accepts document uploads.",
                responsibilities=["validate payloads"],
            )
        ],
        dependencies=["object storage"],
        security_constraints=["Encrypt data at rest."],
        acceptance_criteria=[_sample_criterion()],
        implementation_risks=["Large uploads may exceed memory limits."],
    )


def test_system_brief_constructs_with_valid_acceptance_criteria() -> None:
    brief = _sample_brief()

    assert brief.title == "RAG Pipeline"
    assert len(brief.acceptance_criteria) == 1
    assert brief.acceptance_criteria[0].must_pass is True


def test_system_brief_rejects_empty_acceptance_criteria() -> None:
    with pytest.raises(ValidationError):
        SystemBrief(
            title="Empty Criteria",
            description="Missing acceptance criteria should fail validation.",
            acceptance_criteria=[],
        )


def test_architecture_spec_rejects_empty_acceptance_criteria() -> None:
    with pytest.raises(ValidationError):
        ArchitectureSpec(
            summary="Incomplete architecture.",
            components=[
                ComponentSpec(
                    name="ingestor",
                    description="Accepts document uploads.",
                )
            ],
            acceptance_criteria=[],
        )


def test_generated_file_normalizes_relative_paths() -> None:
    generated_file = GeneratedFile(path="src\\main.py", content="print('ok')")

    assert generated_file.path == "src/main.py"


def test_generated_file_rejects_absolute_paths() -> None:
    with pytest.raises(ValidationError):
        GeneratedFile(path="/etc/passwd", content="secret")

    with pytest.raises(ValidationError):
        GeneratedFile(path="C:/Windows/system.ini", content="secret")

    with pytest.raises(ValidationError):
        GeneratedFile(path="C:file", content="secret")


def test_generated_file_rejects_path_traversal() -> None:
    with pytest.raises(ValidationError):
        GeneratedFile(path="src/../secrets.env", content="leak")


def test_generated_file_rejects_unc_and_empty_segments() -> None:
    with pytest.raises(ValidationError):
        GeneratedFile(path="//server/share/file.txt", content="secret")

    with pytest.raises(ValidationError):
        GeneratedFile(path="src//main.py", content="secret")


def test_artifact_bundle_rejects_duplicate_paths() -> None:
    with pytest.raises(ValidationError):
        ArtifactBundle(
            files=[
                GeneratedFile(path="src/main.py", content="print('a')"),
                GeneratedFile(path="src/main.py", content="print('b')"),
            ]
        )


def test_artifact_bundle_rejects_canonical_collisions() -> None:
    with pytest.raises(ValidationError):
        ArtifactBundle(
            files=[
                GeneratedFile(path="src/main.py", content="print('a')"),
                GeneratedFile(path="src\\main.py", content="print('b')"),
            ]
        )

    with pytest.raises(ValidationError):
        ArtifactBundle(
            files=[
                GeneratedFile(path="src/main.py", content="print('a')"),
                GeneratedFile(path="./src/main.py", content="print('b')"),
            ]
        )

    with pytest.raises(ValidationError):
        ArtifactBundle(
            files=[
                GeneratedFile(path="src/main.py", content="print('a')"),
                GeneratedFile(path="SRC/main.py", content="print('b')"),
            ]
        )


def test_schemas_serialize_to_json() -> None:
    architecture = _sample_architecture()
    bundle = ArtifactBundle(
        files=[GeneratedFile(path="src/main.py", content="print('ok')", language="python")]
    )
    report = ReviewReport(
        status=ReviewStatus.APPROVED,
        findings=[
            ReviewFinding(
                criterion_id="ac-1",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="All checks passed.",
                passed=True,
            )
        ],
        summary="Ready for delivery.",
        unsupported_assumptions=["Assumes single-region deployment."],
        contract_violations=[],
        security_concerns=["Review TLS configuration."],
        recommended_repairs=["Add retry policy."],
    )

    payload = {
        "brief": _sample_brief().model_dump(mode="json"),
        "architecture": architecture.model_dump(mode="json"),
        "bundle": bundle.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
    }

    serialized = json.dumps(payload)
    round_trip = json.loads(serialized)

    assert SystemBrief.model_validate(round_trip["brief"]).title == "RAG Pipeline"
    assert ArchitectureSpec.model_validate(round_trip["architecture"]).summary.startswith("Event")
    assert ReviewReport.model_validate(round_trip["report"]).status == ReviewStatus.APPROVED


def test_usage_metrics_require_non_negative_values() -> None:
    with pytest.raises(ValidationError):
        UsageMetrics(
            prompt_tokens=-1,
            completion_tokens=0,
            total_tokens=-1,
            latency_ms=0.0,
        )

    with pytest.raises(ValidationError):
        UsageMetrics(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=20,
            latency_ms=0.0,
        )


def test_run_report_rejects_negative_timing_metrics() -> None:
    with pytest.raises(ValidationError):
        GateResult(
            gate_name="schema_validation",
            status=GateStatus.PASSED,
            message="ok",
            duration_ms=-1.0,
        )


def test_review_report_rejects_invalid_status() -> None:
    with pytest.raises(ValidationError):
        ReviewReport.model_validate(
            {
                "status": "maybe",
                "findings": [],
                "summary": "Invalid status should fail.",
            }
        )


def test_review_finding_requires_valid_category() -> None:
    with pytest.raises(ValidationError):
        ReviewFinding.model_validate(
            {
                "criterion_id": "ac-1",
                "category": "unknown",
                "severity": "info",
                "message": "Invalid category should fail.",
            }
        )


def test_acceptance_criterion_finding_requires_passed() -> None:
    with pytest.raises(ValidationError):
        ReviewFinding.model_validate(
            {
                "criterion_id": "ac-1",
                "category": "acceptance_criterion",
                "severity": "info",
                "message": "Missing passed flag should fail.",
            }
        )


def test_run_report_constructs_with_valid_metrics() -> None:
    usage = UsageMetrics(
        prompt_tokens=100,
        completion_tokens=40,
        total_tokens=140,
        latency_ms=25.0,
    )
    report = RunReport(
        run_id="run-001",
        brief_title="RAG Pipeline",
        gates=[
            GateResult(
                gate_name="contract_validation",
                status=GateStatus.PASSED,
                message="All contracts validated.",
                duration_ms=12.5,
            )
        ],
        usage=usage,
        success=True,
        artifact_count=1,
    )

    assert report.success is True
    assert report.usage.total_tokens == 140

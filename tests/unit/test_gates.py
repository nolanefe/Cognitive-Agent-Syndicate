"""Unit tests for deterministic pipeline gates."""

from cognitive_agent_syndicate.config import Settings, build_settings
from cognitive_agent_syndicate.schemas import (
    ArtifactBundle,
    GateStatus,
    GeneratedFile,
    ReviewCategory,
    ReviewFinding,
    ReviewReport,
    ReviewSeverity,
    ReviewStatus,
)
from cognitive_agent_syndicate.validation.gates import DEFAULT_GATES, GateRunner
from tests.fixtures.pipeline_fixtures import (
    sample_architecture,
    sample_brief,
    sample_bundle,
    sample_review_approved,
)


def _run_gates(
    *,
    bundle: ArtifactBundle | None = None,
    review: ReviewReport | None = None,
    settings: Settings | None = None,
    permitted_paths: list[str] | None = None,
    required_project_files: list[str] | None = None,
) -> list:
    runner = GateRunner()
    return runner.run(
        brief=sample_brief(),
        architecture=sample_architecture(),
        bundle=bundle or sample_bundle(),
        review=review or sample_review_approved(),
        settings=settings or Settings(_env_file=None),
        permitted_paths=permitted_paths or ["pyproject.toml", "src", "tests"],
        required_project_files=required_project_files or ["pyproject.toml"],
    )


def test_successful_bundle_passes_gates() -> None:
    results = _run_gates()
    assert all(result.status in {GateStatus.PASSED, GateStatus.SKIPPED} for result in results)


def test_excessive_file_count_fails_gate() -> None:
    files = [
        GeneratedFile(path=f"src/demo/file_{index}.py", content="x = 1\n") for index in range(25)
    ]
    settings = Settings(_env_file=None, max_generated_files=20)
    results = _run_gates(bundle=ArtifactBundle(files=files), settings=settings)

    failed = [result for result in results if result.gate_name == "file_count_within_limit"]
    assert failed[0].status == GateStatus.FAILED


def test_excessive_content_size_fails_gate() -> None:
    settings = build_settings(max_output_chars=100)
    bundle = ArtifactBundle(files=[GeneratedFile(path="src/demo/big.py", content="x" * 200)])
    results = _run_gates(bundle=bundle, settings=settings)

    failed = [result for result in results if result.gate_name == "total_content_within_limit"]
    assert failed[0].status == GateStatus.FAILED


def test_disallowed_path_prefix_fails_gate() -> None:
    bundle = ArtifactBundle(files=[GeneratedFile(path="secrets/creds.env", content="token=demo\n")])
    results = _run_gates(bundle=bundle, permitted_paths=["src", "tests"])

    failed = [
        result for result in results if result.gate_name == "paths_comply_with_permitted_prefixes"
    ]
    assert failed[0].status == GateStatus.FAILED


def test_approval_with_error_finding_fails_gate() -> None:
    review = ReviewReport(
        status=ReviewStatus.APPROVED,
        findings=[
            ReviewFinding(
                criterion_id="ac-create",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.ERROR,
                message="Missing create endpoint.",
                passed=True,
            ),
            ReviewFinding(
                criterion_id="ac-resolve",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="Resolve covered.",
                passed=True,
            ),
            ReviewFinding(
                criterion_id="ac-validate",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="Validation covered.",
                passed=True,
            ),
        ],
        summary="Incorrectly approved.",
    )
    results = _run_gates(review=review)

    failed = [
        result
        for result in results
        if result.gate_name == "reviewer_status_consistent_with_findings"
    ]
    assert failed[0].status == GateStatus.FAILED


def test_gate_ordering_is_deterministic() -> None:
    names_first = [gate.gate_name for gate in _run_gates()]
    names_second = [gate.gate_name for gate in _run_gates()]
    assert names_first == names_second
    assert [gate.__name__ for gate in DEFAULT_GATES] == [
        "gate_artifact_bundle_non_empty",
        "gate_file_count_within_limit",
        "gate_total_content_within_limit",
        "gate_paths_comply_with_permitted_prefixes",
        "gate_python_syntax",
        "gate_required_common_project_files",
        "gate_architecture_data_model_consistency",
        "gate_file_hierarchy_collision",
        "gate_forbidden_generated_content",
        "gate_acceptance_criteria_represented_in_review",
        "gate_reviewer_status_consistent_with_findings",
    ]


def test_missing_required_project_file_fails_gate() -> None:
    results = _run_gates(required_project_files=["README.md"])
    failed = [result for result in results if result.gate_name == "required_common_project_files"]
    assert failed[0].status == GateStatus.FAILED


def test_missing_acceptance_criterion_finding_fails_gate() -> None:
    review = ReviewReport(
        status=ReviewStatus.APPROVED,
        findings=[
            ReviewFinding(
                criterion_id="ac-create",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="Only one criterion reviewed.",
                passed=True,
            )
        ],
        summary="Incomplete review.",
    )
    results = _run_gates(review=review)
    failed = [
        result
        for result in results
        if result.gate_name == "acceptance_criteria_represented_in_review"
    ]
    assert failed[0].status == GateStatus.FAILED


def test_criterion_present_with_passed_true_succeeds() -> None:
    results = _run_gates(review=sample_review_approved())
    passed = [
        result
        for result in results
        if result.gate_name == "acceptance_criteria_represented_in_review"
    ]
    assert passed[0].status == GateStatus.PASSED


def test_criterion_present_with_passed_false_fails() -> None:
    review = sample_review_approved()
    findings = list(review.findings)
    findings[0] = findings[0].model_copy(update={"passed": False})
    review = review.model_copy(update={"findings": findings})
    results = _run_gates(review=review)
    failed = [
        result
        for result in results
        if result.gate_name == "acceptance_criteria_represented_in_review"
    ]
    assert failed[0].status == GateStatus.FAILED


def test_info_severity_without_passed_cannot_satisfy_criterion() -> None:
    review = ReviewReport(
        status=ReviewStatus.APPROVED,
        findings=[
            ReviewFinding(
                criterion_id="ac-create",
                category=ReviewCategory.QUALITY,
                severity=ReviewSeverity.INFO,
                message="Mentioned create path without acceptance category.",
            ),
            ReviewFinding(
                criterion_id="ac-resolve",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="Resolve covered.",
                passed=True,
            ),
            ReviewFinding(
                criterion_id="ac-validate",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="Validation covered.",
                passed=True,
            ),
        ],
        summary="Wrong category for one criterion.",
    )
    results = _run_gates(review=review)
    failed = [
        result
        for result in results
        if result.gate_name == "acceptance_criteria_represented_in_review"
    ]
    assert failed[0].status == GateStatus.FAILED


def test_wrong_category_cannot_satisfy_criterion() -> None:
    review = ReviewReport(
        status=ReviewStatus.APPROVED,
        findings=[
            ReviewFinding(
                criterion_id="ac-create",
                category=ReviewCategory.SECURITY,
                severity=ReviewSeverity.INFO,
                message="Security note only.",
            ),
            ReviewFinding(
                criterion_id="ac-resolve",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="Resolve covered.",
                passed=True,
            ),
            ReviewFinding(
                criterion_id="ac-validate",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="Validation covered.",
                passed=True,
            ),
        ],
        summary="Security finding cannot satisfy acceptance criterion.",
    )
    results = _run_gates(review=review)
    failed = [
        result
        for result in results
        if result.gate_name == "acceptance_criteria_represented_in_review"
    ]
    assert failed[0].status == GateStatus.FAILED


def test_approved_review_with_failed_criterion_is_rejected() -> None:
    review = sample_review_approved()
    findings = list(review.findings)
    findings[2] = findings[2].model_copy(update={"passed": False})
    review = review.model_copy(update={"findings": findings, "status": ReviewStatus.APPROVED})
    results = _run_gates(review=review)
    failed = [
        result
        for result in results
        if result.gate_name == "reviewer_status_consistent_with_findings"
    ]
    assert failed[0].status == GateStatus.FAILED

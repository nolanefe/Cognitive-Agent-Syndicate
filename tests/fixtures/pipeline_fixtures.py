"""Shared test fixtures for pipeline tests."""

from __future__ import annotations

from cognitive_agent_syndicate.schemas import (
    AcceptanceCriterion,
    ArchitectureSpec,
    ArtifactBundle,
    ComponentSpec,
    GeneratedFile,
    ReviewCategory,
    ReviewFinding,
    ReviewReport,
    ReviewSeverity,
    ReviewStatus,
    SystemBrief,
    UsageMetrics,
)


def sample_brief() -> SystemBrief:
    return SystemBrief(
        title="URL Shortener Service",
        description="Build a minimal URL shortener.",
        acceptance_criteria=[
            AcceptanceCriterion(id="ac-create", description="Create short links."),
            AcceptanceCriterion(id="ac-resolve", description="Resolve short links."),
            AcceptanceCriterion(id="ac-validate", description="Validate URLs."),
        ],
    )


def sample_architecture() -> ArchitectureSpec:
    brief = sample_brief()
    return ArchitectureSpec(
        summary="Minimal URL shortener.",
        components=[
            ComponentSpec(
                name="service",
                description="Core service.",
                responsibilities=["create", "resolve"],
            )
        ],
        acceptance_criteria=brief.acceptance_criteria,
    )


def sample_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        files=[
            GeneratedFile(path="pyproject.toml", content="[project]\nname='demo'\n"),
            GeneratedFile(path="src/demo/__init__.py", content='"""Demo."""\n'),
            GeneratedFile(path="src/demo/service.py", content="def run() -> None:\n    pass\n"),
            GeneratedFile(
                path="tests/test_service.py",
                content="def test_run() -> None:\n    pass\n",
            ),
        ]
    )


def sample_review_approved() -> ReviewReport:
    return ReviewReport(
        status=ReviewStatus.APPROVED,
        findings=[
            ReviewFinding(
                criterion_id="ac-create",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="Create path covered.",
                passed=True,
            ),
            ReviewFinding(
                criterion_id="ac-resolve",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="Resolve path covered.",
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
        summary="Approved.",
    )


def sample_usage(*, prompt: int = 10, completion: int = 5, latency: float = 1.0) -> UsageMetrics:
    return UsageMetrics(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        latency_ms=latency,
    )

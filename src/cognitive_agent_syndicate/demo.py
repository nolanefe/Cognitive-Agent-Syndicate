"""Built-in deterministic demo responses for offline mock runs."""

from __future__ import annotations

import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from cognitive_agent_syndicate.providers.mock import MockModelProvider
from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    ComponentSpec,
    DataModelSpec,
    EndpointSpec,
    GeneratedFile,
    ReviewCategory,
    ReviewFinding,
    ReviewReport,
    ReviewSeverity,
    ReviewStatus,
    SystemBrief,
    UsageMetrics,
)

URL_SHORTENER_BRIEF_TITLE = "URL Shortener Service"
_CANONICAL_BRIEF_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "briefs" / "url_shortener.json"
)


class MockScenario(StrEnum):
    SUCCESS = "success"
    REPAIR_SUCCESS = "repair-success"
    REPAIR_FAILURE = "repair-failure"


@lru_cache
def canonical_url_shortener_brief() -> SystemBrief:
    """Return the canonical bundled URL shortener demo brief."""
    payload = json.loads(_CANONICAL_BRIEF_PATH.read_text(encoding="utf-8"))
    return SystemBrief.model_validate(payload)


def create_demo_provider(
    *,
    per_stage_usage: UsageMetrics | None = None,
    scenario: MockScenario = MockScenario.SUCCESS,
) -> MockModelProvider:
    """Return a mock provider configured for the URL shortener demo."""
    usage = per_stage_usage or UsageMetrics(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        latency_ms=2.0,
    )
    provider = MockModelProvider(usage=usage)
    _configure_url_shortener_responses(provider, scenario=scenario)
    return provider


def _good_architecture(brief: SystemBrief) -> ArchitectureSpec:
    return ArchitectureSpec(
        summary="Minimal in-memory URL shortener with create and resolve endpoints.",
        assumptions=["Single-process deployment", "In-memory storage only"],
        components=[
            ComponentSpec(
                name="url_shortener_service",
                description="Core encode/decode logic.",
                responsibilities=["validate URLs", "generate codes", "resolve codes"],
            )
        ],
        endpoints=[
            EndpointSpec(
                path="/links",
                method="POST",
                description="Create a short link.",
                request_model="CreateLinkRequest",
                response_model="CreateLinkResponse",
            ),
            EndpointSpec(
                path="/{code}",
                method="GET",
                description="Resolve a short code.",
                response_model="RedirectResponse",
            ),
        ],
        data_models=[
            DataModelSpec(
                name="CreateLinkRequest",
                description="Request to create a short link.",
                fields=["url: str"],
            ),
            DataModelSpec(
                name="CreateLinkResponse",
                description="Response containing a short code.",
                fields=["code: str"],
            ),
            DataModelSpec(
                name="RedirectResponse",
                description="Resolved URL for redirect.",
                fields=["url: str"],
            ),
        ],
        dependencies=["pydantic"],
        security_constraints=["Reject malformed URLs", "Do not persist secrets"],
        acceptance_criteria=list(brief.acceptance_criteria),
        implementation_risks=["In-memory store resets on restart"],
    )


def _good_bundle() -> ArtifactBundle:
    return ArtifactBundle(
        files=[
            GeneratedFile(
                path="pyproject.toml",
                content=(
                    '[project]\nname = "url-shortener"\nversion = "0.1.0"\n'
                    'requires-python = ">=3.11"\n'
                ),
                language="toml",
            ),
            GeneratedFile(
                path="src/url_shortener/__init__.py",
                content='"""URL shortener package."""\n',
                language="python",
            ),
            GeneratedFile(
                path="src/url_shortener/service.py",
                content=(
                    '"""In-memory URL shortener service."""\n\n'
                    "from __future__ import annotations\n\n"
                    "import re\n\n"
                    "_URL_PATTERN = re.compile(r'^https?://\\S+$')\n"
                    "_store: dict[str, str] = {}\n\n\n"
                    "def validate_url(url: str) -> None:\n"
                    "    if not _URL_PATTERN.match(url):\n"
                    '        raise ValueError("Invalid URL")\n\n\n'
                    "def create_short_code(url: str) -> str:\n"
                    "    validate_url(url)\n"
                    "    code = f'c{len(_store) + 1}'\n"
                    "    _store[code] = url\n"
                    "    return code\n\n\n"
                    "def resolve_code(code: str) -> str:\n"
                    "    if code not in _store:\n"
                    '        raise KeyError("Unknown code")\n'
                    "    return _store[code]\n"
                ),
                language="python",
            ),
            GeneratedFile(
                path="tests/test_service.py",
                content=(
                    '"""Tests for url shortener service."""\n\n'
                    "from url_shortener.service import "
                    "create_short_code, resolve_code, validate_url\n\n\n"
                    "def test_create_and_resolve() -> None:\n"
                    '    code = create_short_code("https://example.com")\n'
                    '    assert resolve_code(code) == "https://example.com"\n'
                ),
                language="python",
            ),
        ]
    )


def _initial_failed_bundle() -> ArtifactBundle:
    bundle = _good_bundle()
    return ArtifactBundle(files=[file for file in bundle.files if file.path != "pyproject.toml"])


def _approved_review(brief: SystemBrief) -> ReviewReport:
    return ReviewReport(
        status=ReviewStatus.APPROVED,
        findings=[
            ReviewFinding(
                criterion_id="ac-create",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="Create endpoint represented in service module.",
                passed=True,
            ),
            ReviewFinding(
                criterion_id="ac-resolve",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="Resolve logic present in service module.",
                passed=True,
            ),
            ReviewFinding(
                criterion_id="ac-validate",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="URL validation helper rejects malformed URLs.",
                passed=True,
            ),
        ],
        summary="Artifacts satisfy the brief and architecture contracts.",
        unsupported_assumptions=[],
        contract_violations=[],
        security_concerns=[],
        recommended_repairs=[],
    )


def _rejected_review() -> ReviewReport:
    return ReviewReport(
        status=ReviewStatus.REJECTED,
        findings=[
            ReviewFinding(
                criterion_id="ac-create",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.ERROR,
                message="Missing pyproject.toml and incomplete project layout.",
                passed=False,
            ),
            ReviewFinding(
                criterion_id="ac-resolve",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.WARNING,
                message="Resolve logic present but project metadata missing.",
                passed=False,
            ),
            ReviewFinding(
                criterion_id="ac-validate",
                category=ReviewCategory.ACCEPTANCE_CRITERION,
                severity=ReviewSeverity.INFO,
                message="URL validation helper present.",
                passed=True,
            ),
        ],
        summary="Artifacts missing required project files.",
        unsupported_assumptions=[],
        contract_violations=["Missing pyproject.toml"],
        security_concerns=[],
        recommended_repairs=["Add pyproject.toml under permitted paths."],
    )


def _configure_url_shortener_responses(
    provider: MockModelProvider,
    *,
    scenario: MockScenario,
) -> None:
    brief = canonical_url_shortener_brief()
    architecture = _good_architecture(brief)
    good_bundle = _good_bundle()
    failed_bundle = _initial_failed_bundle()
    approved = _approved_review(brief)
    rejected = _rejected_review()

    provider.configure_response(ArchitectureSpec, architecture)

    if scenario == MockScenario.SUCCESS:
        provider.configure_response(ArtifactBundle, good_bundle)
        provider.configure_response(ReviewReport, approved)
        return

    if scenario == MockScenario.REPAIR_SUCCESS:
        provider.configure_response_sequence(ArtifactBundle, [failed_bundle, good_bundle])
        provider.configure_response_sequence(ReviewReport, [rejected, approved])
        return

    if scenario == MockScenario.REPAIR_FAILURE:
        provider.configure_response_sequence(ArtifactBundle, [failed_bundle, failed_bundle])
        provider.configure_response_sequence(ReviewReport, [rejected, rejected])
        return

    raise ValueError(f"Unsupported mock scenario: {scenario}")


def is_url_shortener_demo_brief(brief: SystemBrief) -> bool:
    """Return True when the brief exactly matches the bundled demo brief."""
    return brief.model_dump() == canonical_url_shortener_brief().model_dump()


def parse_mock_scenario(value: str) -> MockScenario:
    """Parse and validate a mock scenario name."""
    try:
        return MockScenario(value)
    except ValueError as exc:
        valid = ", ".join(item.value for item in MockScenario)
        raise ValueError(f"Invalid mock scenario {value!r}. Expected one of: {valid}.") from exc

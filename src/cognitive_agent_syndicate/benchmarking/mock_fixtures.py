"""Deterministic mock responses for benchmark trials."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkMode,
    BenchmarkTask,
    SingleAgentDelivery,
)
from cognitive_agent_syndicate.providers.base import GenerationResult, T
from cognitive_agent_syndicate.providers.errors import ProviderConnectionError
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

MOCK_BENCHMARK_DISCLAIMER = (
    "Mock benchmark results validate the benchmark harness and do not measure real model quality."
)


class MockBenchmarkScenario(StrEnum):
    IMMEDIATE_SUCCESS = "immediate_success"
    REVIEWER_REJECTION = "reviewer_rejection"
    GATE_FAILURE = "gate_failure"
    REPAIR_SUCCESS = "repair_success"
    REPAIR_FAILURE = "repair_failure"
    PROVIDER_FAILURE = "provider_failure"


@dataclass
class MockBenchmarkProvider:
    """Mock provider wrapper that can fail deterministically."""

    inner: MockModelProvider
    fail_on_generate: bool = False
    calls: list[object] = field(default_factory=list)
    payloads: list[tuple[str, str]] = field(default_factory=list)

    async def generate(
        self,
        *,
        system_instructions: str,
        user_content: str,
        response_type: type[T],
    ) -> GenerationResult[T]:
        self.calls.append(response_type)
        self.payloads.append((system_instructions, user_content))
        if self.fail_on_generate:
            raise ProviderConnectionError("Mock provider failure for benchmark task")
        return await self.inner.generate(
            system_instructions=system_instructions,
            user_content=user_content,
            response_type=response_type,
        )


def scenario_for_trial(task_id: str, mode: BenchmarkMode) -> MockBenchmarkScenario:
    """Return the deterministic mock scenario for a task and mode."""
    mapping: dict[tuple[str, BenchmarkMode], MockBenchmarkScenario] = {
        ("task-url-shortener", BenchmarkMode.SINGLE_AGENT): MockBenchmarkScenario.IMMEDIATE_SUCCESS,
        ("task-url-shortener", BenchmarkMode.CONTRACT_NO_REPAIR): (
            MockBenchmarkScenario.IMMEDIATE_SUCCESS
        ),
        ("task-url-shortener", BenchmarkMode.CONTRACT_WITH_REPAIR): (
            MockBenchmarkScenario.IMMEDIATE_SUCCESS
        ),
        ("task-support-ticket", BenchmarkMode.SINGLE_AGENT): (
            MockBenchmarkScenario.REVIEWER_REJECTION
        ),
        ("task-support-ticket", BenchmarkMode.CONTRACT_NO_REPAIR): (
            MockBenchmarkScenario.IMMEDIATE_SUCCESS
        ),
        ("task-support-ticket", BenchmarkMode.CONTRACT_WITH_REPAIR): (
            MockBenchmarkScenario.IMMEDIATE_SUCCESS
        ),
        ("task-document-ingestion", BenchmarkMode.SINGLE_AGENT): MockBenchmarkScenario.GATE_FAILURE,
        ("task-document-ingestion", BenchmarkMode.CONTRACT_NO_REPAIR): (
            MockBenchmarkScenario.GATE_FAILURE
        ),
        ("task-document-ingestion", BenchmarkMode.CONTRACT_WITH_REPAIR): (
            MockBenchmarkScenario.REPAIR_SUCCESS
        ),
        ("task-feature-flag", BenchmarkMode.SINGLE_AGENT): MockBenchmarkScenario.IMMEDIATE_SUCCESS,
        ("task-feature-flag", BenchmarkMode.CONTRACT_NO_REPAIR): (
            MockBenchmarkScenario.REVIEWER_REJECTION
        ),
        ("task-feature-flag", BenchmarkMode.CONTRACT_WITH_REPAIR): (
            MockBenchmarkScenario.REPAIR_SUCCESS
        ),
        ("task-inventory-reservation", BenchmarkMode.SINGLE_AGENT): (
            MockBenchmarkScenario.IMMEDIATE_SUCCESS
        ),
        ("task-inventory-reservation", BenchmarkMode.CONTRACT_NO_REPAIR): (
            MockBenchmarkScenario.GATE_FAILURE
        ),
        ("task-inventory-reservation", BenchmarkMode.CONTRACT_WITH_REPAIR): (
            MockBenchmarkScenario.REPAIR_FAILURE
        ),
        (
            "task-incident-summary",
            BenchmarkMode.SINGLE_AGENT,
        ): MockBenchmarkScenario.PROVIDER_FAILURE,
        ("task-incident-summary", BenchmarkMode.CONTRACT_NO_REPAIR): (
            MockBenchmarkScenario.PROVIDER_FAILURE
        ),
        ("task-incident-summary", BenchmarkMode.CONTRACT_WITH_REPAIR): (
            MockBenchmarkScenario.PROVIDER_FAILURE
        ),
    }
    return mapping[(task_id, mode)]


def create_benchmark_mock_provider(
    task: BenchmarkTask,
    mode: BenchmarkMode,
    *,
    usage: UsageMetrics | None = None,
) -> MockBenchmarkProvider:
    """Create a deterministic mock provider for one benchmark trial."""
    scenario = scenario_for_trial(task.task_id, mode)
    per_call_usage = usage or UsageMetrics(
        prompt_tokens=80,
        completion_tokens=40,
        total_tokens=120,
        latency_ms=3.0,
    )
    inner = MockModelProvider(usage=per_call_usage)

    if scenario == MockBenchmarkScenario.PROVIDER_FAILURE:
        return MockBenchmarkProvider(inner=inner, fail_on_generate=True)

    brief = task.brief
    architecture = _architecture_for_task(task)
    good_bundle = _good_bundle_for_task(task)
    bad_bundle = _bad_bundle_for_task(task)
    approved = _approved_review(brief)
    rejected = _rejected_review(brief)

    if mode == BenchmarkMode.SINGLE_AGENT:
        delivery = SingleAgentDelivery(architecture=architecture, artifacts=good_bundle)
        if scenario == MockBenchmarkScenario.GATE_FAILURE:
            delivery = SingleAgentDelivery(architecture=architecture, artifacts=bad_bundle)
        inner.configure_response(SingleAgentDelivery, delivery)
        review = approved if scenario != MockBenchmarkScenario.REVIEWER_REJECTION else rejected
        inner.configure_response(ReviewReport, review)
        return MockBenchmarkProvider(inner=inner)

    inner.configure_response(ArchitectureSpec, architecture)
    if scenario == MockBenchmarkScenario.IMMEDIATE_SUCCESS:
        inner.configure_response(ArtifactBundle, good_bundle)
        inner.configure_response(ReviewReport, approved)
    elif scenario == MockBenchmarkScenario.REVIEWER_REJECTION:
        inner.configure_response(ArtifactBundle, good_bundle)
        inner.configure_response(ReviewReport, rejected)
    elif scenario == MockBenchmarkScenario.GATE_FAILURE:
        inner.configure_response(ArtifactBundle, bad_bundle)
        inner.configure_response(ReviewReport, approved)
    elif scenario == MockBenchmarkScenario.REPAIR_SUCCESS:
        inner.configure_response_sequence(ArtifactBundle, [bad_bundle, good_bundle])
        inner.configure_response_sequence(ReviewReport, [rejected, approved])
    elif scenario == MockBenchmarkScenario.REPAIR_FAILURE:
        inner.configure_response_sequence(ArtifactBundle, [bad_bundle, bad_bundle])
        inner.configure_response_sequence(ReviewReport, [rejected, rejected])

    return MockBenchmarkProvider(inner=inner)


def _architecture_for_task(task: BenchmarkTask) -> ArchitectureSpec:
    brief = task.brief
    endpoints: list[EndpointSpec] = []
    data_models: list[DataModelSpec] = []

    if task.task_id == "task-url-shortener":
        endpoints = [
            EndpointSpec(
                path="/links",
                method="POST",
                description="Create short link",
                request_model="CreateLinkRequest",
                response_model="CreateLinkResponse",
            ),
            EndpointSpec(
                path="/{code}",
                method="GET",
                description="Resolve code",
                response_model="RedirectResponse",
            ),
        ]
        data_models = [
            DataModelSpec(
                name="CreateLinkRequest", description="Create request", fields=["url: str"]
            ),
            DataModelSpec(
                name="CreateLinkResponse", description="Create response", fields=["code: str"]
            ),
            DataModelSpec(name="RedirectResponse", description="Redirect", fields=["url: str"]),
        ]
    elif task.task_id == "task-support-ticket":
        endpoints = [
            EndpointSpec(
                path="/classify",
                method="POST",
                description="Classify ticket",
                request_model="TicketRequest",
                response_model="ClassificationResponse",
            )
        ]
        data_models = [
            DataModelSpec(name="TicketRequest", description="Ticket input", fields=["text: str"]),
            DataModelSpec(
                name="ClassificationResponse",
                description="Classification output",
                fields=["category: str", "confidence: float"],
            ),
        ]
    elif task.task_id == "task-document-ingestion":
        endpoints = [
            EndpointSpec(
                path="/documents",
                method="POST",
                description="Ingest document",
                request_model="DocumentRequest",
                response_model="DocumentResponse",
            )
        ]
        data_models = [
            DataModelSpec(
                name="DocumentRequest", description="Document payload", fields=["title: str"]
            ),
            DataModelSpec(
                name="DocumentResponse", description="Stored document", fields=["id: str"]
            ),
        ]
    elif task.task_id == "task-feature-flag":
        endpoints = [
            EndpointSpec(
                path="/flags/{name}",
                method="GET",
                description="Evaluate flag",
                response_model="FlagEvaluation",
            )
        ]
        data_models = [
            DataModelSpec(
                name="FlagEvaluation", description="Flag state", fields=["enabled: bool"]
            ),
        ]
    elif task.task_id == "task-inventory-reservation":
        endpoints = [
            EndpointSpec(
                path="/reservations",
                method="POST",
                description="Reserve inventory",
                request_model="ReservationRequest",
                response_model="ReservationResponse",
            )
        ]
        data_models = [
            DataModelSpec(name="ReservationRequest", description="Reserve", fields=["sku: str"]),
            DataModelSpec(name="ReservationResponse", description="Reserved", fields=["id: str"]),
        ]
    else:
        endpoints = [
            EndpointSpec(
                path="/summaries",
                method="POST",
                description="Summarize incident",
                request_model="IncidentRequest",
                response_model="IncidentSummary",
            )
        ]
        data_models = [
            DataModelSpec(
                name="IncidentRequest", description="Incident", fields=["events: list[str]"]
            ),
            DataModelSpec(name="IncidentSummary", description="Summary", fields=["summary: str"]),
        ]

    return ArchitectureSpec(
        summary=f"Minimal architecture for {task.title}.",
        assumptions=["Single-process deployment"],
        components=[
            ComponentSpec(
                name=task.task_id.replace("-", "_"),
                description=task.title,
                responsibilities=["implement core logic"],
            )
        ],
        endpoints=endpoints,
        data_models=data_models,
        dependencies=["pydantic"],
        security_constraints=["Do not persist secrets", "Validate all inputs"],
        acceptance_criteria=list(brief.acceptance_criteria),
        implementation_risks=["In-memory storage only"],
    )


def _service_module(task: BenchmarkTask) -> GeneratedFile:
    module_name = task.task_id.replace("task-", "").replace("-", "_")
    return GeneratedFile(
        path=f"src/{module_name}/service.py",
        content=(
            f'"""Service module for {task.title}."""\n\n'
            "from __future__ import annotations\n\n\n"
            "def handle() -> str:\n"
            '    return "ok"\n'
        ),
        language="python",
    )


def _good_bundle_for_task(task: BenchmarkTask) -> ArtifactBundle:
    module_name = task.task_id.replace("task-", "").replace("-", "_")
    files = [
        GeneratedFile(
            path="pyproject.toml",
            content=(
                f'[project]\nname = "{module_name}"\nversion = "0.1.0"\n'
                'requires-python = ">=3.11"\n'
            ),
            language="toml",
        ),
        GeneratedFile(
            path=f"src/{module_name}/__init__.py",
            content=f'"""Package for {task.title}."""\n',
            language="python",
        ),
        _service_module(task),
        GeneratedFile(
            path=f"tests/test_{module_name}.py",
            content=(
                f'"""Tests for {module_name}."""\n\n'
                f"from {module_name}.service import handle\n\n\n"
                "def test_handle() -> None:\n"
                '    assert handle() == "ok"\n'
            ),
            language="python",
        ),
    ]
    return ArtifactBundle(files=files)


def _bad_bundle_for_task(task: BenchmarkTask) -> ArtifactBundle:
    bundle = _good_bundle_for_task(task)
    return ArtifactBundle(files=[f for f in bundle.files if f.path != "pyproject.toml"])


def _forbidden_bundle_for_task(task: BenchmarkTask) -> ArtifactBundle:
    bundle = _good_bundle_for_task(task)
    module_name = task.task_id.replace("task-", "").replace("-", "_")
    bad_service = GeneratedFile(
        path=f"src/{module_name}/service.py",
        content=(
            f'"""Unsafe service for {task.title}."""\n\n'
            "import os\n\n\n"
            "def handle() -> str:\n"
            "    os.system('echo unsafe')\n"
            '    return "bad"\n'
        ),
        language="python",
    )
    return ArtifactBundle(
        files=[f for f in bundle.files if not f.path.endswith("service.py")] + [bad_service]
    )


def _approved_review(brief: SystemBrief) -> ReviewReport:
    findings = [
        ReviewFinding(
            criterion_id=criterion.id,
            category=ReviewCategory.ACCEPTANCE_CRITERION,
            severity=ReviewSeverity.INFO,
            message=f"Criterion {criterion.id} satisfied.",
            passed=True,
        )
        for criterion in brief.acceptance_criteria
    ]
    return ReviewReport(
        status=ReviewStatus.APPROVED,
        findings=findings,
        summary="Artifacts satisfy the brief and architecture contracts.",
        unsupported_assumptions=[],
        contract_violations=[],
        security_concerns=[],
        recommended_repairs=[],
    )


def _rejected_review(brief: SystemBrief) -> ReviewReport:
    findings = [
        ReviewFinding(
            criterion_id=criterion.id,
            category=ReviewCategory.ACCEPTANCE_CRITERION,
            severity=ReviewSeverity.ERROR if index == 0 else ReviewSeverity.WARNING,
            message=f"Criterion {criterion.id} not fully satisfied.",
            passed=index > 0,
        )
        for index, criterion in enumerate(brief.acceptance_criteria)
    ]
    return ReviewReport(
        status=ReviewStatus.REJECTED,
        findings=findings,
        summary="Artifacts missing required elements.",
        unsupported_assumptions=[],
        contract_violations=["Missing required project files"],
        security_concerns=[],
        recommended_repairs=["Add pyproject.toml under permitted paths."],
    )

"""Regression tests for live-pilot benchmark reporting clarity."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from cognitive_agent_syndicate.benchmarking.adapters import classify_evaluable_failure
from cognitive_agent_syndicate.benchmarking.dataset import load_benchmark_dataset
from cognitive_agent_syndicate.benchmarking.metrics import build_benchmark_summary
from cognitive_agent_syndicate.benchmarking.mock_fixtures import (
    MockBenchmarkProvider,
    _architecture_for_task,
    _good_bundle_for_task,
    _rejected_review,
    create_benchmark_mock_provider,
)
from cognitive_agent_syndicate.benchmarking.reporting import render_summary_markdown
from cognitive_agent_syndicate.benchmarking.runner import BenchmarkRunContext, run_benchmark_trial
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkMode,
    SingleAgentDelivery,
    TrialFailureCategory,
)
from cognitive_agent_syndicate.config import build_settings
from cognitive_agent_syndicate.orchestration.state import PipelineStage
from cognitive_agent_syndicate.providers.mock import MockModelProvider
from cognitive_agent_syndicate.reporting.report_writer import render_run_report_markdown
from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    ReviewCategory,
    ReviewFinding,
    ReviewReport,
    ReviewSeverity,
    ReviewStatus,
    RunReport,
    UsageMetrics,
)


class FakeClock:
    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        self._value += 0.01
        return self._value


@pytest.fixture
def url_task():
    dataset = load_benchmark_dataset(Path("benchmarks/datasets/software_delivery_v1.json"))
    return next(task for task in dataset.tasks if task.task_id == "task-url-shortener")


def _needs_revision_review(task) -> ReviewReport:
    findings = [
        ReviewFinding(
            criterion_id=criterion.id,
            category=ReviewCategory.ACCEPTANCE_CRITERION,
            severity=ReviewSeverity.WARNING,
            message=f"Criterion {criterion.id} needs minor revision.",
            passed=True,
        )
        for criterion in task.brief.acceptance_criteria
    ]
    return ReviewReport(
        status=ReviewStatus.NEEDS_REVISION,
        findings=findings,
        summary="Minor revisions recommended.",
        unsupported_assumptions=[],
        contract_violations=[],
        security_concerns=[],
        recommended_repairs=["Clarify redirect behavior in tests."],
    )


def _single_agent_needs_revision_provider(task) -> MockBenchmarkProvider:
    inner = MockModelProvider(
        usage=UsageMetrics(
            prompt_tokens=2000,
            completion_tokens=1500,
            total_tokens=3500,
            latency_ms=12000.0,
        )
    )
    delivery = SingleAgentDelivery(
        architecture=_architecture_for_task(task),
        artifacts=_good_bundle_for_task(task),
    )
    inner.configure_response(SingleAgentDelivery, delivery)
    inner.configure_response(ReviewReport, _needs_revision_review(task))
    return MockBenchmarkProvider(inner=inner)


@pytest.mark.asyncio
async def test_single_agent_wall_clock_recorded_in_trial_and_run_report(
    url_task,
    tmp_path,
) -> None:
    context = BenchmarkRunContext(
        benchmark_id="pilot-wall-clock",
        dataset_version="v1",
        model_label="test-model",
        reviewer_model_label="test-model",
    )
    result = await run_benchmark_trial(
        task=url_task,
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        context=context,
        generation_provider=_single_agent_needs_revision_provider(url_task),
        reviewer_provider=_single_agent_needs_revision_provider(url_task),
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "single_agent",
        clock=FakeClock(),
        run_id_factory=lambda: "pilot-single-agent",
    )

    assert result.trial.provider_latency_ms > 0
    assert result.trial.wall_clock_duration_ms > 0
    assert result.run_report_json is not None
    report = RunReport.model_validate(result.run_report_json)
    assert report.wall_clock_duration_ms > 0
    assert report.wall_clock_duration_ms == pytest.approx(result.trial.wall_clock_duration_ms)
    assert report.provider_latency_ms == pytest.approx(result.trial.provider_latency_ms)
    assert report.wall_clock_duration_ms != report.provider_latency_ms


@pytest.mark.asyncio
async def test_single_agent_pilot_report_shape(url_task, tmp_path) -> None:
    context = BenchmarkRunContext(
        benchmark_id="pilot-single-agent",
        dataset_version="v1",
        model_label="test-model",
        reviewer_model_label="test-model",
    )
    result = await run_benchmark_trial(
        task=url_task,
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        context=context,
        generation_provider=_single_agent_needs_revision_provider(url_task),
        reviewer_provider=_single_agent_needs_revision_provider(url_task),
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "single_agent",
        clock=FakeClock(),
        run_id_factory=lambda: "pilot-single-agent",
    )

    assert result.trial.success is False
    assert result.trial.status.value == "completed"
    assert result.trial.reviewer_status == ReviewStatus.NEEDS_REVISION
    assert result.trial.failure_category == TrialFailureCategory.REVIEWER_REJECTED
    assert result.trial.required_files_gate_passed is True

    report = RunReport.model_validate(result.run_report_json)
    assert PipelineStage.SINGLE_AGENT_GENERATION.value in report.stages_completed
    assert PipelineStage.REVIEWER.value in report.stages_completed
    assert PipelineStage.GATES.value in report.stages_completed
    assert len(report.attempts) == 1
    assert report.attempts[0].reviewer_status == ReviewStatus.NEEDS_REVISION
    assert report.attempts[0].gates_passed is True
    assert report.attempts[0].duration_ms > 0

    markdown = render_run_report_markdown(report, current_stage=PipelineStage.COMPLETED)
    assert "single_agent_generation" in markdown
    assert "needs_revision" in markdown
    assert "none" not in markdown.split("## Attempt summary")[1].split("##")[0]


@pytest.mark.asyncio
async def test_contract_no_repair_pilot_report_shape(url_task, tmp_path) -> None:
    inner = MockModelProvider()
    architecture = _architecture_for_task(url_task)
    bundle = _good_bundle_for_task(url_task)
    inner.configure_response(ArchitectureSpec, architecture)
    inner.configure_response(ArtifactBundle, bundle)
    inner.configure_response(ReviewReport, _rejected_review(url_task.brief))
    provider = MockBenchmarkProvider(inner=inner)

    context = BenchmarkRunContext(
        benchmark_id="pilot-contract-no-repair",
        dataset_version="v1",
        model_label="test-model",
        reviewer_model_label="test-model",
    )
    result = await run_benchmark_trial(
        task=url_task,
        mode=BenchmarkMode.CONTRACT_NO_REPAIR,
        repetition=1,
        context=context,
        generation_provider=provider,
        reviewer_provider=provider,
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "contract_no_repair",
        clock=FakeClock(),
        run_id_factory=lambda: "pilot-contract-no-repair",
    )

    assert result.trial.success is False
    assert result.trial.status.value == "completed"
    assert result.trial.reviewer_status == ReviewStatus.REJECTED
    assert result.trial.failure_category == TrialFailureCategory.REVIEWER_REJECTED
    assert any(
        gate.gate_id == "acceptance_criteria_represented_in_review"
        and gate.status.value == "failed"
        for gate in result.trial.gate_results
    )


@pytest.mark.asyncio
async def test_contract_with_repair_success_without_repair_attempt(url_task, tmp_path) -> None:
    run_id = uuid.uuid4().hex
    provider = create_benchmark_mock_provider(url_task, BenchmarkMode.CONTRACT_WITH_REPAIR)
    context = BenchmarkRunContext(
        benchmark_id=f"pilot-cwr-{run_id[:8]}",
        dataset_version="v1",
        model_label="test-model",
        reviewer_model_label="test-model",
    )
    result = await run_benchmark_trial(
        task=url_task,
        mode=BenchmarkMode.CONTRACT_WITH_REPAIR,
        repetition=1,
        context=context,
        generation_provider=provider,
        reviewer_provider=provider,
        settings=build_settings(provider="mock", max_repair_attempts=1),
        trial_dir=tmp_path / "contract_with_repair",
        clock=FakeClock(),
        run_id_factory=lambda: run_id,
    )

    assert result.trial.success is True
    assert result.trial.repair_attempted is False
    assert result.trial.repair_succeeded is False


def test_needs_revision_maps_to_reviewer_rejected_failure_category() -> None:
    category = classify_evaluable_failure(
        reviewer_status=ReviewStatus.NEEDS_REVISION,
        gate_results=[],
    )
    assert category == TrialFailureCategory.REVIEWER_REJECTED


def test_rejected_maps_to_reviewer_rejected_failure_category() -> None:
    category = classify_evaluable_failure(
        reviewer_status=ReviewStatus.REJECTED,
        gate_results=[],
    )
    assert category == TrialFailureCategory.REVIEWER_REJECTED


@pytest.mark.asyncio
async def test_trial_total_tokens_equal_sum_of_provider_call_usage(url_task, tmp_path) -> None:
    per_call = UsageMetrics(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        latency_ms=4.0,
    )
    provider = create_benchmark_mock_provider(
        url_task,
        BenchmarkMode.SINGLE_AGENT,
        usage=per_call,
    )
    context = BenchmarkRunContext(
        benchmark_id="token-sum",
        dataset_version="v1",
        model_label="test-model",
        reviewer_model_label="test-model",
    )
    result = await run_benchmark_trial(
        task=url_task,
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        context=context,
        generation_provider=provider,
        reviewer_provider=provider,
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "single_agent",
        clock=FakeClock(),
    )

    assert result.trial.provider_call_count == 2
    assert result.trial.total_tokens == 300
    report = RunReport.model_validate(result.run_report_json)
    assert report.usage.total_tokens == 300
    assert report.attempts[0].usage.total_tokens == 300


@pytest.mark.asyncio
async def test_contract_attempt_row_excludes_architect_tokens(url_task, tmp_path) -> None:
    provider = create_benchmark_mock_provider(url_task, BenchmarkMode.CONTRACT_NO_REPAIR)
    context = BenchmarkRunContext(
        benchmark_id="token-scope",
        dataset_version="v1",
        model_label="test-model",
        reviewer_model_label="test-model",
    )
    result = await run_benchmark_trial(
        task=url_task,
        mode=BenchmarkMode.CONTRACT_NO_REPAIR,
        repetition=1,
        context=context,
        generation_provider=provider,
        reviewer_provider=provider,
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "contract_no_repair",
        clock=FakeClock(),
    )

    report = RunReport.model_validate(result.run_report_json)
    assert result.pipeline_state is not None
    assert report.attempts[0].usage.total_tokens < report.usage.total_tokens
    assert (
        report.usage.total_tokens
        == report.attempts[0].usage.total_tokens
        + result.pipeline_state.architect_usage.total_tokens
    )

    markdown = render_run_report_markdown(report, current_stage=PipelineStage.COMPLETED)
    assert "Architecture generation tokens are included in usage totals" in markdown


def test_summary_documents_repair_attempted_semantics() -> None:
    summary = build_benchmark_summary(
        benchmark_id="pilot",
        dataset_name="software_delivery",
        dataset_version="v1",
        modes=[BenchmarkMode.CONTRACT_WITH_REPAIR],
        repetitions=1,
        model_label="test-model",
        reviewer_model_label="test-model",
        same_model_reviewer=True,
        pricing_configured=False,
        is_mock=False,
        trials=[],
        task_titles={},
    )
    markdown = render_summary_markdown(summary)
    assert "does not demonstrate repair effectiveness" in markdown
    assert "Repair attempts (actual)" in markdown

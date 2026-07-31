"""Benchmark metrics aggregation."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from statistics import median

from cognitive_agent_syndicate.benchmarking.adapters import gate_passed
from cognitive_agent_syndicate.benchmarking.display import RATE_ROUNDING_NOTE, format_dataset_label
from cognitive_agent_syndicate.benchmarking.mock_fixtures import MOCK_BENCHMARK_DISCLAIMER
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkMode,
    BenchmarkSummary,
    BenchmarkTrial,
    ModeSummary,
    TaskSummary,
    TrialStatus,
)
from cognitive_agent_syndicate.schemas import GateResult, ReviewStatus


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _attempted_trials(trials: list[BenchmarkTrial]) -> list[BenchmarkTrial]:
    return [trial for trial in trials if trial.status != TrialStatus.CANCELLED]


def _aggregate_mode_summary(trials: list[BenchmarkTrial], mode: BenchmarkMode) -> ModeSummary:
    mode_trials = [trial for trial in trials if trial.mode == mode]
    completed = [trial for trial in mode_trials if trial.status == TrialStatus.COMPLETED]
    failed = [trial for trial in mode_trials if trial.status == TrialStatus.FAILED]
    cancelled = [trial for trial in mode_trials if trial.status == TrialStatus.CANCELLED]
    attempted = _attempted_trials(mode_trials)
    successful = [trial for trial in attempted if trial.success]

    reviewer_approved = [
        trial for trial in attempted if trial.reviewer_status == ReviewStatus.APPROVED
    ]
    required_gate_passed = [
        trial for trial in attempted if trial.required_files_gate_passed is True
    ]
    acceptance_passed = [
        trial
        for trial in attempted
        if trial.acceptance_criteria_passed == trial.acceptance_criteria_total
        and trial.acceptance_criteria_total > 0
    ]
    syntax_passed = [trial for trial in attempted if trial.syntax_gate_passed is True]
    forbidden_passed = [trial for trial in attempted if trial.forbidden_content_gate_passed is True]
    repair_attempts = [trial for trial in attempted if trial.repair_attempted]
    repair_successes = [trial for trial in repair_attempts if trial.repair_succeeded]

    attempted_count = len(attempted)
    repair_attempt_count = len(repair_attempts)

    token_values = [trial.total_tokens for trial in completed]
    latency_values = [trial.provider_latency_ms for trial in completed]
    wall_values = [trial.wall_clock_duration_ms for trial in completed]

    costs = [
        trial.estimated_cost.total_cost_usd
        for trial in completed
        if trial.estimated_cost is not None
    ]
    total_cost = sum(costs, start=Decimal("0")) if costs else None
    avg_cost = (total_cost / len(costs)) if costs and total_cost is not None else None

    failure_counts = Counter(
        trial.failure_category.value for trial in mode_trials if trial.failure_category is not None
    )

    return ModeSummary(
        mode=mode,
        trial_count=len(mode_trials),
        attempted_trial_count=attempted_count,
        completed_trial_count=len(completed),
        failed_trial_count=len(failed),
        cancelled_trial_count=len(cancelled),
        successful_trials=len(successful),
        repair_attempt_count=repair_attempt_count,
        repair_success_count=len(repair_successes),
        total_observed_provider_calls=sum(trial.provider_call_count for trial in mode_trials),
        success_rate=_safe_rate(len(successful), attempted_count),
        reviewer_approval_rate=_safe_rate(len(reviewer_approved), attempted_count),
        required_gate_pass_rate=_safe_rate(len(required_gate_passed), attempted_count),
        acceptance_criterion_pass_rate=_safe_rate(len(acceptance_passed), attempted_count),
        syntax_pass_rate=_safe_rate(len(syntax_passed), attempted_count),
        forbidden_content_pass_rate=_safe_rate(len(forbidden_passed), attempted_count),
        required_files_pass_rate=_safe_rate(len(required_gate_passed), attempted_count),
        repair_attempt_rate=_safe_rate(repair_attempt_count, attempted_count),
        repair_success_rate=_safe_rate(len(repair_successes), repair_attempt_count),
        avg_total_tokens=(sum(token_values) / len(token_values)) if token_values else None,
        median_total_tokens=median(token_values) if token_values else None,
        min_total_tokens=min(token_values) if token_values else None,
        max_total_tokens=max(token_values) if token_values else None,
        avg_provider_latency_ms=(sum(latency_values) / len(latency_values))
        if latency_values
        else None,
        median_provider_latency_ms=median(latency_values) if latency_values else None,
        min_provider_latency_ms=min(latency_values) if latency_values else None,
        max_provider_latency_ms=max(latency_values) if latency_values else None,
        avg_wall_clock_duration_ms=(sum(wall_values) / len(wall_values)) if wall_values else None,
        median_wall_clock_duration_ms=median(wall_values) if wall_values else None,
        min_wall_clock_duration_ms=min(wall_values) if wall_values else None,
        max_wall_clock_duration_ms=max(wall_values) if wall_values else None,
        total_estimated_cost_usd=total_cost,
        avg_estimated_cost_usd=avg_cost,
        failure_category_counts=dict(failure_counts),
    )


def build_benchmark_summary(
    *,
    benchmark_id: str,
    dataset_name: str,
    dataset_version: str,
    modes: list[BenchmarkMode],
    repetitions: int,
    model_label: str,
    reviewer_model_label: str,
    same_model_reviewer: bool,
    pricing_configured: bool,
    is_mock: bool,
    trials: list[BenchmarkTrial],
    task_titles: dict[str, str],
) -> BenchmarkSummary:
    """Build overall benchmark summary from trial records."""
    completed = [trial for trial in trials if trial.status == TrialStatus.COMPLETED]
    failed = [trial for trial in trials if trial.status == TrialStatus.FAILED]
    cancelled = [trial for trial in trials if trial.status == TrialStatus.CANCELLED]
    attempted = _attempted_trials(trials)
    successful = [trial for trial in attempted if trial.success]

    mode_summaries = [_aggregate_mode_summary(trials, mode) for mode in modes]
    task_summaries: list[TaskSummary] = []
    for task_id, title in task_titles.items():
        task_trials = [trial for trial in trials if trial.task_id == task_id]
        task_mode_summaries = [_aggregate_mode_summary(task_trials, mode) for mode in modes]
        task_summaries.append(
            TaskSummary(task_id=task_id, title=title, mode_summaries=task_mode_summaries)
        )

    limitations = [
        "Generated code is statically inspected but never executed during benchmark runs.",
        "Gate evaluation is deterministic and does not run tests or scripts.",
        RATE_ROUNDING_NOTE,
    ]
    if same_model_reviewer:
        limitations.append(
            "Generation and review used the same provider/model, which may reduce "
            "review independence."
        )
    if is_mock:
        limitations.append(MOCK_BENCHMARK_DISCLAIMER)

    return BenchmarkSummary(
        benchmark_id=benchmark_id,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_label=format_dataset_label(dataset_name, dataset_version),
        modes=modes,
        repetitions=repetitions,
        model_label=model_label,
        reviewer_model_label=reviewer_model_label,
        same_model_reviewer=same_model_reviewer,
        pricing_configured=pricing_configured,
        is_mock=is_mock,
        total_trials=len(trials),
        attempted_trials=len(attempted),
        completed_trials=len(completed),
        failed_trials=len(failed),
        cancelled_trials=len(cancelled),
        successful_trials=len(successful),
        total_observed_provider_calls=sum(trial.provider_call_count for trial in trials),
        rate_rounding_note=RATE_ROUNDING_NOTE,
        mode_summaries=mode_summaries,
        task_summaries=task_summaries,
        limitations=limitations,
    )


def enrich_trial_gate_fields(
    trial: BenchmarkTrial,
    gate_results: list[GateResult],
    *,
    total_criteria: int = 0,
    review_passed_count: int | None = None,
) -> BenchmarkTrial:
    """Populate gate-specific trial fields from gate results."""
    from cognitive_agent_syndicate.benchmarking.adapters import (
        count_acceptance_criteria_from_review,
    )

    ac_passed, ac_total = count_acceptance_criteria_from_review(
        gate_results,
        total_criteria=total_criteria,
        review_passed_count=review_passed_count,
    )
    return trial.model_copy(
        update={
            "required_files_gate_passed": gate_passed(
                gate_results, "required_common_project_files"
            ),
            "syntax_gate_passed": gate_passed(gate_results, "python_syntax"),
            "forbidden_content_gate_passed": gate_passed(
                gate_results, "forbidden_generated_content"
            ),
            "acceptance_criteria_passed": ac_passed,
            "acceptance_criteria_total": ac_total,
        }
    )

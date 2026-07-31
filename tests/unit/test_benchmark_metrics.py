"""Tests for benchmark metrics and pricing."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from cognitive_agent_syndicate.benchmarking.metrics import build_benchmark_summary
from cognitive_agent_syndicate.benchmarking.pricing import estimate_trial_cost
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkMode,
    BenchmarkTrial,
    PricingConfig,
    TrialStatus,
)


def _trial(**overrides) -> BenchmarkTrial:
    base = {
        "benchmark_id": "b1",
        "dataset_version": "v1",
        "task_id": "task-url-shortener",
        "mode": BenchmarkMode.SINGLE_AGENT,
        "repetition": 1,
        "model_label": "mock",
        "reviewer_model_label": "mock",
        "status": TrialStatus.COMPLETED,
        "success": True,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "provider_latency_ms": 5.0,
        "wall_clock_duration_ms": 10.0,
    }
    base.update(overrides)
    return BenchmarkTrial(**base)


def test_success_rate_calculation() -> None:
    trials = [
        _trial(success=True),
        _trial(success=False, failure_category="reviewer_rejected"),
    ]
    summary = build_benchmark_summary(
        benchmark_id="b1",
        dataset_name="d",
        dataset_version="v1",
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        model_label="mock",
        reviewer_model_label="mock",
        same_model_reviewer=True,
        pricing_configured=False,
        is_mock=True,
        trials=trials,
        task_titles={"task-url-shortener": "URL"},
    )
    mode = summary.mode_summaries[0]
    assert mode.success_rate == 0.5


def test_zero_trials_safe() -> None:
    summary = build_benchmark_summary(
        benchmark_id="b1",
        dataset_name="d",
        dataset_version="v1",
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        model_label="mock",
        reviewer_model_label="mock",
        same_model_reviewer=True,
        pricing_configured=False,
        is_mock=True,
        trials=[],
        task_titles={},
    )
    assert summary.mode_summaries[0].success_rate is None


def test_median_tokens() -> None:
    trials = [
        _trial(total_tokens=100),
        _trial(total_tokens=200),
        _trial(total_tokens=300),
    ]
    summary = build_benchmark_summary(
        benchmark_id="b1",
        dataset_name="d",
        dataset_version="v1",
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=3,
        model_label="mock",
        reviewer_model_label="mock",
        same_model_reviewer=True,
        pricing_configured=False,
        is_mock=True,
        trials=trials,
        task_titles={"task-url-shortener": "URL"},
    )
    assert summary.mode_summaries[0].median_total_tokens == 200


def test_repair_rates() -> None:
    trials = [
        _trial(
            mode=BenchmarkMode.CONTRACT_WITH_REPAIR, repair_attempted=True, repair_succeeded=True
        ),
        _trial(
            mode=BenchmarkMode.CONTRACT_WITH_REPAIR, repair_attempted=True, repair_succeeded=False
        ),
    ]
    summary = build_benchmark_summary(
        benchmark_id="b1",
        dataset_name="d",
        dataset_version="v1",
        modes=[BenchmarkMode.CONTRACT_WITH_REPAIR],
        repetitions=2,
        model_label="mock",
        reviewer_model_label="mock",
        same_model_reviewer=True,
        pricing_configured=False,
        is_mock=True,
        trials=trials,
        task_titles={"task-url-shortener": "URL"},
    )
    mode = summary.mode_summaries[0]
    assert mode.repair_attempt_rate == 1.0
    assert mode.repair_success_rate == 0.5


def test_failure_category_counts() -> None:
    trials = [
        _trial(success=False, failure_category="provider_connection"),
        _trial(success=False, failure_category="provider_connection"),
    ]
    summary = build_benchmark_summary(
        benchmark_id="b1",
        dataset_name="d",
        dataset_version="v1",
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=2,
        model_label="mock",
        reviewer_model_label="mock",
        same_model_reviewer=True,
        pricing_configured=False,
        is_mock=True,
        trials=trials,
        task_titles={"task-url-shortener": "URL"},
    )
    assert summary.mode_summaries[0].failure_category_counts["provider_connection"] == 2


def test_decimal_cost_calculation() -> None:
    pricing = PricingConfig(
        model_label="m",
        input_usd_per_million_tokens=Decimal("1.0"),
        output_usd_per_million_tokens=Decimal("2.0"),
        source_or_note="test",
        effective_date=date(2026, 1, 1),
    )
    cost = estimate_trial_cost(
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        pricing=pricing,
    )
    assert cost is not None
    assert cost.total_cost_usd == Decimal("2.0")


def test_no_cost_without_pricing() -> None:
    assert estimate_trial_cost(prompt_tokens=100, completion_tokens=50, pricing=None) is None


def test_mock_warning_in_limitations() -> None:
    summary = build_benchmark_summary(
        benchmark_id="b1",
        dataset_name="d",
        dataset_version="v1",
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        model_label="mock",
        reviewer_model_label="mock",
        same_model_reviewer=True,
        pricing_configured=False,
        is_mock=True,
        trials=[_trial()],
        task_titles={"task-url-shortener": "URL"},
    )
    assert any("Mock benchmark results" in item for item in summary.limitations)

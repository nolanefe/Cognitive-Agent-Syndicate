"""Tests for benchmark planning."""

from __future__ import annotations

from cognitive_agent_syndicate.benchmarking.planning import (
    build_benchmark_plan,
    provider_calls_for_mode,
    render_benchmark_plan,
)
from cognitive_agent_syndicate.benchmarking.schemas import BenchmarkMode


def test_provider_calls_single_agent() -> None:
    assert provider_calls_for_mode(BenchmarkMode.SINGLE_AGENT) == (2, 2)


def test_provider_calls_contract_no_repair() -> None:
    assert provider_calls_for_mode(BenchmarkMode.CONTRACT_NO_REPAIR) == (3, 3)


def test_provider_calls_contract_with_repair() -> None:
    assert provider_calls_for_mode(BenchmarkMode.CONTRACT_WITH_REPAIR) == (3, 5)


def test_plan_total_trials() -> None:
    plan = build_benchmark_plan(
        dataset_name="d",
        dataset_version="v1",
        task_ids=["a", "b"],
        modes=[BenchmarkMode.SINGLE_AGENT, BenchmarkMode.CONTRACT_NO_REPAIR],
        repetitions=2,
        generation_provider="mock",
        generation_model="mock-model",
        reviewer_provider="mock",
        reviewer_model="mock-model",
        pricing_configured=False,
        is_mock=True,
        is_live=False,
    )
    assert plan.total_trials == 8


def test_plan_min_max_calls() -> None:
    plan = build_benchmark_plan(
        dataset_name="d",
        dataset_version="v1",
        task_ids=["a"],
        modes=[
            BenchmarkMode.SINGLE_AGENT,
            BenchmarkMode.CONTRACT_WITH_REPAIR,
        ],
        repetitions=1,
        generation_provider="mock",
        generation_model="mock-model",
        reviewer_provider="mock",
        reviewer_model="mock-model",
        pricing_configured=False,
        is_mock=True,
        is_live=False,
    )
    assert plan.min_provider_calls == 5
    assert plan.max_provider_calls == 7


def test_live_plan_identified() -> None:
    plan = build_benchmark_plan(
        dataset_name="d",
        dataset_version="v1",
        task_ids=["a"],
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        generation_provider="openai",
        generation_model="gpt-test",
        reviewer_provider="openai",
        reviewer_model="gpt-test",
        pricing_configured=False,
        is_mock=False,
        is_live=True,
    )
    rendered = render_benchmark_plan(plan)
    assert "live" in rendered.lower()
    assert plan.is_live is True


def test_plan_task_filter_reflected() -> None:
    plan = build_benchmark_plan(
        dataset_name="d",
        dataset_version="v1",
        task_ids=["task-a", "task-b"],
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        generation_provider="mock",
        generation_model="mock-model",
        reviewer_provider="mock",
        reviewer_model="mock-model",
        pricing_configured=True,
        is_mock=True,
        is_live=False,
    )
    assert plan.task_ids == ["task-a", "task-b"]
    assert plan.pricing_configured is True


def test_full_dataset_plan_calls_independent() -> None:
    task_count = 6
    mode_mins = {
        BenchmarkMode.SINGLE_AGENT: 2,
        BenchmarkMode.CONTRACT_NO_REPAIR: 3,
        BenchmarkMode.CONTRACT_WITH_REPAIR: 3,
    }
    mode_maxs = {
        BenchmarkMode.SINGLE_AGENT: 2,
        BenchmarkMode.CONTRACT_NO_REPAIR: 3,
        BenchmarkMode.CONTRACT_WITH_REPAIR: 5,
    }
    modes = list(mode_mins)
    expected_trials = task_count * len(modes)
    expected_min = sum(mode_mins[mode] for mode in modes) * task_count
    expected_max = sum(mode_maxs[mode] for mode in modes) * task_count

    plan = build_benchmark_plan(
        dataset_name="software_delivery",
        dataset_version="v1",
        task_ids=[f"task-{index}" for index in range(task_count)],
        modes=modes,
        repetitions=1,
        generation_provider="mock",
        generation_model="mock-model",
        reviewer_provider="mock",
        reviewer_model="mock-model",
        pricing_configured=False,
        is_mock=True,
        is_live=False,
    )
    rendered = render_benchmark_plan(plan)
    assert plan.total_trials == expected_trials == 18
    assert plan.min_provider_calls == expected_min == 48
    assert plan.max_provider_calls == expected_max == 60
    assert "Dataset: software_delivery v1" in rendered
    assert "vv1" not in rendered

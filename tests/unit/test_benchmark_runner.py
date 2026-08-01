"""Tests for benchmark runner."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from cognitive_agent_syndicate.benchmarking.dataset import load_benchmark_dataset
from cognitive_agent_syndicate.benchmarking.mock_fixtures import create_benchmark_mock_provider
from cognitive_agent_syndicate.benchmarking.runner import (
    BenchmarkRunContext,
    execute_benchmark,
    run_benchmark_trial,
)
from cognitive_agent_syndicate.benchmarking.schemas import BenchmarkMode
from cognitive_agent_syndicate.config import build_settings


class FakeClock:
    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        self._value += 0.01
        return self._value


@pytest.fixture
def dataset():
    return load_benchmark_dataset(Path("benchmarks/datasets/software_delivery_v1.json"))


def _factory(task, mode):
    return create_benchmark_mock_provider(task, mode)


@pytest.mark.asyncio
async def test_all_three_modes_run(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")
    settings = build_settings(provider="mock")
    context = BenchmarkRunContext(
        benchmark_id=f"all-modes-{uuid.uuid4().hex[:8]}",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    clock = FakeClock()
    run_ids = iter(("run-a", "run-b", "run-c"))
    for mode in BenchmarkMode:
        result = await run_benchmark_trial(
            task=task,
            mode=mode,
            repetition=1,
            context=context,
            generation_provider=_factory(task, mode),
            reviewer_provider=_factory(task, mode),
            settings=settings,
            trial_dir=tmp_path / mode.value,
            clock=clock,
            run_id_factory=lambda: next(run_ids),
        )
        assert result.trial.mode == mode
        assert result.trial.status.value == "completed"


@pytest.mark.asyncio
async def test_repetitions_numbered(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")
    settings = build_settings(provider="mock")
    context = BenchmarkRunContext(
        benchmark_id="b1",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    for rep in (1, 2):
        result = await run_benchmark_trial(
            task=task,
            mode=BenchmarkMode.SINGLE_AGENT,
            repetition=rep,
            context=context,
            generation_provider=_factory(task, BenchmarkMode.SINGLE_AGENT),
            reviewer_provider=_factory(task, BenchmarkMode.SINGLE_AGENT),
            settings=settings,
            trial_dir=tmp_path / "trial",
            clock=FakeClock(),
        )
        assert result.trial.repetition == rep


@pytest.mark.asyncio
async def test_failed_trial_does_not_abort_remaining(dataset, tmp_path) -> None:
    settings = build_settings(provider="mock")
    run, _ = await execute_benchmark(
        benchmark_id="bench-fail-continue",
        dataset=dataset,
        tasks=list(dataset.tasks),
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        settings=settings,
        output_dir=tmp_path / "benchmark_results",
        generation_provider_factory=_factory,
        is_mock=True,
        clock=FakeClock(),
        run_id_factory=lambda: "fixed-run",
    )
    assert len(run.trials) == 6


@pytest.mark.asyncio
async def test_mock_repair_success(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-document-ingestion")
    settings = build_settings(provider="mock", max_repair_attempts=1)
    context = BenchmarkRunContext(
        benchmark_id="b1",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    result = await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.CONTRACT_WITH_REPAIR,
        repetition=1,
        context=context,
        generation_provider=_factory(task, BenchmarkMode.CONTRACT_WITH_REPAIR),
        reviewer_provider=_factory(task, BenchmarkMode.CONTRACT_WITH_REPAIR),
        settings=settings,
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
    )
    assert result.trial.repair_attempted is True
    assert result.trial.repair_succeeded is True


@pytest.mark.asyncio
async def test_mock_repair_failure(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-inventory-reservation")
    settings = build_settings(provider="mock", max_repair_attempts=1)
    context = BenchmarkRunContext(
        benchmark_id="b1",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    result = await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.CONTRACT_WITH_REPAIR,
        repetition=1,
        context=context,
        generation_provider=_factory(task, BenchmarkMode.CONTRACT_WITH_REPAIR),
        reviewer_provider=_factory(task, BenchmarkMode.CONTRACT_WITH_REPAIR),
        settings=settings,
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
    )
    assert result.trial.repair_attempted is True
    assert result.trial.repair_succeeded is False


@pytest.mark.asyncio
async def test_provider_failure_category(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-incident-summary")
    settings = build_settings(provider="mock")
    context = BenchmarkRunContext(
        benchmark_id="b1",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    result = await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        context=context,
        generation_provider=_factory(task, BenchmarkMode.SINGLE_AGENT),
        reviewer_provider=_factory(task, BenchmarkMode.SINGLE_AGENT),
        settings=settings,
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
    )
    assert result.trial.failure_category is not None
    assert result.trial.failure_category.value == "provider_connection"


@pytest.mark.asyncio
async def test_reviewer_rejection_category(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-support-ticket")
    settings = build_settings(provider="mock")
    context = BenchmarkRunContext(
        benchmark_id="b1",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    result = await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        context=context,
        generation_provider=_factory(task, BenchmarkMode.SINGLE_AGENT),
        reviewer_provider=_factory(task, BenchmarkMode.SINGLE_AGENT),
        settings=settings,
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
    )
    assert result.trial.failure_category is not None
    assert result.trial.failure_category.value == "reviewer_rejected"


@pytest.mark.asyncio
async def test_usage_reconciles(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")
    settings = build_settings(provider="mock")
    context = BenchmarkRunContext(
        benchmark_id="b1",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    result = await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        context=context,
        generation_provider=_factory(task, BenchmarkMode.SINGLE_AGENT),
        reviewer_provider=_factory(task, BenchmarkMode.SINGLE_AGENT),
        settings=settings,
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
    )
    trial = result.trial
    assert trial.total_tokens == trial.prompt_tokens + trial.completion_tokens


@pytest.mark.asyncio
async def test_cancelled_trial(dataset, tmp_path) -> None:
    task = dataset.tasks[0]
    settings = build_settings(provider="mock")
    context = BenchmarkRunContext(
        benchmark_id="b1",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
        cancelled=True,
    )
    result = await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        context=context,
        generation_provider=_factory(task, BenchmarkMode.SINGLE_AGENT),
        reviewer_provider=_factory(task, BenchmarkMode.SINGLE_AGENT),
        settings=settings,
        trial_dir=tmp_path / "trial",
    )
    assert result.trial.status.value == "cancelled"


@pytest.mark.asyncio
async def test_same_model_reviewer_true_for_matching_labels(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")

    def generation_factory(task_arg, mode):
        return create_benchmark_mock_provider(task_arg, mode)

    def reviewer_factory(task_arg, mode):
        return create_benchmark_mock_provider(task_arg, mode)

    run, _ = await execute_benchmark(
        benchmark_id="same-model-true",
        dataset=dataset,
        tasks=[task],
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        settings=build_settings(provider="mock"),
        output_dir=tmp_path / "benchmark_results",
        generation_provider_factory=generation_factory,
        reviewer_provider_factory=reviewer_factory,
        reviewer_model_label="mock-model",
        reviewer_provider_label="mock",
        is_mock=True,
        clock=FakeClock(),
        run_id_factory=lambda: "fixed-run",
    )
    assert run.summary is not None
    assert run.summary.same_model_reviewer is True


@pytest.mark.asyncio
async def test_same_model_reviewer_false_for_different_provider_label(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")

    def generation_factory(task_arg, mode):
        return create_benchmark_mock_provider(task_arg, mode)

    def reviewer_factory(task_arg, mode):
        return create_benchmark_mock_provider(task_arg, mode)

    run, _ = await execute_benchmark(
        benchmark_id="same-model-false-provider",
        dataset=dataset,
        tasks=[task],
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        settings=build_settings(provider="mock"),
        output_dir=tmp_path / "benchmark_results",
        generation_provider_factory=generation_factory,
        reviewer_provider_factory=reviewer_factory,
        reviewer_model_label="mock-model",
        reviewer_provider_label="openai",
        is_mock=True,
        clock=FakeClock(),
        run_id_factory=lambda: "fixed-run",
    )
    assert run.summary is not None
    assert run.summary.same_model_reviewer is False


@pytest.mark.asyncio
async def test_same_model_reviewer_false_for_different_model_label(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")

    def generation_factory(task_arg, mode):
        return create_benchmark_mock_provider(task_arg, mode)

    def reviewer_factory(task_arg, mode):
        return create_benchmark_mock_provider(task_arg, mode)

    run, _ = await execute_benchmark(
        benchmark_id="same-model-false-model",
        dataset=dataset,
        tasks=[task],
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        settings=build_settings(provider="mock"),
        output_dir=tmp_path / "benchmark_results",
        generation_provider_factory=generation_factory,
        reviewer_provider_factory=reviewer_factory,
        reviewer_model_label="reviewer-model",
        reviewer_provider_label="mock",
        is_mock=True,
        clock=FakeClock(),
        run_id_factory=lambda: "fixed-run",
    )
    assert run.summary is not None
    assert run.summary.same_model_reviewer is False

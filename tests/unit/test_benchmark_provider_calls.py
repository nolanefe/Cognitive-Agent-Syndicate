"""Tests for exact observed provider call counts."""

from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_agent_syndicate.benchmarking.dataset import load_benchmark_dataset
from cognitive_agent_syndicate.benchmarking.mock_fixtures import create_benchmark_mock_provider
from cognitive_agent_syndicate.benchmarking.runner import (
    BenchmarkRunContext,
    execute_benchmark,
    run_benchmark_trial,
)
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkMode,
    TrialFailureCategory,
    TrialStatus,
)
from cognitive_agent_syndicate.config import build_settings
from cognitive_agent_syndicate.providers.errors import ProviderAuthenticationError
from cognitive_agent_syndicate.providers.mock import MockModelProvider


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


class FailingProvider(MockModelProvider):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    async def generate(self, *, system_instructions: str, user_content: str, response_type):
        raise self._exc


@pytest.mark.asyncio
async def test_success_single_agent_has_two_calls(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")
    provider = _factory(task, BenchmarkMode.SINGLE_AGENT)
    context = BenchmarkRunContext(
        benchmark_id="call-test",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    result = await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        context=context,
        generation_provider=provider,
        reviewer_provider=provider,
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
    )
    assert result.trial.provider_call_count == 2
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_provider_failure_single_agent_has_one_call(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-incident-summary")
    provider = _factory(task, BenchmarkMode.SINGLE_AGENT)
    context = BenchmarkRunContext(
        benchmark_id="call-test",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    result = await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        context=context,
        generation_provider=provider,
        reviewer_provider=provider,
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
    )
    assert result.trial.provider_call_count == 1
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [BenchmarkMode.CONTRACT_NO_REPAIR, BenchmarkMode.CONTRACT_WITH_REPAIR],
)
async def test_architect_failure_records_one_call(dataset, tmp_path, mode) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")
    provider = FailingProvider(ProviderAuthenticationError("authentication failed"))
    context = BenchmarkRunContext(
        benchmark_id="call-test",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    result = await run_benchmark_trial(
        task=task,
        mode=mode,
        repetition=1,
        context=context,
        generation_provider=provider,
        reviewer_provider=provider,
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
        run_id_factory=lambda: "fixed-run",
    )
    assert result.trial.status == TrialStatus.FAILED
    assert result.trial.failure_category == TrialFailureCategory.PROVIDER_AUTHENTICATION
    assert result.trial.provider_call_count == 1


@pytest.mark.asyncio
async def test_contract_no_repair_success_has_three_calls(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")
    provider = _factory(task, BenchmarkMode.CONTRACT_NO_REPAIR)
    context = BenchmarkRunContext(
        benchmark_id="call-test",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    result = await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.CONTRACT_NO_REPAIR,
        repetition=1,
        context=context,
        generation_provider=provider,
        reviewer_provider=provider,
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
        run_id_factory=lambda: "fixed-run",
    )
    assert result.trial.provider_call_count == 3
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_repair_success_has_five_calls(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-document-ingestion")
    provider = _factory(task, BenchmarkMode.CONTRACT_WITH_REPAIR)
    context = BenchmarkRunContext(
        benchmark_id="call-test",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    result = await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.CONTRACT_WITH_REPAIR,
        repetition=1,
        context=context,
        generation_provider=provider,
        reviewer_provider=provider,
        settings=build_settings(provider="mock", max_repair_attempts=1),
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
        run_id_factory=lambda: "fixed-run",
    )
    assert result.trial.provider_call_count == 5
    assert len(provider.calls) == 5


@pytest.mark.asyncio
async def test_mode_summary_observed_calls_reconcile(dataset, tmp_path) -> None:
    run, _ = await execute_benchmark(
        benchmark_id="call-reconcile",
        dataset=dataset,
        tasks=[next(t for t in dataset.tasks if t.task_id == "task-url-shortener")],
        modes=list(BenchmarkMode),
        repetitions=1,
        settings=build_settings(provider="mock"),
        output_dir=tmp_path / "benchmark_results",
        generation_provider_factory=_factory,
        is_mock=True,
        clock=FakeClock(),
        run_id_factory=lambda: "fixed-run",
    )
    assert run.summary is not None
    trial_sum = sum(trial.provider_call_count for trial in run.trials)
    mode_sum = sum(mode.total_observed_provider_calls for mode in run.summary.mode_summaries)
    assert trial_sum == mode_sum
    assert run.summary.total_observed_provider_calls == trial_sum

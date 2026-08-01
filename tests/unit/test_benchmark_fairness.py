"""Tests for benchmark fairness controls."""

from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_agent_syndicate.benchmarking.dataset import load_benchmark_dataset
from cognitive_agent_syndicate.benchmarking.mock_fixtures import create_benchmark_mock_provider
from cognitive_agent_syndicate.benchmarking.runner import BenchmarkRunContext, run_benchmark_trial
from cognitive_agent_syndicate.benchmarking.schemas import BenchmarkMode
from cognitive_agent_syndicate.config import build_settings
from cognitive_agent_syndicate.validation.gates import DEFAULT_GATES


class FakeClock:
    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        self._value += 0.01
        return self._value


@pytest.fixture
def dataset():
    return load_benchmark_dataset(Path("benchmarks/datasets/software_delivery_v1.json"))


@pytest.mark.asyncio
async def test_baseline_runs_architecture_gate(dataset, tmp_path) -> None:
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
        generation_provider=create_benchmark_mock_provider(task, BenchmarkMode.SINGLE_AGENT),
        reviewer_provider=create_benchmark_mock_provider(task, BenchmarkMode.SINGLE_AGENT),
        settings=settings,
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
    )
    gate_ids = {gate.gate_id for gate in result.trial.gate_results}
    assert "architecture_data_model_consistency" in gate_ids


@pytest.mark.asyncio
async def test_same_gate_count_across_modes(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")
    settings = build_settings(provider="mock")
    context = BenchmarkRunContext(
        benchmark_id="b1",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    gate_counts = []
    for mode in BenchmarkMode:
        result = await run_benchmark_trial(
            task=task,
            mode=mode,
            repetition=1,
            context=context,
            generation_provider=create_benchmark_mock_provider(task, mode),
            reviewer_provider=create_benchmark_mock_provider(task, mode),
            settings=settings,
            trial_dir=tmp_path / "trial",
            clock=FakeClock(),
        )
        gate_counts.append(len(result.trial.gate_results))
    assert len(set(gate_counts)) == 1
    assert gate_counts[0] == len(DEFAULT_GATES)


def test_task_constraints_identical_across_modes(dataset) -> None:
    task = dataset.tasks[0]
    context_fields = task.generation_context().model_dump()
    assert "allowed_technologies" in context_fields
    assert "required_files" in context_fields
    assert "notes" not in context_fields


@pytest.mark.asyncio
async def test_generation_payloads_share_task_context(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-feature-flag")
    settings = build_settings(provider="mock")
    context = BenchmarkRunContext(
        benchmark_id="fairness-payload",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    payloads: dict[str, list[str]] = {}

    for mode in BenchmarkMode:
        provider = create_benchmark_mock_provider(task, mode)
        await run_benchmark_trial(
            task=task,
            mode=mode,
            repetition=1,
            context=context,
            generation_provider=provider,
            reviewer_provider=provider,
            settings=settings,
            trial_dir=tmp_path / mode.value,
            clock=FakeClock(),
            run_id_factory=lambda: "fixed-run",
        )
        payloads[mode.value] = [user_content for _, user_content in provider.payloads]

    expected_snippets = [
        task.brief.title,
        task.allowed_technologies[0],
        task.required_files[0],
        task.implementation_constraints[0],
    ]
    for mode, calls in payloads.items():
        assert calls, f"expected provider calls for {mode}"
        joined = "\n".join(calls)
        for snippet in expected_snippets:
            assert snippet in joined
        assert task.notes is not None
        assert task.notes not in joined


@pytest.mark.asyncio
async def test_single_agent_uses_one_typed_generation_call(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")
    provider = create_benchmark_mock_provider(task, BenchmarkMode.SINGLE_AGENT)
    from cognitive_agent_syndicate.benchmarking.schemas import SingleAgentDelivery

    await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        context=BenchmarkRunContext(
            benchmark_id="fairness-payload",
            dataset_version="v1",
            model_label="mock-model",
            reviewer_model_label="mock-model",
        ),
        generation_provider=provider,
        reviewer_provider=provider,
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "single",
        clock=FakeClock(),
    )
    assert provider.calls[0] is SingleAgentDelivery

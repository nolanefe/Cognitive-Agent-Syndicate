"""Tests for benchmark reporting and persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cognitive_agent_syndicate.benchmarking.dataset import load_benchmark_dataset
from cognitive_agent_syndicate.benchmarking.display import RATE_ROUNDING_NOTE
from cognitive_agent_syndicate.benchmarking.metrics import build_benchmark_summary
from cognitive_agent_syndicate.benchmarking.mock_fixtures import (
    MOCK_BENCHMARK_DISCLAIMER,
    create_benchmark_mock_provider,
)
from cognitive_agent_syndicate.benchmarking.reporting import (
    CSV_COLUMNS,
    BenchmarkOutputError,
    build_config_snapshot,
    render_summary_markdown,
    resolve_benchmark_output_dir,
    trial_to_csv_row,
    write_trial_reports,
)
from cognitive_agent_syndicate.benchmarking.runner import TrialExecutionResult, execute_benchmark
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkMode,
    BenchmarkTrial,
    TrialStatus,
)
from cognitive_agent_syndicate.config import build_settings
from cognitive_agent_syndicate.orchestration.state import PipelineState
from cognitive_agent_syndicate.reporting.artifacts import ArtifactPersistenceError
from cognitive_agent_syndicate.schemas import ArtifactBundle, GeneratedFile
from tests.fixtures.pipeline_fixtures import sample_architecture, sample_brief


class FakeClock:
    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        self._value += 0.01
        return self._value


@pytest.mark.asyncio
async def test_atomic_benchmark_output(tmp_path) -> None:
    dataset = load_benchmark_dataset(Path("benchmarks/datasets/software_delivery_v1.json"))
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")
    settings = build_settings(provider="mock")

    def factory(t, m):
        return create_benchmark_mock_provider(t, m)

    run, output_path = await execute_benchmark(
        benchmark_id="bench-atomic",
        dataset=dataset,
        tasks=[task],
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        settings=settings,
        output_dir=tmp_path / "benchmark_results",
        generation_provider_factory=factory,
        is_mock=True,
        clock=FakeClock(),
        run_id_factory=lambda: "fixed",
    )
    assert output_path.exists()
    assert (output_path / "benchmark-config.json").exists()
    assert (output_path / "trials.jsonl").exists()
    assert (output_path / "summary.json").exists()
    assert (output_path / "summary.md").exists()
    assert (output_path / "results.csv").exists()
    assert (output_path / "failures.json").exists()
    assert run.summary is not None


def test_no_overwrite_existing_output(tmp_path) -> None:
    output_dir = tmp_path / "benchmark_results"
    target = output_dir / "bench-id"
    target.mkdir(parents=True)
    with pytest.raises(BenchmarkOutputError, match="already exists"):
        resolve_benchmark_output_dir(output_dir, "bench-id")


def test_csv_deterministic_columns() -> None:
    trial = BenchmarkTrial(
        benchmark_id="b1",
        dataset_version="v1",
        task_id="task-a",
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        model_label="mock",
        reviewer_model_label="mock",
        status=TrialStatus.COMPLETED,
        success=True,
    )
    row = trial_to_csv_row(trial)
    assert list(row.keys()) == CSV_COLUMNS


def test_summary_mock_warning() -> None:
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
    md = render_summary_markdown(summary)
    assert MOCK_BENCHMARK_DISCLAIMER in md


def test_summary_rounding_note_is_accurate() -> None:
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
        is_mock=False,
        trials=[],
        task_titles={},
    )
    markdown = render_summary_markdown(summary)
    assert summary.rate_rounding_note == RATE_ROUNDING_NOTE
    assert "half-up" not in summary.rate_rounding_note.lower()
    assert "half-up" not in markdown.lower()
    assert "standard Python formatting" in summary.rate_rounding_note
    assert "standard Python formatting" in markdown


def test_config_snapshot_no_secrets() -> None:
    dataset = load_benchmark_dataset(Path("benchmarks/datasets/software_delivery_v1.json"))
    config = build_config_snapshot(
        benchmark_id="b1",
        dataset=dataset,
        task_ids=["task-url-shortener"],
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        model_label="mock",
        reviewer_model_label="mock",
        reviewer_provider_label="mock",
        generation_provider_label="mock",
        pricing=None,
        is_mock=True,
        temperature=0.0,
    )
    dumped = config.model_dump_json()
    assert "api_key" not in dumped.lower()
    assert "OPENAI_API_KEY" not in dumped


@pytest.mark.asyncio
async def test_failed_artifacts_not_in_success_runs(tmp_path) -> None:
    dataset = load_benchmark_dataset(Path("benchmarks/datasets/software_delivery_v1.json"))
    task = next(t for t in dataset.tasks if t.task_id == "task-support-ticket")

    def factory(t, m):
        return create_benchmark_mock_provider(t, m)

    _, output_path = await execute_benchmark(
        benchmark_id="bench-failed-artifacts",
        dataset=dataset,
        tasks=[task],
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        settings=build_settings(provider="mock"),
        output_dir=tmp_path / "benchmark_results",
        generation_provider_factory=factory,
        is_mock=True,
        clock=FakeClock(),
        run_id_factory=lambda: "fixed",
    )
    artifacts_dir = (
        output_path / "runs" / "task-support-ticket" / "single_agent" / "1" / "artifacts"
    )
    assert not artifacts_dir.exists() or list(artifacts_dir.glob("**/*")) == []


@pytest.mark.asyncio
async def test_jsonl_one_row_per_trial(tmp_path) -> None:
    dataset = load_benchmark_dataset(Path("benchmarks/datasets/software_delivery_v1.json"))
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")

    def factory(t, m):
        return create_benchmark_mock_provider(t, m)

    _, output_path = await execute_benchmark(
        benchmark_id="bench-jsonl",
        dataset=dataset,
        tasks=[task],
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=2,
        settings=build_settings(provider="mock"),
        output_dir=tmp_path / "benchmark_results",
        generation_provider_factory=factory,
        is_mock=True,
        clock=FakeClock(),
        run_id_factory=lambda: "fixed",
    )
    lines = (output_path / "trials.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    json.loads(lines[0])


@pytest.mark.skipif(not getattr(os, "supports_symlinks", True), reason="symlinks unsupported")
def test_write_trial_reports_rejects_symlink_escape(tmp_path) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    artifacts_dir = trial_dir / "artifacts"
    artifacts_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("unchanged", encoding="utf-8")
    symlink = artifacts_dir / "linked.py"
    symlink.symlink_to(outside)

    bundle = ArtifactBundle(
        files=[GeneratedFile(path="linked.py", content="malicious overwrite\n")]
    )
    state = PipelineState(
        run_id="trial-run",
        brief=sample_brief(),
        architecture=sample_architecture(),
        artifacts=bundle,
    )
    result = TrialExecutionResult(
        trial=BenchmarkTrial(
            benchmark_id="bench-symlink",
            dataset_version="v1",
            task_id="task-url-shortener",
            mode=BenchmarkMode.CONTRACT_WITH_REPAIR,
            repetition=1,
            model_label="mock",
            reviewer_model_label="mock",
            status=TrialStatus.COMPLETED,
            success=True,
        ),
        pipeline_state=state,
        generated_files=["linked.py"],
    )

    with pytest.raises(ArtifactPersistenceError, match="Symlink"):
        write_trial_reports(trial_dir=trial_dir, result=result)

    assert outside.read_text(encoding="utf-8") == "unchanged"

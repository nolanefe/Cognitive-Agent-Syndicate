"""Integration tests for cross-file benchmark report consistency."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cognitive_agent_syndicate.benchmarking.dataset import load_benchmark_dataset
from cognitive_agent_syndicate.benchmarking.display import (
    format_rate_percent,
    format_success_summary,
)
from cognitive_agent_syndicate.benchmarking.mock_fixtures import (
    MOCK_BENCHMARK_DISCLAIMER,
    create_benchmark_mock_provider,
)
from cognitive_agent_syndicate.benchmarking.runner import execute_benchmark
from cognitive_agent_syndicate.benchmarking.schemas import BenchmarkMode
from cognitive_agent_syndicate.cli import app
from cognitive_agent_syndicate.config import build_settings

runner = CliRunner()


class FakeClock:
    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        self._value += 0.01
        return self._value


def _factory(task, mode):
    return create_benchmark_mock_provider(task, mode)


@pytest.mark.asyncio
async def test_structured_report_consistency(tmp_path) -> None:
    dataset = load_benchmark_dataset(Path("benchmarks/datasets/software_delivery_v1.json"))
    benchmark_id = "report-consistency"
    output_dir = tmp_path / "benchmark_results"
    run, output_path = await execute_benchmark(
        benchmark_id=benchmark_id,
        dataset=dataset,
        tasks=list(dataset.tasks),
        modes=list(BenchmarkMode),
        repetitions=1,
        settings=build_settings(provider="mock"),
        output_dir=output_dir,
        generation_provider_factory=_factory,
        is_mock=True,
        clock=FakeClock(),
        run_id_factory=lambda: "fixed-run",
    )
    assert run.summary is not None
    summary = run.summary

    summary_json = json.loads((output_path / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["total_trials"] == summary.total_trials
    assert summary_json["attempted_trials"] == summary.attempted_trials
    assert summary_json["successful_trials"] == summary.successful_trials
    assert summary_json["dataset_label"] == "software_delivery v1"
    assert summary_json["total_observed_provider_calls"] == summary.total_observed_provider_calls

    markdown = (output_path / "summary.md").read_text(encoding="utf-8")
    assert "software_delivery v1" in markdown
    assert "vv1" not in markdown
    assert MOCK_BENCHMARK_DISCLAIMER in markdown

    jsonl_trials = [
        json.loads(line)
        for line in (output_path / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(jsonl_trials) == summary.total_trials

    with (output_path / "results.csv").open(encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == summary.total_trials

    trial_call_sum = sum(int(row["provider_call_count"]) for row in csv_rows)
    assert trial_call_sum == summary.total_observed_provider_calls

    for mode_summary in summary.mode_summaries:
        expected_display = format_success_summary(
            successful=mode_summary.successful_trials,
            attempted=mode_summary.attempted_trial_count,
            rate=mode_summary.success_rate,
        )
        assert format_rate_percent(mode_summary.success_rate) in markdown
        assert f"{mode_summary.successful_trials}/{mode_summary.attempted_trial_count}" in markdown
        assert expected_display.split("(")[0].strip() in expected_display


def test_cli_mixed_mock_run_exits_three(tmp_path) -> None:
    cli_result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--dataset",
            "benchmarks/datasets/software_delivery_v1.json",
            "--modes",
            "single_agent,contract_no_repair,contract_with_repair",
            "--repetitions",
            "1",
            "--provider",
            "mock",
            "--output-dir",
            str(tmp_path / "cli-benchmark-results"),
            "--benchmark-id",
            "report-consistency-cli",
        ],
    )
    assert cli_result.exit_code == 3
    assert "Mock benchmark results validate" in cli_result.stdout
    assert (tmp_path / "cli-benchmark-results" / "report-consistency-cli" / "summary.json").exists()

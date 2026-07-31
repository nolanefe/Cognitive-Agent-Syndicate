"""Regression tests for benchmark rate denominators and display consistency."""

from __future__ import annotations

import csv
import io

from typer.testing import CliRunner

from cognitive_agent_syndicate.benchmarking.display import (
    format_dataset_label,
    format_rate_percent,
    format_success_summary,
)
from cognitive_agent_syndicate.benchmarking.metrics import build_benchmark_summary
from cognitive_agent_syndicate.benchmarking.reporting import (
    render_summary_markdown,
    trial_to_csv_row,
)
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkMode,
    BenchmarkTrial,
    TrialFailureCategory,
    TrialStatus,
)
from cognitive_agent_syndicate.cli import app

runner = CliRunner()


def _trial(**overrides) -> BenchmarkTrial:
    base = {
        "benchmark_id": "denom-test",
        "dataset_version": "v1",
        "task_id": "task-url-shortener",
        "mode": BenchmarkMode.SINGLE_AGENT,
        "repetition": 1,
        "model_label": "mock",
        "reviewer_model_label": "mock",
        "status": TrialStatus.COMPLETED,
        "success": True,
    }
    base.update(overrides)
    return BenchmarkTrial(**base)


def test_success_rate_uses_non_cancelled_denominator() -> None:
    trials = [
        _trial(repetition=1, success=True),
        _trial(repetition=2, success=True),
        _trial(repetition=3, success=True),
        _trial(repetition=4, success=False, failure_category="reviewer_rejected"),
        _trial(
            repetition=5,
            status=TrialStatus.FAILED,
            success=False,
            failure_category=TrialFailureCategory.PROVIDER_CONNECTION,
        ),
        _trial(
            repetition=6,
            status=TrialStatus.FAILED,
            success=False,
            failure_category=TrialFailureCategory.PROVIDER_CONNECTION,
        ),
        _trial(repetition=7, status=TrialStatus.CANCELLED, success=False),
    ]
    summary = build_benchmark_summary(
        benchmark_id="denom-test",
        dataset_name="software_delivery",
        dataset_version="v1",
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=7,
        model_label="mock",
        reviewer_model_label="mock",
        same_model_reviewer=True,
        pricing_configured=False,
        is_mock=True,
        trials=trials,
        task_titles={"task-url-shortener": "URL"},
    )
    mode = summary.mode_summaries[0]
    assert summary.attempted_trials == 6
    assert summary.cancelled_trials == 1
    assert mode.successful_trials == 3
    assert mode.attempted_trial_count == 6
    assert mode.success_rate == 0.5
    assert format_rate_percent(mode.success_rate) == "50.00%"
    assert (
        format_success_summary(
            successful=mode.successful_trials,
            attempted=mode.attempted_trial_count,
            rate=mode.success_rate,
        )
        == "3/6 success (50.00%)"
    )


def test_dataset_label_renders_exact_version() -> None:
    label = format_dataset_label("software_delivery", "v1")
    assert label == "software_delivery v1"
    assert "vv1" not in label


def test_cross_output_rate_consistency() -> None:
    trials = [
        _trial(repetition=1, success=True),
        _trial(repetition=2, success=True),
        _trial(repetition=3, success=True),
        _trial(repetition=4, success=False, failure_category="reviewer_rejected"),
        _trial(
            repetition=5,
            status=TrialStatus.FAILED,
            success=False,
            failure_category=TrialFailureCategory.PROVIDER_CONNECTION,
        ),
        _trial(
            repetition=6,
            status=TrialStatus.FAILED,
            success=False,
            failure_category=TrialFailureCategory.PROVIDER_CONNECTION,
        ),
    ]
    summary = build_benchmark_summary(
        benchmark_id="denom-test",
        dataset_name="software_delivery",
        dataset_version="v1",
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=6,
        model_label="mock",
        reviewer_model_label="mock",
        same_model_reviewer=True,
        pricing_configured=False,
        is_mock=True,
        trials=trials,
        task_titles={"task-url-shortener": "URL"},
    )
    mode = summary.mode_summaries[0]
    summary_json = summary.model_dump(mode="json")
    assert summary_json["attempted_trials"] == 6
    assert summary_json["successful_trials"] == 3
    assert summary_json["mode_summaries"][0]["success_rate"] == 0.5

    markdown = render_summary_markdown(summary)
    assert "software_delivery v1" in markdown
    assert "vv1" not in markdown
    assert "3/6" in markdown
    assert "50.00%" in markdown

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(trial_to_csv_row(trials[0]).keys()))
    writer.writeheader()
    for trial in trials:
        writer.writerow(trial_to_csv_row(trial))
    csv_text = csv_buffer.getvalue()
    assert csv_text.count("completed") >= 1
    assert csv_text.count("failed") >= 1

    plan_result = runner.invoke(
        app,
        [
            "benchmark",
            "plan",
            "--dataset",
            "benchmarks/datasets/software_delivery_v1.json",
            "--modes",
            "single_agent",
            "--repetitions",
            "1",
            "--provider",
            "mock",
        ],
    )
    assert plan_result.exit_code == 0
    assert "Dataset: software_delivery v1" in plan_result.stdout
    assert "vv1" not in plan_result.stdout

    cli_display = format_success_summary(
        successful=mode.successful_trials,
        attempted=mode.attempted_trial_count,
        rate=mode.success_rate,
    )
    assert cli_display == "3/6 success (50.00%)"

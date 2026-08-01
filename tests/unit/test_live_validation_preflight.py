"""Unit tests for live validation preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_agent_syndicate.live_validation.preflight import (
    PreflightError,
    run_live_validation_preflight,
)


def test_missing_model_rejected() -> None:
    with pytest.raises(PreflightError, match="--model"):
        run_live_validation_preflight(
            dataset="benchmarks/datasets/software_delivery_v1.json",
            task_ids="task-url-shortener",
            modes="single_agent",
            repetitions=1,
            model=None,
            reviewer_model=None,
            output_dir="benchmark_results",
            benchmark_id="live-test-id",
            allow_dirty=True,
            confirm_live=True,
        )


def test_confirm_live_required() -> None:
    with pytest.raises(PreflightError, match="confirm-live"):
        run_live_validation_preflight(
            dataset="benchmarks/datasets/software_delivery_v1.json",
            task_ids="task-url-shortener",
            modes="single_agent",
            repetitions=1,
            model="gpt-test",
            reviewer_model=None,
            output_dir="benchmark_results",
            benchmark_id="live-test-id",
            allow_dirty=True,
            confirm_live=False,
        )


def test_bad_dataset_rejected() -> None:
    with pytest.raises(PreflightError):
        run_live_validation_preflight(
            dataset="missing-dataset.json",
            task_ids=None,
            modes="single_agent",
            repetitions=1,
            model="gpt-test",
            reviewer_model=None,
            output_dir="benchmark_results",
            benchmark_id="live-test-id",
            allow_dirty=True,
            confirm_live=True,
        )


def test_invalid_task_rejected() -> None:
    with pytest.raises(PreflightError):
        run_live_validation_preflight(
            dataset="benchmarks/datasets/software_delivery_v1.json",
            task_ids="not-a-task",
            modes="single_agent",
            repetitions=1,
            model="gpt-test",
            reviewer_model=None,
            output_dir="benchmark_results",
            benchmark_id="live-test-id",
            allow_dirty=True,
            confirm_live=True,
        )


def test_unsafe_benchmark_id_rejected() -> None:
    with pytest.raises(PreflightError):
        run_live_validation_preflight(
            dataset="benchmarks/datasets/software_delivery_v1.json",
            task_ids="task-url-shortener",
            modes="single_agent",
            repetitions=1,
            model="gpt-test",
            reviewer_model=None,
            output_dir="benchmark_results",
            benchmark_id="../bad",
            allow_dirty=True,
            confirm_live=True,
        )


def test_existing_output_rejected(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark_results"
    existing = output_dir / "dup-live"
    existing.mkdir(parents=True)
    with pytest.raises(PreflightError, match="already exists"):
        run_live_validation_preflight(
            dataset="benchmarks/datasets/software_delivery_v1.json",
            task_ids="task-url-shortener",
            modes="single_agent",
            repetitions=1,
            model="gpt-test",
            reviewer_model=None,
            output_dir=str(output_dir),
            benchmark_id="dup-live",
            allow_dirty=True,
            confirm_live=True,
        )


def test_repetitions_over_five_rejected() -> None:
    with pytest.raises(PreflightError):
        run_live_validation_preflight(
            dataset="benchmarks/datasets/software_delivery_v1.json",
            task_ids="task-url-shortener",
            modes="single_agent",
            repetitions=6,
            model="gpt-test",
            reviewer_model=None,
            output_dir="benchmark_results",
            benchmark_id="live-test-id",
            allow_dirty=True,
            confirm_live=True,
        )


def test_dirty_tree_refused(monkeypatch) -> None:
    monkeypatch.setattr(
        "cognitive_agent_syndicate.live_validation.preflight.collect_git_metadata",
        lambda: __import__(
            "cognitive_agent_syndicate.live_validation.preflight", fromlist=["GitMetadata"]
        ).GitMetadata(
            available=True,
            commit_sha="abc123",
            branch="main",
            working_tree_clean=False,
        ),
    )
    with pytest.raises(PreflightError, match="dirty"):
        run_live_validation_preflight(
            dataset="benchmarks/datasets/software_delivery_v1.json",
            task_ids="task-url-shortener",
            modes="single_agent",
            repetitions=1,
            model="gpt-test",
            reviewer_model=None,
            output_dir="benchmark_results",
            benchmark_id="live-test-id",
            allow_dirty=False,
            confirm_live=True,
        )


def test_allow_dirty_permits_dirty_tree(monkeypatch) -> None:
    monkeypatch.setattr(
        "cognitive_agent_syndicate.live_validation.preflight.collect_git_metadata",
        lambda: __import__(
            "cognitive_agent_syndicate.live_validation.preflight", fromlist=["GitMetadata"]
        ).GitMetadata(
            available=True,
            commit_sha="abc123",
            branch="main",
            working_tree_clean=False,
        ),
    )
    result = run_live_validation_preflight(
        dataset="benchmarks/datasets/software_delivery_v1.json",
        task_ids="task-url-shortener",
        modes="single_agent",
        repetitions=1,
        model="gpt-test",
        reviewer_model=None,
        output_dir="benchmark_results",
        benchmark_id="live-allow-dirty",
        allow_dirty=True,
        confirm_live=True,
        smoke_only=True,
    )
    assert result.benchmark_id == "live-allow-dirty"

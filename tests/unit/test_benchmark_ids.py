"""Tests for benchmark identifier validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cognitive_agent_syndicate.benchmarking.ids import (
    InvalidBenchmarkIdError,
    validate_benchmark_id,
)
from cognitive_agent_syndicate.benchmarking.reporting import (
    BenchmarkOutputError,
    resolve_benchmark_output_dir,
)
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkMode,
    BenchmarkTrial,
    TrialStatus,
)


@pytest.mark.parametrize(
    "value",
    ["bench-1", "bench.test", "bench_test", "A1"],
)
def test_valid_benchmark_ids(value: str) -> None:
    assert validate_benchmark_id(value) == value


def test_valid_benchmark_id_max_length() -> None:
    benchmark_id = "a" + ("b" * 63)
    assert len(benchmark_id) == 64
    assert validate_benchmark_id(benchmark_id) == benchmark_id


def test_benchmark_id_rejects_over_max_length() -> None:
    benchmark_id = "a" + ("b" * 64)
    assert len(benchmark_id) == 65
    with pytest.raises(InvalidBenchmarkIdError):
        validate_benchmark_id(benchmark_id)


@pytest.mark.parametrize(
    "value",
    ["", " ", "../escape", "a/b", r"a\b", ".", "..", "a/../b", "/abs", "a\x00b"],
)
def test_invalid_benchmark_ids(value: str) -> None:
    with pytest.raises(InvalidBenchmarkIdError):
        validate_benchmark_id(value)


def test_schema_rejects_invalid_benchmark_id() -> None:
    with pytest.raises(ValidationError):
        BenchmarkTrial(
            benchmark_id="../bad",
            dataset_version="v1",
            task_id="task-url-shortener",
            mode=BenchmarkMode.SINGLE_AGENT,
            repetition=1,
            model_label="mock",
            reviewer_model_label="mock",
            status=TrialStatus.COMPLETED,
            success=True,
        )


def test_resolve_output_dir_rejects_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(InvalidBenchmarkIdError):
        resolve_benchmark_output_dir(tmp_path / "benchmark_results", "../evil")

    with pytest.raises(BenchmarkOutputError):
        resolve_benchmark_output_dir(Path("../outside-benchmarks"), "valid-id")

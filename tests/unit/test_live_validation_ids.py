"""Unit tests for live benchmark ID generation."""

from __future__ import annotations

from datetime import datetime

from cognitive_agent_syndicate.benchmarking.ids import validate_benchmark_id
from cognitive_agent_syndicate.live_validation.ids import generate_live_benchmark_id


def test_generate_live_benchmark_id_single_task() -> None:
    benchmark_id = generate_live_benchmark_id(
        ["task-url-shortener"],
        3,
        now=datetime(2026, 8, 1, 13, 45, 30),
    )
    assert benchmark_id == "live-url-shortener-r3-20260801-134530"
    assert validate_benchmark_id(benchmark_id) == benchmark_id


def test_generate_live_benchmark_id_suite() -> None:
    benchmark_id = generate_live_benchmark_id(
        ["task-url-shortener", "task-feature-flag"],
        1,
        now=datetime(2026, 8, 1, 13, 45, 30),
    )
    assert benchmark_id == "live-suite-r1-20260801-134530"
    assert validate_benchmark_id(benchmark_id) == benchmark_id

"""Validate frozen portfolio evidence under benchmarks/results/."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

EVIDENCE_DIR = Path("benchmarks/results/live-suite-r1")

SENSITIVE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"OPENAI_API_KEY"),
    re.compile(r"/Users/"),
    re.compile(r"/home/"),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"AuthenticationError"),
]


@pytest.fixture
def evidence_files() -> list[Path]:
    return sorted(EVIDENCE_DIR.iterdir())


@pytest.fixture
def summary() -> dict:
    return json.loads((EVIDENCE_DIR / "summary.json").read_text())


@pytest.fixture
def methodology() -> dict:
    return json.loads((EVIDENCE_DIR / "methodology.json").read_text())


@pytest.fixture
def csv_rows() -> list[dict[str, str]]:
    with open(EVIDENCE_DIR / "results.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def test_evidence_directory_exists() -> None:
    assert EVIDENCE_DIR.is_dir()
    names = {p.name for p in EVIDENCE_DIR.iterdir()}
    assert names == {"README.md", "summary.json", "results.csv", "methodology.json"}


def test_summary_counts_reconcile(summary: dict, csv_rows: list[dict[str, str]]) -> None:
    assert summary["total_trials"] == 18
    assert summary["completed_trials"] == 18
    assert summary["failed_trials"] == 0
    assert summary["cancelled_trials"] == 0
    assert summary["successful_trials"] == 7
    assert len(csv_rows) == 18

    mode_attempted = sum(m["attempted"] for m in summary["mode_summaries"])
    assert mode_attempted == 18

    mode_successful = sum(m["successful"] for m in summary["mode_summaries"])
    assert mode_successful == 7

    csv_successes = sum(1 for row in csv_rows if row["success"] == "True")
    assert csv_successes == 7


def test_token_totals_reconcile(summary: dict, csv_rows: list[dict[str, str]]) -> None:
    csv_prompt = sum(int(row["prompt_tokens"]) for row in csv_rows)
    csv_completion = sum(int(row["completion_tokens"]) for row in csv_rows)
    csv_total = sum(int(row["total_tokens"]) for row in csv_rows)

    assert csv_prompt == summary["prompt_tokens"] == 119766
    assert csv_completion == summary["completion_tokens"] == 88333
    assert csv_total == summary["total_tokens"] == 208099


def test_repair_bounds(summary: dict, csv_rows: list[dict[str, str]]) -> None:
    assert summary["repair_successes"] <= summary["repair_attempts"]
    assert summary["repair_attempts"] == 4
    assert summary["repair_successes"] == 3

    repair_attempted = sum(1 for row in csv_rows if row["repair_attempted"] == "True")
    repair_succeeded = sum(1 for row in csv_rows if row["repair_succeeded"] == "True")
    assert repair_attempted == 4
    assert repair_succeeded == 3


def test_mode_success_rates(summary: dict) -> None:
    by_mode = {m["mode"]: m for m in summary["mode_summaries"]}
    assert by_mode["single_agent"]["successful"] == 1
    assert by_mode["contract_no_repair"]["successful"] == 1
    assert by_mode["contract_with_repair"]["successful"] == 5


def test_methodology_exploratory_not_significant(methodology: dict) -> None:
    assert methodology["statistical_significance"] is False
    assert methodology["benchmark_type"] == "exploratory_live_benchmark"
    assert methodology["evaluation"]["generated_code_executed"] is False
    caveats_text = " ".join(methodology["caveats"]).lower()
    assert "not statistically significant" in caveats_text


def test_no_sensitive_content_in_evidence(evidence_files: list[Path]) -> None:
    for path in evidence_files:
        content = path.read_text()
        for pattern in SENSITIVE_PATTERNS:
            assert pattern.search(content) is None, f"{pattern.pattern} found in {path.name}"


def test_summary_has_no_local_paths(summary: dict) -> None:
    serialized = json.dumps(summary)
    assert "/Users/" not in serialized
    assert "/home/" not in serialized


def test_csv_deterministic_column_order() -> None:
    with open(EVIDENCE_DIR / "results.csv", newline="") as handle:
        header = handle.readline().strip()
    expected = (
        "task_id,mode,repetition,status,success,reviewer_status,failure_category,"
        "repair_attempted,repair_succeeded,provider_call_count,prompt_tokens,"
        "completion_tokens,total_tokens,provider_latency_ms,wall_clock_ms"
    )
    assert header == expected

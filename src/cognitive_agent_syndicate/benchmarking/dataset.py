"""Benchmark dataset loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkDataset,
    BenchmarkMode,
    BenchmarkTask,
)


class DatasetLoadError(ValueError):
    """Raised when a benchmark dataset cannot be loaded or validated."""


def load_benchmark_dataset(path: Path) -> BenchmarkDataset:
    """Load and validate a benchmark dataset from a JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetLoadError(f"Cannot read dataset file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetLoadError(f"Invalid JSON in dataset file: {path}") from exc

    try:
        return BenchmarkDataset.model_validate(payload)
    except ValidationError as exc:
        raise DatasetLoadError(f"Invalid benchmark dataset: {exc}") from exc


def parse_benchmark_modes(value: str) -> list[BenchmarkMode]:
    """Parse a comma-separated list of benchmark modes."""
    if not value.strip():
        raise ValueError("At least one benchmark mode is required")
    modes: list[BenchmarkMode] = []
    seen: set[BenchmarkMode] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            mode = BenchmarkMode(token)
        except ValueError as exc:
            valid = ", ".join(item.value for item in BenchmarkMode)
            raise ValueError(
                f"Invalid benchmark mode {token!r}. Expected one of: {valid}."
            ) from exc
        if mode not in seen:
            modes.append(mode)
            seen.add(mode)
    if not modes:
        raise ValueError("At least one benchmark mode is required")
    return modes


def filter_dataset_tasks(
    dataset: BenchmarkDataset,
    task_ids: list[str] | None,
) -> list[BenchmarkTask]:
    """Return selected tasks from a dataset."""
    if not task_ids:
        return list(dataset.tasks)
    available = {task.task_id: task for task in dataset.tasks}
    missing = sorted(set(task_ids) - set(available))
    if missing:
        raise ValueError(f"Unknown task IDs: {', '.join(missing)}")
    return [available[task_id] for task_id in task_ids]


def validate_repetitions(value: int, *, live: bool = False) -> int:
    """Validate repetition count within bounded limits."""
    if value < 1 or value > 10:
        raise ValueError("Repetitions must be between 1 and 10")
    if live and value > 5:
        raise ValueError("Live benchmark repetitions must be at most 5")
    return value

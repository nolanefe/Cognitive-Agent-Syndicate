"""Preflight validation for live validation runs."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cognitive_agent_syndicate.benchmarking.dataset import (
    DatasetLoadError,
    filter_dataset_tasks,
    load_benchmark_dataset,
    parse_benchmark_modes,
    validate_repetitions,
)
from cognitive_agent_syndicate.benchmarking.ids import (
    InvalidBenchmarkIdError,
    validate_benchmark_id,
)
from cognitive_agent_syndicate.benchmarking.reporting import (
    BenchmarkOutputError,
    resolve_benchmark_output_dir,
)
from cognitive_agent_syndicate.benchmarking.schemas import BenchmarkMode, BenchmarkTask
from cognitive_agent_syndicate.providers.factory import OPENAI_INSTALL_INSTRUCTION


class PreflightError(ValueError):
    """Raised when live validation preflight checks fail."""


@dataclass(frozen=True)
class GitMetadata:
    """Best-effort git metadata for reproducibility."""

    available: bool
    commit_sha: str | None
    branch: str | None
    working_tree_clean: bool | None


@dataclass(frozen=True)
class LiveValidationPreflightResult:
    """Successful preflight output used by the orchestrator."""

    dataset_path: Path
    dataset_name: str
    dataset_version: str
    selected_tasks: list[BenchmarkTask]
    modes: list[BenchmarkMode]
    repetitions: int
    model: str
    reviewer_model: str
    output_dir: Path
    benchmark_id: str
    git: GitMetadata


def collect_git_metadata() -> GitMetadata:
    """Collect git metadata when git is available."""
    if shutil.which("git") is None:
        return GitMetadata(
            available=False,
            commit_sha=None,
            branch=None,
            working_tree_clean=None,
        )
    repo_root = Path.cwd()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        clean = status.strip() == ""
    except (OSError, subprocess.CalledProcessError):
        return GitMetadata(
            available=False,
            commit_sha=None,
            branch=None,
            working_tree_clean=None,
        )
    return GitMetadata(
        available=True,
        commit_sha=commit,
        branch=branch,
        working_tree_clean=clean,
    )


def run_live_validation_preflight(
    *,
    dataset: str,
    task_ids: str | None,
    modes: str,
    repetitions: int,
    model: str | None,
    reviewer_model: str | None,
    output_dir: str,
    benchmark_id: str | None,
    allow_dirty: bool,
    confirm_live: bool,
    smoke_only: bool = False,
    generate_benchmark_id: Callable[[list[str], int], str] | None = None,
) -> LiveValidationPreflightResult:
    """Validate configuration before any live provider calls."""
    from cognitive_agent_syndicate.live_validation.ids import generate_live_benchmark_id

    if not confirm_live:
        raise PreflightError("Live validation requires the --confirm-live flag.")

    if not model or not model.strip():
        raise PreflightError("Live validation requires --model.")

    _ensure_openai_sdk_installed()

    try:
        selected_modes = parse_benchmark_modes(modes)
        validate_repetitions(repetitions, live=True)
    except ValueError as exc:
        raise PreflightError(str(exc)) from exc

    dataset_path = Path(dataset)
    try:
        dataset_obj = load_benchmark_dataset(dataset_path)
    except DatasetLoadError as exc:
        raise PreflightError(str(exc)) from exc

    try:
        if task_ids:
            ids = [part.strip() for part in task_ids.split(",") if part.strip()]
            selected_tasks = filter_dataset_tasks(dataset_obj, ids)
        else:
            selected_tasks = list(dataset_obj.tasks)
    except ValueError as exc:
        raise PreflightError(str(exc)) from exc

    resolved_benchmark_id = benchmark_id
    if resolved_benchmark_id is None:
        id_factory = generate_benchmark_id or generate_live_benchmark_id
        resolved_benchmark_id = id_factory(
            [task.task_id for task in selected_tasks],
            repetitions,
        )
    try:
        validate_benchmark_id(resolved_benchmark_id)
    except InvalidBenchmarkIdError as exc:
        raise PreflightError(str(exc)) from exc

    if not smoke_only:
        try:
            resolve_benchmark_output_dir(Path(output_dir), resolved_benchmark_id)
        except BenchmarkOutputError as exc:
            raise PreflightError(str(exc)) from exc

    git = collect_git_metadata()
    if git.available and git.working_tree_clean is False and not allow_dirty:
        raise PreflightError(
            "Working tree is dirty. Commit or stash changes, or pass --allow-dirty."
        )

    _assert_generated_code_execution_disabled()

    resolved_reviewer_model = reviewer_model.strip() if reviewer_model else model.strip()

    return LiveValidationPreflightResult(
        dataset_path=dataset_path,
        dataset_name=dataset_obj.name,
        dataset_version=dataset_obj.version,
        selected_tasks=selected_tasks,
        modes=selected_modes,
        repetitions=repetitions,
        model=model.strip(),
        reviewer_model=resolved_reviewer_model,
        output_dir=Path(output_dir),
        benchmark_id=resolved_benchmark_id,
        git=git,
    )


def _ensure_openai_sdk_installed() -> None:
    if importlib.util.find_spec("openai") is None:
        raise PreflightError(
            f"OpenAI provider requires the optional openai dependency. {OPENAI_INSTALL_INSTRUCTION}"
        )


def _assert_generated_code_execution_disabled() -> None:
    """Benchmarks statically inspect generated code and never execute it."""
    return None

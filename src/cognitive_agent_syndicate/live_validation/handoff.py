"""Final handoff rendering and machine-readable live validation output."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from cognitive_agent_syndicate.benchmarking.display import format_success_summary
from cognitive_agent_syndicate.benchmarking.schemas import BenchmarkRun
from cognitive_agent_syndicate.live_validation.preflight import GitMetadata
from cognitive_agent_syndicate.live_validation.smoke import LiveSmokeResult

LIVE_VALIDATION_SCHEMA_VERSION = "1.0"


def _decimal_default(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def render_live_validation_handoff_from_run(
    *,
    run: BenchmarkRun,
    results_path: Path,
    benchmark_exit_status: int,
    estimated_cost_usd: Decimal | None,
) -> str:
    """Render handoff using full benchmark run data."""
    summary = run.summary
    assert summary is not None
    repair_attempts = sum(mode.repair_attempt_count for mode in summary.mode_summaries)
    repair_successes = sum(mode.repair_success_count for mode in summary.mode_summaries)
    prompt_tokens = sum(trial.prompt_tokens for trial in run.trials)
    completion_tokens = sum(trial.completion_tokens for trial in run.trials)
    total_tokens = sum(trial.total_tokens for trial in run.trials)
    wall_clock_ms = sum(trial.wall_clock_duration_ms for trial in run.trials)

    lines = [
        "LIVE VALIDATION COMPLETE",
        f"Benchmark ID: {run.benchmark_id}",
        f"Dataset: {summary.dataset_label}",
        f"Trials: {summary.total_trials}",
        f"Completed: {summary.completed_trials}",
        f"Failed: {summary.failed_trials}",
        f"Successful: {summary.successful_trials}",
        f"Provider calls: {summary.total_observed_provider_calls}",
        f"Prompt tokens: {prompt_tokens}",
        f"Completion tokens: {completion_tokens}",
        f"Total tokens: {total_tokens}",
        f"Wall clock: {wall_clock_ms / 1000.0:.1f}s",
        f"Repair attempts: {repair_attempts}",
        f"Repair successes: {repair_successes}",
        f"Benchmark exit status: {benchmark_exit_status}",
        f"Results: {results_path / 'summary.md'}",
        "",
        "Mode results:",
    ]
    for mode_summary in summary.mode_summaries:
        success_text = format_success_summary(
            successful=mode_summary.successful_trials,
            attempted=mode_summary.attempted_trial_count,
            rate=mode_summary.success_rate,
        )
        lines.append(f"  {mode_summary.mode.value}: {success_text}")

    if estimated_cost_usd is not None:
        lines.extend(["", f"Estimated cost: ${estimated_cost_usd:.4f}"])
    else:
        lines.extend(["", "Estimated cost: unavailable (no pricing configuration)"])

    return "\n".join(lines)


def write_live_validation_json(
    *,
    output_path: Path,
    benchmark_id: str,
    smoke: LiveSmokeResult,
    git: GitMetadata,
    run: BenchmarkRun | None,
    results_dir: Path | None,
    final_status: str,
    benchmark_exit_status: int | None,
) -> Path:
    """Persist machine-readable live validation metadata atomically."""
    summary_payload: dict[str, Any] | None = None
    if run is not None and run.summary is not None:
        summary_payload = {
            "total_trials": run.summary.total_trials,
            "completed_trials": run.summary.completed_trials,
            "failed_trials": run.summary.failed_trials,
            "successful_trials": run.summary.successful_trials,
            "total_observed_provider_calls": run.summary.total_observed_provider_calls,
            "mode_summaries": [
                {
                    "mode": mode.mode.value,
                    "successful_trials": mode.successful_trials,
                    "attempted_trial_count": mode.attempted_trial_count,
                }
                for mode in run.summary.mode_summaries
            ],
        }

    payload = {
        "schema_version": LIVE_VALIDATION_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "smoke": asdict(smoke),
        "git": {
            "available": git.available,
            "commit_sha": git.commit_sha,
            "branch": git.branch,
            "working_tree_clean": git.working_tree_clean,
        },
        "benchmark_summary": summary_payload,
        "output_paths": {
            "results_dir": str(results_dir) if results_dir is not None else None,
            "summary_md": str(results_dir / "summary.md") if results_dir is not None else None,
            "live_validation_json": str(output_path),
        },
        "final_status": final_status,
        "benchmark_exit_status": benchmark_exit_status,
    }
    content = json.dumps(payload, indent=2, default=_decimal_default) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_fd: int | None = None
    temp_path: Path | None = None
    try:
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            text=True,
        )
        temp_path = Path(temp_name)
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            temp_fd = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
        temp_path = None
    except Exception:
        if temp_fd is not None:
            os.close(temp_fd)
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise

    return output_path

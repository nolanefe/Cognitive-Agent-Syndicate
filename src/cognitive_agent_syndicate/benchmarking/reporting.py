"""Benchmark output persistence and reporting."""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

from cognitive_agent_syndicate.benchmarking.display import (
    format_dataset_label,
    format_rate_percent,
    format_success_fraction,
)
from cognitive_agent_syndicate.benchmarking.ids import validate_benchmark_id
from cognitive_agent_syndicate.benchmarking.mock_fixtures import MOCK_BENCHMARK_DISCLAIMER
from cognitive_agent_syndicate.benchmarking.runner import TrialExecutionResult
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkConfigSnapshot,
    BenchmarkDataset,
    BenchmarkMode,
    BenchmarkSummary,
    BenchmarkTrial,
    PricingConfig,
)
from cognitive_agent_syndicate.orchestration.state import PipelineStage
from cognitive_agent_syndicate.paths import normalize_relative_posix_path
from cognitive_agent_syndicate.reporting.report_writer import render_run_report_markdown
from cognitive_agent_syndicate.schemas import RunReport


class BenchmarkOutputError(ValueError):
    """Raised when benchmark output cannot be written safely."""


CSV_COLUMNS = [
    "benchmark_id",
    "task_id",
    "mode",
    "repetition",
    "status",
    "success",
    "reviewer_status",
    "acceptance_criteria_passed",
    "acceptance_criteria_total",
    "required_files_gate_passed",
    "syntax_gate_passed",
    "forbidden_content_gate_passed",
    "repair_attempted",
    "repair_succeeded",
    "provider_call_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "provider_latency_ms",
    "wall_clock_duration_ms",
    "estimated_cost_usd",
    "generated_file_count",
    "failure_category",
    "failure_reason",
]


def resolve_benchmark_output_dir(output_dir: Path, benchmark_id: str) -> Path:
    """Resolve and validate a safe benchmark output directory."""
    safe_id = validate_benchmark_id(benchmark_id)
    if output_dir.is_absolute():
        root = output_dir.resolve()
    else:
        try:
            normalized = normalize_relative_posix_path(str(output_dir).replace("\\", "/"))
        except ValueError as exc:
            raise BenchmarkOutputError(str(exc)) from exc
        if normalized.startswith("..") or "/../" in f"/{normalized}/":
            raise BenchmarkOutputError("Output directory must stay within the workspace")
        root = Path(normalized).resolve()
    target = (root / safe_id).resolve()
    if target.exists():
        raise BenchmarkOutputError(
            f"Benchmark output directory already exists: {target}. Refusing to overwrite."
        )
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise BenchmarkOutputError(
            "Benchmark output must remain under the configured root"
        ) from exc
    return target


def trial_to_csv_row(trial: BenchmarkTrial) -> dict[str, object]:
    """Convert a trial to a deterministic CSV row."""
    cost = trial.estimated_cost.total_cost_usd if trial.estimated_cost else None
    return {
        "benchmark_id": trial.benchmark_id,
        "task_id": trial.task_id,
        "mode": trial.mode.value,
        "repetition": trial.repetition,
        "status": trial.status.value,
        "success": trial.success,
        "reviewer_status": trial.reviewer_status.value if trial.reviewer_status else "",
        "acceptance_criteria_passed": trial.acceptance_criteria_passed,
        "acceptance_criteria_total": trial.acceptance_criteria_total,
        "required_files_gate_passed": trial.required_files_gate_passed,
        "syntax_gate_passed": trial.syntax_gate_passed,
        "forbidden_content_gate_passed": trial.forbidden_content_gate_passed,
        "repair_attempted": trial.repair_attempted,
        "repair_succeeded": trial.repair_succeeded,
        "provider_call_count": trial.provider_call_count,
        "prompt_tokens": trial.prompt_tokens,
        "completion_tokens": trial.completion_tokens,
        "total_tokens": trial.total_tokens,
        "provider_latency_ms": trial.provider_latency_ms,
        "wall_clock_duration_ms": trial.wall_clock_duration_ms,
        "estimated_cost_usd": str(cost) if cost is not None else "",
        "generated_file_count": trial.generated_file_count,
        "failure_category": trial.failure_category.value if trial.failure_category else "",
        "failure_reason": trial.failure_reason or "",
    }


def render_summary_markdown(summary: BenchmarkSummary) -> str:
    """Render benchmark summary as Markdown."""
    mode_header = (
        "| Mode | Attempted | Success | Success rate | Repair attempts | "
        "Repair success rate | Observed calls |"
    )
    mode_sep = "| --- | --- | --- | --- | --- | --- | --- |"
    mode_rows = [mode_header, mode_sep]
    for mode_summary in summary.mode_summaries:
        success_rate = format_rate_percent(mode_summary.success_rate)
        repair_attempt_rate = format_rate_percent(mode_summary.repair_attempt_rate)
        repair_success_rate = format_rate_percent(mode_summary.repair_success_rate)
        success_fraction = format_success_fraction(
            successful=mode_summary.successful_trials,
            attempted=mode_summary.attempted_trial_count,
        )
        mode_rows.append(
            f"| {mode_summary.mode.value} | {mode_summary.attempted_trial_count} | "
            f"{success_fraction} | {success_rate} | "
            f"{mode_summary.repair_attempt_count} ({repair_attempt_rate}) | "
            f"{mode_summary.repair_success_count} ({repair_success_rate}) | "
            f"{mode_summary.total_observed_provider_calls} |"
        )

    failure_rows = ["| Category | Count |", "| --- | --- |"]
    all_failures: dict[str, int] = {}
    for mode_summary in summary.mode_summaries:
        for category, count in mode_summary.failure_category_counts.items():
            all_failures[category] = all_failures.get(category, 0) + count
    for category in sorted(all_failures):
        failure_rows.append(f"| {category} | {all_failures[category]} |")

    task_sections: list[str] = []
    for task_summary in summary.task_summaries:
        lines = [f"### {task_summary.title} (`{task_summary.task_id}`)", ""]
        lines.extend(["| Mode | Success rate | Repair attempt rate |", "| --- | --- | --- |"])
        for mode_summary in task_summary.mode_summaries:
            rate = format_rate_percent(mode_summary.success_rate)
            repair_rate = format_rate_percent(mode_summary.repair_attempt_rate)
            success_fraction = format_success_fraction(
                successful=mode_summary.successful_trials,
                attempted=mode_summary.attempted_trial_count,
            )
            lines.append(
                f"| {mode_summary.mode.value} | {success_fraction} ({rate}) | "
                f"{mode_summary.repair_attempt_count} ({repair_rate}) |"
            )
        task_sections.extend(lines + [""])

    sections = [
        "# Benchmark Summary",
        "",
        "## Purpose",
        "",
        "Compare single-agent baseline delivery against contract-driven pipeline modes "
        "with and without bounded repair.",
        "",
        "## Configuration",
        "",
        f"- Benchmark ID: `{summary.benchmark_id}`",
        f"- Dataset: {summary.dataset_label}",
        f"- Modes: {', '.join(mode.value for mode in summary.modes)}",
        f"- Repetitions: {summary.repetitions}",
        f"- Generation model: {summary.model_label}",
        f"- Reviewer model: {summary.reviewer_model_label}",
        f"- Same model reviewer: {'yes' if summary.same_model_reviewer else 'no'}",
        f"- Pricing configured: {'yes' if summary.pricing_configured else 'no'}",
        f"- Mock execution: {'yes' if summary.is_mock else 'no'}",
        "",
        "## Trial counts",
        "",
        f"- Total: {summary.total_trials}",
        f"- Attempted (non-cancelled): {summary.attempted_trials}",
        f"- Completed: {summary.completed_trials}",
        f"- Failed: {summary.failed_trials}",
        f"- Cancelled: {summary.cancelled_trials}",
        f"- Successful: {summary.successful_trials}",
        f"- Observed provider calls: {summary.total_observed_provider_calls}",
        f"- Rate rounding: {summary.rate_rounding_note}",
        "",
        "## Mode comparison",
        "",
        *mode_rows,
        "",
        "## Per-task comparison",
        "",
        *task_sections,
        "## Failure categories",
        "",
        *failure_rows,
        "",
        "## Methodology",
        "",
        "- All modes share the same task definitions, reviewer policy, gates, and model settings.",
        "- Single-agent baseline uses one generation call plus review.",
        "- Contract modes use architect, implementer, and reviewer stages.",
        "- Generated code is never executed during benchmark runs.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in summary.limitations],
        "",
    ]
    if summary.is_mock:
        sections.extend(
            [
                "## Mock warning",
                "",
                f"> {MOCK_BENCHMARK_DISCLAIMER}",
                "",
            ]
        )
    return "\n".join(sections)


def write_trial_reports(
    *,
    trial_dir: Path,
    result: TrialExecutionResult,
) -> str | None:
    """Write per-trial run reports and optional artifacts."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    relative = None

    if result.run_report_json is not None:
        report = RunReport.model_validate(result.run_report_json)
        json_path = trial_dir / "run-report.json"
        md_path = trial_dir / "run-report.md"
        json_path.write_text(
            json.dumps(result.run_report_json, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(
            render_run_report_markdown(report, current_stage=PipelineStage.COMPLETED),
            encoding="utf-8",
        )
        relative = str(trial_dir / "run-report.json")

    if result.generated_files and result.pipeline_state is not None:
        artifacts_dir = trial_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        bundle = result.pipeline_state.final_artifacts or result.pipeline_state.artifacts
        if bundle is not None:
            for generated_file in bundle.files:
                target = artifacts_dir / generated_file.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(generated_file.content, encoding="utf-8")

    return relative


def persist_benchmark_output(
    *,
    output_root: Path,
    benchmark_id: str,
    config: BenchmarkConfigSnapshot,
    trials: list[BenchmarkTrial],
    summary: BenchmarkSummary,
    trial_results: dict[tuple[str, BenchmarkMode, int], TrialExecutionResult],
) -> Path:
    """Atomically persist benchmark outputs."""
    staging = output_root.parent / f".{output_root.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        (staging / "benchmark-config.json").write_text(
            json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        jsonl_path = staging / "trials.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for trial in trials:
                handle.write(json.dumps(trial.model_dump(mode="json"), sort_keys=True) + "\n")

        (staging / "summary.json").write_text(
            json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for trial in trials:
            writer.writerow(trial_to_csv_row(trial))
        (staging / "results.csv").write_text(buffer.getvalue(), encoding="utf-8")

        failures = [
            trial.model_dump(mode="json")
            for trial in trials
            if not trial.success and trial.status.value == "completed"
        ]
        failures.extend(
            trial.model_dump(mode="json") for trial in trials if trial.status.value == "failed"
        )
        (staging / "failures.json").write_text(
            json.dumps(failures, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        runs_dir = staging / "runs"
        runs_dir.mkdir()
        updated_trials: list[BenchmarkTrial] = []
        for trial in trials:
            key = (trial.task_id, trial.mode, trial.repetition)
            result = trial_results.get(key)
            if result is None:
                updated_trials.append(trial)
                continue
            trial_dir = runs_dir / trial.task_id / trial.mode.value / str(trial.repetition)
            report_path = write_trial_reports(trial_dir=trial_dir, result=result)
            if report_path:
                updated_trials.append(trial.model_copy(update={"run_report_path": report_path}))
            else:
                updated_trials.append(trial)

        staging.rename(output_root)
        return output_root
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def build_config_snapshot(
    *,
    benchmark_id: str,
    dataset: BenchmarkDataset,
    task_ids: list[str],
    modes: list[BenchmarkMode],
    repetitions: int,
    model_label: str,
    reviewer_model_label: str,
    reviewer_provider_label: str,
    generation_provider_label: str,
    pricing: PricingConfig | None,
    is_mock: bool,
    temperature: float,
) -> BenchmarkConfigSnapshot:
    """Build persisted benchmark configuration."""
    limitations = [
        "Generated code is statically inspected but never executed.",
    ]
    if is_mock:
        limitations.append(MOCK_BENCHMARK_DISCLAIMER)
    return BenchmarkConfigSnapshot(
        benchmark_id=benchmark_id,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_label=format_dataset_label(dataset.name, dataset.version),
        task_ids=task_ids,
        modes=modes,
        repetitions=repetitions,
        model_label=model_label,
        reviewer_model_label=reviewer_model_label,
        reviewer_provider_label=reviewer_provider_label,
        generation_provider_label=generation_provider_label,
        pricing=pricing,
        is_mock=is_mock,
        temperature=temperature,
        limitations=limitations,
    )

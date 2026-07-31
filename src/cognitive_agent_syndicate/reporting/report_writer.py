"""Run report generation for pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

from cognitive_agent_syndicate.orchestration.state import PipelineState
from cognitive_agent_syndicate.schemas import RunReport

REPORT_LIMITATIONS = [
    "Generated code was not executed during this pipeline run.",
    "Gate evaluation is deterministic and does not run tests or scripts.",
]


def build_run_report(state: PipelineState, generated_files: list[str]) -> RunReport:
    reviewer_status = state.review.status if state.review is not None else None
    manifest = sorted(generated_files) if state.success else []
    return RunReport(
        run_id=state.run_id,
        brief_title=state.brief.title,
        gates=state.gate_results,
        usage=state.usage,
        success=state.success,
        artifact_count=len(manifest),
        stages_completed=[stage.value for stage in state.stages_completed],
        reviewer_status=reviewer_status,
        failure_reason=state.failure_reason,
        generated_files=manifest,
        limitations=list(REPORT_LIMITATIONS),
    )


def write_run_reports(*, run_dir: Path, state: PipelineState, generated_files: list[str]) -> None:
    report = build_run_report(state, generated_files)
    json_path = run_dir / "run-report.json"
    markdown_path = run_dir / "run-report.md"

    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("Run report files already exist.")

    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_run_report_markdown(report, state), encoding="utf-8")


def render_run_report_markdown(report: RunReport, state: PipelineState) -> str:
    reviewer_decision = report.reviewer_status.value if report.reviewer_status else "n/a"
    gate_lines = [
        f"- **{gate.gate_name}**: {gate.status.value} — {gate.message}" for gate in report.gates
    ]
    file_lines = [f"- `{path}`" for path in report.generated_files]
    limitation_lines = [f"- {item}" for item in report.limitations]
    stages = ", ".join(report.stages_completed) if report.stages_completed else "none"
    outcome = "success" if report.success else "failure"

    sections = [
        "# Pipeline Run Report",
        "",
        f"- **Run ID**: `{report.run_id}`",
        f"- **Brief title**: {report.brief_title}",
        f"- **Stages completed**: {stages}",
        f"- **Current stage**: {state.stage.value}",
        f"- **Reviewer decision**: {reviewer_decision}",
        f"- **Outcome**: {outcome}",
    ]

    if report.failure_reason:
        sections.extend(["", f"- **Failure reason**: {report.failure_reason}"])

    sections.extend(
        [
            "",
            "## Deterministic gate results",
            "",
            *(gate_lines or ["- none"]),
            "",
            "## Generated file manifest",
            "",
            *(file_lines or ["- none"]),
            "",
            "## Usage totals",
            "",
            f"- Prompt tokens: {report.usage.prompt_tokens}",
            f"- Completion tokens: {report.usage.completion_tokens}",
            f"- Total tokens: {report.usage.total_tokens}",
            f"- Latency (ms): {report.usage.latency_ms}",
            "",
            "## Limitations",
            "",
            *limitation_lines,
            "",
        ]
    )
    return "\n".join(sections)

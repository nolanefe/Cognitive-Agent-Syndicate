"""Command-line interface for the contract-driven pipeline."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from cognitive_agent_syndicate.agents.architect import ArchitectAgent
from cognitive_agent_syndicate.agents.implementer import ImplementerAgent
from cognitive_agent_syndicate.agents.reviewer import ReviewerAgent
from cognitive_agent_syndicate.config import build_settings
from cognitive_agent_syndicate.demo import (
    MockScenario,
    create_demo_provider,
    is_url_shortener_demo_brief,
    parse_mock_scenario,
)
from cognitive_agent_syndicate.orchestration.pipeline import ContractDrivenPipeline
from cognitive_agent_syndicate.orchestration.state import PipelineState
from cognitive_agent_syndicate.schemas import SystemBrief

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.callback()
def cli_root() -> None:
    """Contract-driven multi-agent software delivery pipeline."""


@app.command(name="run")
def run_pipeline(
    brief_path: str = typer.Argument(..., help="Path to a SystemBrief JSON file."),
    artifact_dir: str | None = typer.Option(
        None,
        "--artifact-dir",
        help="Override the artifact output directory.",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Run with built-in deterministic mock responses (no API key required).",
    ),
    mock_scenario: str = typer.Option(
        MockScenario.SUCCESS.value,
        "--mock-scenario",
        help="Deterministic mock scenario: success, repair-success, or repair-failure.",
    ),
    max_repair_attempts: int | None = typer.Option(
        None,
        "--max-repair-attempts",
        min=0,
        max=1,
        help="Override max repair attempts for this invocation (0 or 1).",
    ),
) -> None:
    """Run the contract-driven architect → implementer → reviewer pipeline."""
    if not mock:
        console.print("[red]Stage 3 supports explicit --mock mode only.[/red]")
        raise typer.Exit(code=1)

    try:
        brief = _load_brief(Path(brief_path))
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        console.print(f"[red]Invalid brief file:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        scenario = parse_mock_scenario(mock_scenario)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if max_repair_attempts is None:
        repair_attempts = 1
    else:
        repair_attempts = max_repair_attempts

    overrides: dict[str, object] = {
        "provider": "mock",
        "max_repair_attempts": repair_attempts,
    }
    if artifact_dir is not None:
        overrides["artifact_output_dir"] = artifact_dir
    settings = build_settings(**overrides)

    if not is_url_shortener_demo_brief(brief):
        console.print("[red]Mock mode currently supports the URL Shortener demo brief only.[/red]")
        raise typer.Exit(code=1)

    provider = create_demo_provider(scenario=scenario)
    pipeline = ContractDrivenPipeline(
        architect=ArchitectAgent(provider),
        implementer=ImplementerAgent(provider),
        reviewer=ReviewerAgent(provider),
        settings=settings,
    )

    allowed_technologies = ["python", "pydantic"]
    permitted_paths = ["pyproject.toml", "src", "tests"]
    implementation_constraints = [
        "Keep generated files small and safe.",
        "Do not include secrets or network calls.",
    ]
    required_project_files = ["pyproject.toml"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running mock pipeline...", total=None)
        state: PipelineState = asyncio.run(
            pipeline.run(
                brief,
                allowed_technologies=allowed_technologies,
                permitted_paths=permitted_paths,
                implementation_constraints=implementation_constraints,
                required_project_files=required_project_files,
            )
        )
        progress.update(task, description="Pipeline finished.")

    if state.success:
        console.print("[green]Pipeline succeeded.[/green]")
        if state.repair_attempted:
            console.print("[green]Repair attempt succeeded.[/green]")
        if state.artifact_directory:
            console.print(f"Artifacts written to: {state.artifact_directory}")
        raise typer.Exit(code=0)

    console.print("[red]Pipeline failed.[/red]")
    if state.failure_reason:
        console.print(f"Reason: {state.failure_reason}")
    if state.artifact_directory:
        console.print(f"Failure report written to: {state.artifact_directory}")
    raise typer.Exit(code=1)


def _load_brief(path: Path) -> SystemBrief:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SystemBrief.model_validate(payload)


def main() -> None:
    try:
        app()
    except typer.Exit as exc:
        sys.exit(exc.exit_code)


if __name__ == "__main__":
    main()

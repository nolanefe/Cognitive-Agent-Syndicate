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
from cognitive_agent_syndicate.benchmark_cli import benchmark_app
from cognitive_agent_syndicate.config import ProviderName, Settings, build_settings
from cognitive_agent_syndicate.demo import (
    MockScenario,
    create_demo_provider,
    is_url_shortener_demo_brief,
    parse_mock_scenario,
)
from cognitive_agent_syndicate.orchestration.pipeline import ContractDrivenPipeline
from cognitive_agent_syndicate.orchestration.state import PipelineState
from cognitive_agent_syndicate.providers.base import ModelProvider
from cognitive_agent_syndicate.providers.errors import ProviderConfigurationError
from cognitive_agent_syndicate.providers.factory import (
    create_model_provider,
    validate_provider_configuration,
)
from cognitive_agent_syndicate.schemas import SystemBrief

app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(benchmark_app, name="benchmark")
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
    provider: str = typer.Option(
        "mock",
        "--provider",
        help="Model provider: mock or openai.",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Backward-compatible alias for --provider mock.",
    ),
    mock_scenario: str = typer.Option(
        MockScenario.SUCCESS.value,
        "--mock-scenario",
        help="Deterministic mock scenario: success, repair-success, or repair-failure.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model name (required for --provider openai).",
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
    provider_value = provider.strip().lower()
    if mock and provider_value == ProviderName.OPENAI.value:
        console.print("[red]--mock cannot be combined with --provider openai.[/red]")
        raise typer.Exit(code=1)

    try:
        selected_provider = ProviderName.MOCK if mock else _parse_provider_name(provider_value)
    except ProviderConfigurationError as exc:
        _print_cli_error(_safe_cli_error(exc))
        raise typer.Exit(code=1) from exc

    if selected_provider != ProviderName.MOCK and mock_scenario != MockScenario.SUCCESS.value:
        console.print("[red]--mock-scenario is valid only with the mock provider.[/red]")
        raise typer.Exit(code=1)

    if selected_provider == ProviderName.OPENAI and not model:
        console.print("[red]OpenAI provider requires --model.[/red]")
        raise typer.Exit(code=1)

    try:
        brief = _load_brief(Path(brief_path))
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        _print_cli_error(f"Invalid brief file: {_safe_cli_error(exc)}")
        raise typer.Exit(code=1) from exc

    scenario: MockScenario | None = None
    if selected_provider == ProviderName.MOCK:
        try:
            scenario = parse_mock_scenario(mock_scenario)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

        if not is_url_shortener_demo_brief(brief):
            console.print(
                "[red]Mock mode currently supports the URL Shortener demo brief only.[/red]"
            )
            raise typer.Exit(code=1)

    repair_attempts = 1 if max_repair_attempts is None else max_repair_attempts

    overrides: dict[str, object] = {
        "provider": selected_provider.value,
        "max_repair_attempts": repair_attempts,
    }
    if artifact_dir is not None:
        overrides["artifact_output_dir"] = artifact_dir
    if model is not None:
        overrides["model"] = model.strip()

    try:
        settings = build_settings(**overrides)
        validate_provider_configuration(settings)
    except (ValidationError, ProviderConfigurationError) as exc:
        _print_cli_error(_safe_cli_error(exc))
        raise typer.Exit(code=1) from exc

    try:
        pipeline_provider = _build_provider(selected_provider, scenario=scenario, settings=settings)
    except ProviderConfigurationError as exc:
        _print_cli_error(_safe_cli_error(exc))
        raise typer.Exit(code=1) from exc

    pipeline = ContractDrivenPipeline(
        architect=ArchitectAgent(pipeline_provider),
        implementer=ImplementerAgent(pipeline_provider),
        reviewer=ReviewerAgent(pipeline_provider),
        settings=settings,
    )

    allowed_technologies = ["python", "pydantic"]
    permitted_paths = ["pyproject.toml", "src", "tests"]
    implementation_constraints = [
        "Keep generated files small and safe.",
        "Do not include secrets or network calls.",
    ]
    required_project_files = ["pyproject.toml"]

    progress_label = (
        "Running mock pipeline..."
        if selected_provider == ProviderName.MOCK
        else "Running OpenAI pipeline..."
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(progress_label, total=None)
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


def _parse_provider_name(value: str) -> ProviderName:
    try:
        return ProviderName(value)
    except ValueError as exc:
        raise ProviderConfigurationError(
            f"Invalid provider {value!r}. Expected mock or openai."
        ) from exc


def _build_provider(
    selected_provider: ProviderName,
    *,
    scenario: MockScenario | None,
    settings: Settings,
) -> ModelProvider:
    if selected_provider == ProviderName.MOCK:
        assert scenario is not None
        return create_demo_provider(scenario=scenario)
    return create_model_provider(settings)


def _load_brief(path: Path) -> SystemBrief:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SystemBrief.model_validate(payload)


def _print_cli_error(message: str) -> None:
    escaped = message.replace("[", "\\[")
    console.print(f"[red]{escaped}[/red]")


def _safe_cli_error(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ()))
        message = str(first.get("msg", "Invalid configuration."))
        if message.startswith("Value error, "):
            message = message.removeprefix("Value error, ")
        message = " ".join(message.split())
        if location:
            return f"Invalid configuration ({location}): {message}"
        return f"Invalid configuration: {message}"
    if isinstance(exc, ProviderConfigurationError):
        return " ".join(str(exc).split())
    message = " ".join(str(exc).strip().split())
    return message or exc.__class__.__name__


def main() -> None:
    try:
        app()
    except typer.Exit as exc:
        sys.exit(exc.exit_code)


if __name__ == "__main__":
    main()

"""Benchmark CLI commands."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console

from cognitive_agent_syndicate.benchmarking.dataset import (
    DatasetLoadError,
    filter_dataset_tasks,
    load_benchmark_dataset,
    parse_benchmark_modes,
    validate_repetitions,
)
from cognitive_agent_syndicate.benchmarking.display import format_success_summary
from cognitive_agent_syndicate.benchmarking.exit_codes import (
    EXIT_COMPLETED_WITH_FAILURES,
    EXIT_FATAL,
    EXIT_SUCCESS,
)
from cognitive_agent_syndicate.benchmarking.ids import (
    InvalidBenchmarkIdError,
    validate_benchmark_id,
)
from cognitive_agent_syndicate.benchmarking.mock_fixtures import (
    MOCK_BENCHMARK_DISCLAIMER,
    create_benchmark_mock_provider,
)
from cognitive_agent_syndicate.benchmarking.planning import (
    build_benchmark_plan,
    render_benchmark_plan,
)
from cognitive_agent_syndicate.benchmarking.pricing import PricingLoadError, load_pricing_config
from cognitive_agent_syndicate.benchmarking.reporting import BenchmarkOutputError
from cognitive_agent_syndicate.benchmarking.runner import execute_benchmark
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkDataset,
    BenchmarkMode,
    BenchmarkTask,
    PricingConfig,
)
from cognitive_agent_syndicate.config import ProviderName, Settings, build_settings
from cognitive_agent_syndicate.providers.base import ModelProvider
from cognitive_agent_syndicate.providers.errors import ProviderConfigurationError
from cognitive_agent_syndicate.providers.factory import (
    create_model_provider,
    validate_provider_configuration,
)

benchmark_app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@benchmark_app.callback()
def benchmark_root() -> None:
    """Reproducible benchmark commands for pipeline comparison."""


@benchmark_app.command(name="plan")
def benchmark_plan(
    dataset: str = typer.Option(
        "benchmarks/datasets/software_delivery_v1.json",
        "--dataset",
        help="Path to benchmark dataset JSON.",
    ),
    modes: str = typer.Option(
        "single_agent,contract_no_repair,contract_with_repair",
        "--modes",
        help="Comma-separated benchmark modes.",
    ),
    repetitions: int = typer.Option(1, "--repetitions", min=1, max=10),
    provider: str = typer.Option("mock", "--provider", help="Generation provider: mock or openai."),
    reviewer_provider: str | None = typer.Option(
        None,
        "--reviewer-provider",
        help="Optional reviewer provider (defaults to generation provider).",
    ),
    model: str | None = typer.Option(None, "--model", help="Model name (required for openai)."),
    reviewer_model: str | None = typer.Option(
        None,
        "--reviewer-model",
        help="Optional reviewer model (defaults to generation model).",
    ),
    task_ids: str | None = typer.Option(
        None,
        "--task-ids",
        help="Comma-separated task IDs to include.",
    ),
    pricing_file: str | None = typer.Option(
        None,
        "--pricing-file",
        help="Optional pricing JSON for cost estimation.",
    ),
) -> None:
    """Print a dry-run benchmark plan without provider calls."""
    try:
        selected_modes = parse_benchmark_modes(modes)
        selected_provider = _parse_provider(provider)
        selected_reviewer_provider = _parse_provider(reviewer_provider or provider)
        validate_repetitions(repetitions, live=selected_provider == ProviderName.OPENAI)
        dataset_obj = load_benchmark_dataset(Path(dataset))
        selected_tasks = _select_tasks(dataset_obj, task_ids)
        pricing = _load_optional_pricing(pricing_file)
        settings = _build_benchmark_settings(
            provider=selected_provider,
            model=model,
            live=False,
        )
    except (DatasetLoadError, ValueError, ProviderConfigurationError, PricingLoadError) as exc:
        _print_error(str(exc))
        raise typer.Exit(code=EXIT_FATAL) from exc

    plan = build_benchmark_plan(
        dataset_name=dataset_obj.name,
        dataset_version=dataset_obj.version,
        task_ids=[task.task_id for task in selected_tasks],
        modes=selected_modes,
        repetitions=repetitions,
        generation_provider=selected_provider.value,
        generation_model=settings.model,
        reviewer_provider=selected_reviewer_provider.value,
        reviewer_model=reviewer_model or settings.model,
        pricing_configured=pricing is not None,
        is_mock=selected_provider == ProviderName.MOCK,
        is_live=False,
    )
    console.print(render_benchmark_plan(plan))


@benchmark_app.command(name="run")
def benchmark_run(
    dataset: str = typer.Option(
        "benchmarks/datasets/software_delivery_v1.json",
        "--dataset",
        help="Path to benchmark dataset JSON.",
    ),
    modes: str = typer.Option(
        "single_agent,contract_no_repair,contract_with_repair",
        "--modes",
        help="Comma-separated benchmark modes.",
    ),
    repetitions: int = typer.Option(1, "--repetitions", min=1, max=10),
    provider: str = typer.Option("mock", "--provider", help="Generation provider: mock or openai."),
    reviewer_provider: str | None = typer.Option(
        None,
        "--reviewer-provider",
        help="Optional reviewer provider (defaults to generation provider).",
    ),
    model: str | None = typer.Option(None, "--model", help="Model name (required for openai)."),
    reviewer_model: str | None = typer.Option(
        None,
        "--reviewer-model",
        help="Optional reviewer model (defaults to generation model).",
    ),
    output_dir: str = typer.Option(
        "benchmark_results",
        "--output-dir",
        help="Relative output directory for benchmark artifacts.",
    ),
    task_ids: str | None = typer.Option(
        None,
        "--task-ids",
        help="Comma-separated task IDs to include.",
    ),
    pricing_file: str | None = typer.Option(
        None,
        "--pricing-file",
        help="Optional pricing JSON for cost estimation.",
    ),
    benchmark_id: str | None = typer.Option(
        None,
        "--benchmark-id",
        help="Optional benchmark identifier.",
    ),
    confirm_live: bool = typer.Option(
        False,
        "--confirm-live",
        help="Required confirmation flag for live OpenAI benchmarks.",
    ),
) -> None:
    """Run an offline or live benchmark."""
    try:
        selected_modes = parse_benchmark_modes(modes)
        selected_provider = _parse_provider(provider)
        selected_reviewer_provider = _parse_provider(reviewer_provider or provider)
        is_live = selected_provider == ProviderName.OPENAI
        validate_repetitions(repetitions, live=is_live)
        dataset_obj = load_benchmark_dataset(Path(dataset))
        selected_tasks = _select_tasks(dataset_obj, task_ids)
        pricing = _load_optional_pricing(pricing_file)
    except (DatasetLoadError, ValueError, PricingLoadError) as exc:
        _print_error(str(exc))
        raise typer.Exit(code=EXIT_FATAL) from exc

    if benchmark_id is not None:
        try:
            validate_benchmark_id(benchmark_id)
        except InvalidBenchmarkIdError as exc:
            _print_error(str(exc))
            raise typer.Exit(code=EXIT_FATAL) from exc

    if is_live:
        try:
            _validate_live_benchmark(confirm_live=confirm_live, model=model)
        except ProviderConfigurationError as exc:
            _print_error(str(exc))
            raise typer.Exit(code=EXIT_FATAL) from exc
        console.print("[yellow]Live benchmark selected. API usage may incur cost.[/yellow]")
        plan = build_benchmark_plan(
            dataset_name=dataset_obj.name,
            dataset_version=dataset_obj.version,
            task_ids=[task.task_id for task in selected_tasks],
            modes=selected_modes,
            repetitions=repetitions,
            generation_provider=selected_provider.value,
            generation_model=model or "",
            reviewer_provider=selected_reviewer_provider.value,
            reviewer_model=reviewer_model or model or "",
            pricing_configured=pricing is not None,
            is_mock=False,
            is_live=True,
        )
        console.print(render_benchmark_plan(plan))

    try:
        settings = _build_benchmark_settings(
            provider=selected_provider,
            model=model,
            live=is_live,
        )
        validate_provider_configuration(settings)
    except (ValidationError, ProviderConfigurationError) as exc:
        _print_error(str(exc))
        raise typer.Exit(code=EXIT_FATAL) from exc

    resolved_benchmark_id = benchmark_id or uuid.uuid4().hex[:12]
    is_mock = selected_provider == ProviderName.MOCK

    def generation_factory(task: BenchmarkTask, mode: BenchmarkMode) -> ModelProvider:
        if is_mock:
            return create_benchmark_mock_provider(task, mode)
        return create_model_provider(settings)

    def reviewer_factory(task: BenchmarkTask, mode: BenchmarkMode) -> ModelProvider:
        if is_mock:
            return create_benchmark_mock_provider(task, mode)
        if reviewer_provider or reviewer_model:
            reviewer_settings = build_settings(
                provider=selected_reviewer_provider.value,
                model=reviewer_model or settings.model,
            )
            validate_provider_configuration(reviewer_settings)
            return create_model_provider(reviewer_settings)
        return create_model_provider(settings)

    try:
        run, final_path = asyncio.run(
            execute_benchmark(
                benchmark_id=resolved_benchmark_id,
                dataset=dataset_obj,
                tasks=selected_tasks,
                modes=selected_modes,
                repetitions=repetitions,
                settings=settings,
                output_dir=Path(output_dir),
                generation_provider_factory=generation_factory,
                reviewer_provider_factory=reviewer_factory,
                reviewer_model_label=reviewer_model or settings.model,
                reviewer_provider_label=selected_reviewer_provider.value,
                pricing=pricing,
                is_mock=is_mock,
            )
        )
    except BenchmarkOutputError as exc:
        _print_error(str(exc))
        raise typer.Exit(code=EXIT_FATAL) from exc
    except InvalidBenchmarkIdError as exc:
        _print_error(str(exc))
        raise typer.Exit(code=EXIT_FATAL) from exc

    summary = run.summary
    assert summary is not None
    console.print(f"Benchmark written to: {final_path}")
    console.print(
        f"Trials: {summary.total_trials} total, "
        f"{summary.completed_trials} completed, "
        f"{summary.failed_trials} failed, "
        f"{summary.cancelled_trials} cancelled"
    )
    for mode_summary in summary.mode_summaries:
        console.print(
            f"  {mode_summary.mode.value}: "
            f"{
                format_success_summary(
                    successful=mode_summary.successful_trials,
                    attempted=mode_summary.attempted_trial_count,
                    rate=mode_summary.success_rate,
                )
            }"
        )
    if is_mock:
        console.print(f"[yellow]{MOCK_BENCHMARK_DISCLAIMER}[/yellow]")

    if summary.attempted_trials > 0 and summary.successful_trials < summary.attempted_trials:
        raise typer.Exit(code=EXIT_COMPLETED_WITH_FAILURES)
    raise typer.Exit(code=EXIT_SUCCESS)


def _parse_provider(value: str) -> ProviderName:
    token = value.strip().lower()
    try:
        return ProviderName(token)
    except ValueError as exc:
        raise ProviderConfigurationError(
            f"Invalid provider {value!r}. Expected mock or openai."
        ) from exc


def _select_tasks(dataset_obj: BenchmarkDataset, task_ids: str | None) -> list[BenchmarkTask]:
    if not task_ids:
        return list(dataset_obj.tasks)
    ids = [part.strip() for part in task_ids.split(",") if part.strip()]
    return filter_dataset_tasks(dataset_obj, ids)


def _load_optional_pricing(path: str | None) -> PricingConfig | None:
    if path is None:
        return None
    return load_pricing_config(Path(path))


def _build_benchmark_settings(
    *,
    provider: ProviderName,
    model: str | None,
    live: bool,
) -> Settings:
    if provider == ProviderName.OPENAI and not model:
        raise ProviderConfigurationError("OpenAI provider requires --model.")
    overrides: dict[str, object] = {"provider": provider.value}
    if model:
        overrides["model"] = model.strip()
    if provider == ProviderName.MOCK:
        overrides["model"] = "mock-model"
    if live:
        overrides["artifact_output_dir"] = "generated_artifacts"
    return build_settings(**overrides)


def _validate_live_benchmark(*, confirm_live: bool, model: str | None) -> None:
    if os.environ.get("RUN_LIVE_BENCHMARKS") != "1":
        raise ProviderConfigurationError(
            "Live benchmarks require RUN_LIVE_BENCHMARKS=1 in the environment."
        )
    if not confirm_live:
        raise ProviderConfigurationError("Live benchmarks require the --confirm-live flag.")
    if not model:
        raise ProviderConfigurationError("Live benchmarks require --model.")


def _print_error(message: str) -> None:
    escaped = message.replace("[", "\\[")
    console.print(f"[red]{escaped}[/red]")

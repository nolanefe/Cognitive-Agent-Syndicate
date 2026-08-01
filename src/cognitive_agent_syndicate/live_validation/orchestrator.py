"""Live validation orchestration."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from cognitive_agent_syndicate.benchmarking.exit_codes import (
    EXIT_CANCELLED,
    EXIT_COMPLETED_WITH_FAILURES,
    EXIT_FATAL,
    EXIT_SUCCESS,
)
from cognitive_agent_syndicate.benchmarking.pricing import PricingLoadError, load_pricing_config
from cognitive_agent_syndicate.benchmarking.reporting import BenchmarkOutputError
from cognitive_agent_syndicate.benchmarking.runner import (
    ProviderFactory,
    ReviewerProviderFactory,
    execute_benchmark,
)
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkDataset,
    BenchmarkMode,
    BenchmarkRun,
    BenchmarkTask,
)
from cognitive_agent_syndicate.config import ProviderName, Settings, apply_settings_overrides
from cognitive_agent_syndicate.live_validation.credentials import scoped_live_environment
from cognitive_agent_syndicate.live_validation.handoff import (
    render_live_validation_handoff_from_run,
    write_live_validation_json,
)
from cognitive_agent_syndicate.live_validation.plan import (
    build_live_validation_plan,
    render_live_validation_plan,
)
from cognitive_agent_syndicate.live_validation.preflight import (
    LiveValidationPreflightResult,
    PreflightError,
    run_live_validation_preflight,
)
from cognitive_agent_syndicate.live_validation.progress_display import (
    LiveValidationProgressReporter,
    build_progress_callback,
)
from cognitive_agent_syndicate.live_validation.smoke import LiveSmokeResult, run_live_provider_smoke
from cognitive_agent_syndicate.providers.base import ModelProvider
from cognitive_agent_syndicate.providers.factory import (
    create_model_provider,
    validate_provider_configuration,
)

if TYPE_CHECKING:
    from cognitive_agent_syndicate.benchmarking.schemas import PricingConfig
    from cognitive_agent_syndicate.providers.openai_types import OpenAIResponsesClient


@dataclass(frozen=True)
class LiveValidationOutcome:
    """Result of a live validation orchestration run."""

    exit_code: int
    smoke: LiveSmokeResult | None
    preflight: LiveValidationPreflightResult | None
    run: BenchmarkRun | None
    results_path: Path | None
    handoff_text: str | None
    cancelled: bool = False
    smoke_only: bool = False


SmokeRunner = Callable[[Settings], Awaitable[LiveSmokeResult]]
PromptFn = Callable[[str], str]


async def run_live_validation(
    *,
    dataset: str = "benchmarks/datasets/software_delivery_v1.json",
    task_ids: str | None = None,
    modes: str = "single_agent,contract_no_repair,contract_with_repair",
    repetitions: int = 1,
    model: str | None = None,
    reviewer_model: str | None = None,
    output_dir: str = "benchmark_results",
    benchmark_id: str | None = None,
    pricing_file: str | None = None,
    confirm_live: bool = False,
    smoke_only: bool = False,
    allow_dirty: bool = False,
    prompt_fn: PromptFn | None = None,
    smoke_runner: SmokeRunner | None = None,
    openai_client: OpenAIResponsesClient | None = None,
    progress_reporter: LiveValidationProgressReporter | None = None,
    cancelled_check: Callable[[], bool] | None = None,
    generate_benchmark_id: Callable[[list[str], int], str] | None = None,
    generation_provider_factory: ProviderFactory | None = None,
    reviewer_provider_factory: ReviewerProviderFactory | None = None,
) -> LiveValidationOutcome:
    """Orchestrate preflight, smoke, plan, and benchmark execution."""
    pricing: PricingConfig | None = None
    try:
        if pricing_file is not None:
            pricing = load_pricing_config(Path(pricing_file))
        preflight = run_live_validation_preflight(
            dataset=dataset,
            task_ids=task_ids,
            modes=modes,
            repetitions=repetitions,
            model=model,
            reviewer_model=reviewer_model,
            output_dir=output_dir,
            benchmark_id=benchmark_id,
            allow_dirty=allow_dirty,
            confirm_live=confirm_live,
            smoke_only=smoke_only,
            generate_benchmark_id=generate_benchmark_id,
        )
    except (PreflightError, PricingLoadError) as exc:
        return LiveValidationOutcome(
            exit_code=EXIT_FATAL,
            smoke=None,
            preflight=None,
            run=None,
            results_path=None,
            handoff_text=str(exc),
        )

    smoke: LiveSmokeResult | None = None
    run: BenchmarkRun | None = None
    results_path: Path | None = None
    handoff_text: str | None = None
    cancelled = False

    with scoped_live_environment(prompt_if_missing=True, prompt_fn=prompt_fn):
        settings = _build_live_settings(model=preflight.model, api_key_from_env=True)
        validate_provider_configuration(settings)

        smoke_call = smoke_runner or (
            lambda configured: run_live_provider_smoke(configured, client=openai_client)
        )
        smoke = await _await_smoke_call(smoke_call, settings)
        if not smoke.success:
            return LiveValidationOutcome(
                exit_code=EXIT_FATAL,
                smoke=smoke,
                preflight=preflight,
                run=None,
                results_path=None,
                handoff_text=_render_smoke_failure(smoke),
            )

        plan = build_live_validation_plan(
            dataset_name=preflight.dataset_name,
            dataset_version=preflight.dataset_version,
            task_ids=[task.task_id for task in preflight.selected_tasks],
            modes=preflight.modes,
            repetitions=preflight.repetitions,
            model=preflight.model,
            reviewer_model=preflight.reviewer_model,
            pricing=pricing,
            git=preflight.git,
            output_path=str(preflight.output_dir / preflight.benchmark_id),
        )
        plan_text = render_live_validation_plan(
            plan,
            git=preflight.git,
            output_path=str(preflight.output_dir / preflight.benchmark_id),
            pricing=pricing,
        )
        print(plan_text, flush=True)

        if smoke_only:
            json_path = preflight.output_dir / f"{preflight.benchmark_id}-smoke.json"
            write_live_validation_json(
                output_path=json_path,
                benchmark_id=preflight.benchmark_id,
                smoke=smoke,
                git=preflight.git,
                run=None,
                results_dir=None,
                final_status="smoke_only",
                benchmark_exit_status=None,
            )
            return LiveValidationOutcome(
                exit_code=EXIT_SUCCESS,
                smoke=smoke,
                preflight=preflight,
                run=None,
                results_path=None,
                handoff_text="Live smoke succeeded. Benchmark not started (--smoke-only).",
                smoke_only=True,
            )

        reporter = progress_reporter or LiveValidationProgressReporter()
        progress_callback = build_progress_callback(reporter)

        if generation_provider_factory is None:

            def generation_factory(task: BenchmarkTask, mode: BenchmarkMode) -> ModelProvider:
                return create_model_provider(settings, client=openai_client)

            resolved_generation_factory: ProviderFactory = generation_factory
        else:
            resolved_generation_factory = generation_provider_factory

        if reviewer_provider_factory is None:

            def reviewer_factory(task: BenchmarkTask, mode: BenchmarkMode) -> ModelProvider:
                if reviewer_model and reviewer_model != preflight.model:
                    reviewer_settings = apply_settings_overrides(
                        settings,
                        model=preflight.reviewer_model,
                    )
                    validate_provider_configuration(reviewer_settings)
                    return create_model_provider(reviewer_settings, client=openai_client)
                return create_model_provider(settings, client=openai_client)

            resolved_reviewer_factory: ReviewerProviderFactory = reviewer_factory
        else:
            resolved_reviewer_factory = reviewer_provider_factory

        try:
            run, results_path = await execute_benchmark(
                benchmark_id=preflight.benchmark_id,
                dataset=load_dataset(preflight),
                tasks=preflight.selected_tasks,
                modes=preflight.modes,
                repetitions=preflight.repetitions,
                settings=_build_live_settings(
                    model=preflight.model,
                    api_key_from_env=True,
                    artifact_output_dir="generated_artifacts",
                ),
                output_dir=preflight.output_dir,
                generation_provider_factory=resolved_generation_factory,
                reviewer_provider_factory=resolved_reviewer_factory,
                reviewer_model_label=preflight.reviewer_model,
                reviewer_provider_label=ProviderName.OPENAI.value,
                pricing=pricing,
                is_mock=False,
                progress_callback=progress_callback,
                cancelled_check=cancelled_check,
            )
        except BenchmarkOutputError as exc:
            return LiveValidationOutcome(
                exit_code=EXIT_FATAL,
                smoke=smoke,
                preflight=preflight,
                run=None,
                results_path=None,
                handoff_text=str(exc),
            )
        except Exception as exc:
            return LiveValidationOutcome(
                exit_code=EXIT_FATAL,
                smoke=smoke,
                preflight=preflight,
                run=run,
                results_path=results_path,
                handoff_text=f"Benchmark execution failed: {exc}",
            )
        except asyncio.CancelledError:
            cancelled = True
            return LiveValidationOutcome(
                exit_code=EXIT_CANCELLED,
                smoke=smoke,
                preflight=preflight,
                run=run,
                results_path=results_path,
                handoff_text="Live validation cancelled.",
                cancelled=True,
            )

        if cancelled_check and cancelled_check():
            cancelled = True

        assert run is not None
        assert results_path is not None
        assert run.summary is not None
        exit_code = _benchmark_exit_code(run, cancelled=cancelled)
        estimated_cost = _total_estimated_cost(run)
        handoff_text = render_live_validation_handoff_from_run(
            run=run,
            results_path=results_path,
            benchmark_exit_status=exit_code,
            estimated_cost_usd=estimated_cost,
        )
        write_live_validation_json(
            output_path=results_path / "live-validation.json",
            benchmark_id=preflight.benchmark_id,
            smoke=smoke,
            git=preflight.git,
            run=run,
            results_dir=results_path,
            final_status="cancelled" if cancelled else "completed",
            benchmark_exit_status=exit_code,
        )
        return LiveValidationOutcome(
            exit_code=exit_code,
            smoke=smoke,
            preflight=preflight,
            run=run,
            results_path=results_path,
            handoff_text=handoff_text,
            cancelled=cancelled,
        )


async def _await_smoke_call(
    smoke_call: SmokeRunner,
    settings: Settings,
) -> LiveSmokeResult:
    result = smoke_call(settings)
    if inspect.isawaitable(result):
        return await result
    return result


def load_dataset(preflight: LiveValidationPreflightResult) -> BenchmarkDataset:
    from cognitive_agent_syndicate.benchmarking.dataset import load_benchmark_dataset

    return load_benchmark_dataset(preflight.dataset_path)


def _build_live_settings(
    *,
    model: str,
    api_key_from_env: bool,
    artifact_output_dir: str | None = None,
) -> Settings:
    overrides: dict[str, object] = {
        "provider": ProviderName.OPENAI.value,
        "model": model,
    }
    if artifact_output_dir is not None:
        overrides["artifact_output_dir"] = artifact_output_dir
    if api_key_from_env:
        return apply_settings_overrides(base=None, **overrides)
    return apply_settings_overrides(base=None, **overrides)


def _render_smoke_failure(smoke: LiveSmokeResult) -> str:
    return "\n".join(
        [
            "LIVE SMOKE FAILED",
            f"Category: {smoke.failure_category}",
            f"Reason: {smoke.failure_reason}",
            "Benchmark not started.",
            "Credential state restored.",
        ]
    )


def _benchmark_exit_code(run: BenchmarkRun, *, cancelled: bool) -> int:
    if cancelled:
        return EXIT_CANCELLED
    summary = run.summary
    assert summary is not None
    if summary.attempted_trials > 0 and summary.successful_trials < summary.attempted_trials:
        return EXIT_COMPLETED_WITH_FAILURES
    return EXIT_SUCCESS


def _total_estimated_cost(run: BenchmarkRun) -> Decimal | None:
    summary = run.summary
    if summary is None or not summary.pricing_configured:
        return None
    costs = [
        mode.total_estimated_cost_usd
        for mode in summary.mode_summaries
        if mode.total_estimated_cost_usd is not None
    ]
    if not costs:
        return None
    return sum(costs, start=Decimal("0"))

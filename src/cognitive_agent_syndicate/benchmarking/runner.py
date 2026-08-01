"""Benchmark trial execution."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cognitive_agent_syndicate.agents.architect import ArchitectAgent
from cognitive_agent_syndicate.agents.implementer import ImplementerAgent
from cognitive_agent_syndicate.agents.reviewer import ReviewerAgent
from cognitive_agent_syndicate.benchmarking.adapters import (
    categorize_exception,
    classify_evaluable_failure,
    pipeline_to_trial_category,
    sanitize_failure_reason,
    trial_status_from_failure_category,
    trial_status_from_pipeline_category,
)
from cognitive_agent_syndicate.benchmarking.baseline import SingleAgentBaselineAgent
from cognitive_agent_syndicate.benchmarking.ids import validate_benchmark_id
from cognitive_agent_syndicate.benchmarking.metrics import enrich_trial_gate_fields
from cognitive_agent_syndicate.benchmarking.pricing import estimate_trial_cost
from cognitive_agent_syndicate.benchmarking.progress import (
    BenchmarkProgressEvent,
    BenchmarkProgressEventType,
    ProgressCallback,
    wrap_provider_for_progress,
)
from cognitive_agent_syndicate.benchmarking.provider_instrumentation import (
    ProviderCallCounter,
    observed_provider_call_count,
)
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkDataset,
    BenchmarkMode,
    BenchmarkRun,
    BenchmarkTask,
    BenchmarkTrial,
    PricingConfig,
    TrialFailureCategory,
    TrialStatus,
)
from cognitive_agent_syndicate.config import Settings
from cognitive_agent_syndicate.orchestration.clock import MonotonicClock, default_monotonic_clock
from cognitive_agent_syndicate.orchestration.pipeline import ContractDrivenPipeline
from cognitive_agent_syndicate.orchestration.state import PipelineState
from cognitive_agent_syndicate.providers.base import ModelProvider
from cognitive_agent_syndicate.reporting.report_writer import (
    build_single_agent_run_report,
    build_success_run_report_snapshot,
)
from cognitive_agent_syndicate.schemas import (
    GateResult,
    ReviewCategory,
    ReviewReport,
    ReviewStatus,
    UsageMetrics,
)
from cognitive_agent_syndicate.validation.gates import GateRunner

BenchmarkIdFactory = Callable[[], str]
RunIdFactory = Callable[[], str]
ProviderFactory = Callable[[BenchmarkTask, BenchmarkMode], ModelProvider]
ReviewerProviderFactory = Callable[[BenchmarkTask, BenchmarkMode], ModelProvider]


@dataclass
class BenchmarkRunContext:
    """Shared context for a benchmark execution."""

    benchmark_id: str
    dataset_version: str
    model_label: str
    reviewer_model_label: str
    pricing: PricingConfig | None = None
    cancelled: bool = False


@dataclass
class TrialExecutionResult:
    """Internal result from executing one trial."""

    trial: BenchmarkTrial
    pipeline_state: PipelineState | None = None
    generated_files: list[str] = field(default_factory=list)
    run_report_json: dict[str, object] | None = None


def _aggregate_usage(existing: UsageMetrics, latest: UsageMetrics) -> UsageMetrics:
    prompt_tokens = existing.prompt_tokens + latest.prompt_tokens
    completion_tokens = existing.completion_tokens + latest.completion_tokens
    return UsageMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=existing.latency_ms + latest.latency_ms,
    )


def _review_acceptance_passed_count(review: ReviewReport) -> int:
    return sum(
        1
        for finding in review.findings
        if finding.category == ReviewCategory.ACCEPTANCE_CRITERION and finding.passed is True
    )


async def run_single_agent_trial(
    *,
    task: BenchmarkTask,
    context: BenchmarkRunContext,
    mode: BenchmarkMode,
    repetition: int,
    generation_provider: ModelProvider,
    reviewer_provider: ModelProvider,
    settings: Settings,
    gate_runner: GateRunner,
    clock: MonotonicClock,
    run_id_factory: RunIdFactory,
    generation_counter: ProviderCallCounter,
    reviewer_counter: ProviderCallCounter,
) -> TrialExecutionResult:
    run_id = run_id_factory()
    start = clock()
    usage = UsageMetrics(prompt_tokens=0, completion_tokens=0, total_tokens=0, latency_ms=0.0)
    architecture = None
    bundle = None
    review = None
    gate_results: list[GateResult] = []

    baseline = SingleAgentBaselineAgent(generation_provider)
    reviewer = ReviewerAgent(reviewer_provider)
    gen_context = task.generation_context()

    try:
        delivery_result = await baseline.run(gen_context)
        usage = _aggregate_usage(usage, delivery_result.usage)
        architecture = delivery_result.response.architecture
        bundle = delivery_result.response.artifacts

        reviewer_result = await reviewer.run(
            brief=task.brief,
            architecture=architecture,
            bundle=bundle,
        )
        usage = _aggregate_usage(usage, reviewer_result.usage)
        review = reviewer_result.response

        gate_results = gate_runner.run(
            brief=task.brief,
            architecture=architecture,
            bundle=bundle,
            review=review,
            settings=settings,
            permitted_paths=task.permitted_paths,
            required_project_files=task.required_files,
        )
    except Exception as exc:
        end = clock()
        failure_category = categorize_exception(exc)
        trial = BenchmarkTrial(
            benchmark_id=context.benchmark_id,
            dataset_version=context.dataset_version,
            task_id=task.task_id,
            mode=mode,
            repetition=repetition,
            model_label=context.model_label,
            reviewer_model_label=context.reviewer_model_label,
            status=TrialStatus.FAILED,
            success=False,
            provider_call_count=observed_provider_call_count(
                generation_counter,
                reviewer_counter,
            ),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            provider_latency_ms=usage.latency_ms,
            wall_clock_duration_ms=max(0.0, (end - start) * 1000.0),
            failure_category=failure_category,
            failure_reason=sanitize_failure_reason(str(exc)),
        )
        return TrialExecutionResult(trial=trial)

    end = clock()
    wall_clock_duration_ms = max(0.0, (end - start) * 1000.0)
    gates_passed = GateRunner.all_required_passed(gate_results)
    reviewer_approved = review.status == ReviewStatus.APPROVED
    success = gates_passed and reviewer_approved

    evaluable_failure_category: TrialFailureCategory | None
    if success:
        evaluable_failure_category = None
    else:
        evaluable_failure_category = classify_evaluable_failure(
            reviewer_status=review.status,
            gate_results=gate_results,
        )

    trial = BenchmarkTrial(
        benchmark_id=context.benchmark_id,
        dataset_version=context.dataset_version,
        task_id=task.task_id,
        mode=mode,
        repetition=repetition,
        model_label=context.model_label,
        reviewer_model_label=context.reviewer_model_label,
        status=trial_status_from_failure_category(evaluable_failure_category, success=success),
        success=success,
        reviewer_status=review.status,
        gate_results=gate_results,
        repair_attempted=False,
        repair_succeeded=False,
        provider_call_count=observed_provider_call_count(
            generation_counter,
            reviewer_counter,
        ),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        provider_latency_ms=usage.latency_ms,
        wall_clock_duration_ms=wall_clock_duration_ms,
        estimated_cost=estimate_trial_cost(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            pricing=context.pricing,
        ),
        generated_file_count=len(bundle.files),
        failure_category=evaluable_failure_category,
        failure_reason=None
        if success
        else sanitize_failure_reason("Reviewer rejected or deterministic gates failed."),
    )
    trial = enrich_trial_gate_fields(
        trial,
        gate_results,
        total_criteria=len(task.brief.acceptance_criteria),
        review_passed_count=_review_acceptance_passed_count(review) if review else None,
    )

    state = PipelineState(
        run_id=run_id,
        brief=task.brief,
        architecture=architecture,
        artifacts=bundle,
        review=review,
        gate_results=gate_results,
        usage=usage,
        success=success,
    )
    generated_files = sorted(file.path for file in bundle.files)
    report = build_single_agent_run_report(
        run_id=run_id,
        brief_title=task.brief.title,
        gate_results=gate_results,
        usage=usage,
        success=success,
        generated_files=generated_files,
        review=review,
        wall_clock_duration_ms=wall_clock_duration_ms,
        gates_passed=gates_passed,
    )
    return TrialExecutionResult(
        trial=trial,
        pipeline_state=state,
        generated_files=generated_files if success else [],
        run_report_json=report.model_dump(mode="json"),
    )


async def run_contract_trial(
    *,
    task: BenchmarkTask,
    context: BenchmarkRunContext,
    mode: BenchmarkMode,
    repetition: int,
    generation_provider: ModelProvider,
    reviewer_provider: ModelProvider,
    settings: Settings,
    artifact_dir: Path,
    clock: MonotonicClock,
    run_id_factory: RunIdFactory,
    generation_counter: ProviderCallCounter,
    reviewer_counter: ProviderCallCounter,
) -> TrialExecutionResult:
    max_repair = 1 if mode == BenchmarkMode.CONTRACT_WITH_REPAIR else 0
    relative_artifact_dir = (
        f"generated_artifacts/benchmarks/{context.benchmark_id}/"
        f"{task.task_id}/{mode.value}/{repetition}"
    )
    trial_settings = settings.model_copy(
        update={
            "max_repair_attempts": max_repair,
            "artifact_output_dir": relative_artifact_dir,
        }
    )
    start = clock()

    pipeline = ContractDrivenPipeline(
        architect=ArchitectAgent(generation_provider),
        implementer=ImplementerAgent(generation_provider),
        reviewer=ReviewerAgent(reviewer_provider),
        settings=trial_settings,
        run_id_factory=run_id_factory,
        monotonic_clock=clock,
    )

    try:
        state = await pipeline.run(
            task.brief,
            allowed_technologies=task.allowed_technologies,
            permitted_paths=task.permitted_paths,
            implementation_constraints=task.implementation_constraints,
            required_project_files=task.required_files,
        )
    except Exception as exc:
        end = clock()
        category = categorize_exception(exc)
        trial = BenchmarkTrial(
            benchmark_id=context.benchmark_id,
            dataset_version=context.dataset_version,
            task_id=task.task_id,
            mode=mode,
            repetition=repetition,
            model_label=context.model_label,
            reviewer_model_label=context.reviewer_model_label,
            status=TrialStatus.FAILED,
            success=False,
            provider_call_count=observed_provider_call_count(
                generation_counter,
                reviewer_counter,
            ),
            failure_category=category,
            failure_reason=sanitize_failure_reason(str(exc)),
            wall_clock_duration_ms=max(0.0, (end - start) * 1000.0),
        )
        return TrialExecutionResult(trial=trial)

    end = clock()
    gate_results = state.gate_results
    if state.attempts:
        gate_results = state.attempts[-1].gate_results

    reviewer_status = state.review.status if state.review is not None else None
    generated_count = len(state.final_artifacts.files) if state.final_artifacts else 0
    if not state.success and state.artifacts is not None:
        generated_count = len(state.artifacts.files)

    failure_category = None
    failure_reason = state.failure_reason
    if state.failure_category is not None:
        failure_category = pipeline_to_trial_category(state.failure_category)
    elif not state.success:
        failure_category = classify_evaluable_failure(
            reviewer_status=reviewer_status,
            gate_results=gate_results,
        )

    trial_status = trial_status_from_pipeline_category(
        state.failure_category,
        success=state.success,
    )

    trial = BenchmarkTrial(
        benchmark_id=context.benchmark_id,
        dataset_version=context.dataset_version,
        task_id=task.task_id,
        mode=mode,
        repetition=repetition,
        model_label=context.model_label,
        reviewer_model_label=context.reviewer_model_label,
        status=trial_status,
        success=state.success,
        reviewer_status=reviewer_status,
        gate_results=gate_results,
        repair_attempted=state.repair_attempted,
        repair_succeeded=state.success and state.repair_attempted,
        provider_call_count=observed_provider_call_count(
            generation_counter,
            reviewer_counter,
        ),
        prompt_tokens=state.usage.prompt_tokens,
        completion_tokens=state.usage.completion_tokens,
        total_tokens=state.usage.total_tokens,
        provider_latency_ms=state.usage.latency_ms,
        wall_clock_duration_ms=max(0.0, (end - start) * 1000.0),
        estimated_cost=estimate_trial_cost(
            prompt_tokens=state.usage.prompt_tokens,
            completion_tokens=state.usage.completion_tokens,
            pricing=context.pricing,
        ),
        generated_file_count=generated_count,
        failure_category=failure_category,
        failure_reason=sanitize_failure_reason(failure_reason) if failure_reason else None,
    )
    trial = enrich_trial_gate_fields(
        trial,
        gate_results,
        total_criteria=len(task.brief.acceptance_criteria),
        review_passed_count=_review_acceptance_passed_count(state.review) if state.review else None,
    )

    generated_files: list[str] = []
    run_report_json: dict[str, object] | None = None
    if state.final_artifacts is not None:
        generated_files = sorted(file.path for file in state.final_artifacts.files)
    if state.success and state.attempts:
        report = build_success_run_report_snapshot(
            state=state,
            successful_attempt=state.attempts[-1],
            generated_files=generated_files,
            wall_clock_duration_ms=state.wall_clock_duration_ms,
        )
        run_report_json = report.model_dump(mode="json")
    elif state.artifact_directory:
        report_path = Path(state.artifact_directory) / "run-report.json"
        if report_path.exists():
            run_report_json = json.loads(report_path.read_text(encoding="utf-8"))

    return TrialExecutionResult(
        trial=trial,
        pipeline_state=state,
        generated_files=generated_files if state.success else [],
        run_report_json=run_report_json,
    )


async def execute_benchmark(
    *,
    benchmark_id: str,
    dataset: BenchmarkDataset,
    tasks: list[BenchmarkTask],
    modes: list[BenchmarkMode],
    repetitions: int,
    settings: Settings,
    output_dir: Path,
    generation_provider_factory: ProviderFactory,
    reviewer_provider_factory: ReviewerProviderFactory | None = None,
    reviewer_model_label: str | None = None,
    reviewer_provider_label: str | None = None,
    pricing: PricingConfig | None = None,
    is_mock: bool = True,
    clock: MonotonicClock | None = None,
    run_id_factory: RunIdFactory | None = None,
    cancelled_check: Callable[[], bool] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[BenchmarkRun, Path]:
    """Run a full benchmark sequentially and persist outputs."""
    from cognitive_agent_syndicate.benchmarking.metrics import build_benchmark_summary
    from cognitive_agent_syndicate.benchmarking.reporting import (
        build_config_snapshot,
        persist_benchmark_output,
        resolve_benchmark_output_dir,
    )

    validate_benchmark_id(benchmark_id)
    reviewer_factory = reviewer_provider_factory or generation_provider_factory
    model_label = settings.model
    resolved_reviewer_label = reviewer_model_label or model_label
    resolved_reviewer_provider = reviewer_provider_label or settings.provider.value
    same_model_reviewer = (
        resolved_reviewer_provider == settings.provider.value
        and resolved_reviewer_label == model_label
    )

    context = BenchmarkRunContext(
        benchmark_id=benchmark_id,
        dataset_version=dataset.version,
        model_label=model_label,
        reviewer_model_label=resolved_reviewer_label,
        pricing=pricing,
    )

    trials: list[BenchmarkTrial] = []
    trial_results: dict[tuple[str, BenchmarkMode, int], TrialExecutionResult] = {}
    output_root = resolve_benchmark_output_dir(output_dir, benchmark_id)
    staging_parent = output_root.parent
    total_trials = len(tasks) * len(modes) * repetitions
    trial_index = 0

    if progress_callback is not None:
        progress_callback(
            BenchmarkProgressEvent(
                event_type=BenchmarkProgressEventType.BENCHMARK_STARTED,
                total_trials=total_trials,
            )
        )

    for task in tasks:
        if context.cancelled:
            break
        for mode in modes:
            if context.cancelled:
                break
            for repetition in range(1, repetitions + 1):
                if cancelled_check and cancelled_check():
                    context.cancelled = True
                    break
                trial_index += 1
                if progress_callback is not None:
                    progress_callback(
                        BenchmarkProgressEvent(
                            event_type=BenchmarkProgressEventType.TRIAL_STARTED,
                            trial_index=trial_index,
                            total_trials=total_trials,
                            task_id=task.task_id,
                            mode=mode.value,
                            repetition=repetition,
                        )
                    )
                generation_provider = generation_provider_factory(task, mode)
                reviewer_provider = reviewer_factory(task, mode)
                trial_dir = (
                    staging_parent
                    / f".{benchmark_id}.trial"
                    / task.task_id
                    / mode.value
                    / str(repetition)
                )
                result = await run_benchmark_trial(
                    task=task,
                    mode=mode,
                    repetition=repetition,
                    context=context,
                    generation_provider=generation_provider,
                    reviewer_provider=reviewer_provider,
                    settings=settings,
                    trial_dir=trial_dir,
                    clock=clock,
                    run_id_factory=run_id_factory,
                    progress_callback=progress_callback,
                )
                trials.append(result.trial)
                trial_results[(task.task_id, mode, repetition)] = result
                if progress_callback is not None:
                    if result.trial.repair_attempted:
                        progress_callback(
                            BenchmarkProgressEvent(
                                event_type=BenchmarkProgressEventType.REPAIR_COMPLETED,
                                trial_index=trial_index,
                                total_trials=total_trials,
                                task_id=task.task_id,
                                mode=mode.value,
                                repetition=repetition,
                                repair_attempted=True,
                            )
                        )
                    event_type = (
                        BenchmarkProgressEventType.TRIAL_COMPLETED
                        if result.trial.status != TrialStatus.FAILED
                        else BenchmarkProgressEventType.TRIAL_FAILED
                    )
                    progress_callback(
                        BenchmarkProgressEvent(
                            event_type=event_type,
                            trial_index=trial_index,
                            total_trials=total_trials,
                            task_id=task.task_id,
                            mode=mode.value,
                            repetition=repetition,
                            trial_status=result.trial.status.value,
                            failure_category=(
                                result.trial.failure_category.value
                                if result.trial.failure_category is not None
                                else None
                            ),
                        )
                    )

    task_titles = {task.task_id: task.title for task in tasks}
    summary = build_benchmark_summary(
        benchmark_id=benchmark_id,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        modes=modes,
        repetitions=repetitions,
        model_label=model_label,
        reviewer_model_label=resolved_reviewer_label,
        same_model_reviewer=same_model_reviewer,
        pricing_configured=pricing is not None,
        is_mock=is_mock,
        trials=trials,
        task_titles=task_titles,
    )

    config = build_config_snapshot(
        benchmark_id=benchmark_id,
        dataset=dataset,
        task_ids=[task.task_id for task in tasks],
        modes=modes,
        repetitions=repetitions,
        model_label=model_label,
        reviewer_model_label=resolved_reviewer_label,
        reviewer_provider_label=resolved_reviewer_provider,
        generation_provider_label=settings.provider.value,
        pricing=pricing,
        is_mock=is_mock,
        temperature=settings.temperature,
    )

    if progress_callback is not None:
        progress_callback(
            BenchmarkProgressEvent(
                event_type=BenchmarkProgressEventType.BENCHMARK_PERSISTENCE_STARTED,
                total_trials=total_trials,
            )
        )

    final_path = persist_benchmark_output(
        output_root=output_root,
        benchmark_id=benchmark_id,
        config=config,
        trials=trials,
        summary=summary,
        trial_results=trial_results,
    )

    staging_trial_root = staging_parent / f".{benchmark_id}.trial"
    if staging_trial_root.exists():
        shutil.rmtree(staging_trial_root, ignore_errors=True)

    run = BenchmarkRun(
        benchmark_id=benchmark_id,
        dataset=dataset,
        modes=modes,
        repetitions=repetitions,
        model_label=model_label,
        reviewer_model_label=resolved_reviewer_label,
        reviewer_provider_label=resolved_reviewer_provider,
        generation_provider_label=settings.provider.value,
        pricing=pricing,
        is_mock=is_mock,
        trials=trials,
        summary=summary,
    )

    if progress_callback is not None:
        progress_callback(
            BenchmarkProgressEvent(
                event_type=BenchmarkProgressEventType.BENCHMARK_COMPLETED,
                total_trials=total_trials,
            )
        )

    return run, final_path


async def run_benchmark_trial(
    *,
    task: BenchmarkTask,
    mode: BenchmarkMode,
    repetition: int,
    context: BenchmarkRunContext,
    generation_provider: ModelProvider,
    reviewer_provider: ModelProvider,
    settings: Settings,
    trial_dir: Path,
    clock: MonotonicClock | None = None,
    run_id_factory: RunIdFactory | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TrialExecutionResult:
    """Execute one benchmark trial."""
    if context.cancelled:
        trial = BenchmarkTrial(
            benchmark_id=context.benchmark_id,
            dataset_version=context.dataset_version,
            task_id=task.task_id,
            mode=mode,
            repetition=repetition,
            model_label=context.model_label,
            reviewer_model_label=context.reviewer_model_label,
            status=TrialStatus.CANCELLED,
            success=False,
        )
        return TrialExecutionResult(trial=trial)

    resolved_clock = clock or default_monotonic_clock
    resolved_run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)
    gate_runner = GateRunner()

    generation_counter = ProviderCallCounter()
    reviewer_counter = ProviderCallCounter()
    counting_generation = wrap_provider_for_progress(
        generation_provider,
        generation_counter,
        progress_callback=progress_callback,
    )
    counting_reviewer = wrap_provider_for_progress(
        reviewer_provider,
        reviewer_counter,
        progress_callback=progress_callback,
    )

    if mode == BenchmarkMode.SINGLE_AGENT:
        return await run_single_agent_trial(
            task=task,
            context=context,
            mode=mode,
            repetition=repetition,
            generation_provider=counting_generation,
            reviewer_provider=counting_reviewer,
            settings=settings,
            gate_runner=gate_runner,
            clock=resolved_clock,
            run_id_factory=resolved_run_id_factory,
            generation_counter=generation_counter,
            reviewer_counter=reviewer_counter,
        )

    trial_dir.mkdir(parents=True, exist_ok=True)
    return await run_contract_trial(
        task=task,
        context=context,
        mode=mode,
        repetition=repetition,
        generation_provider=counting_generation,
        reviewer_provider=counting_reviewer,
        settings=settings,
        artifact_dir=trial_dir / "artifacts",
        clock=resolved_clock,
        run_id_factory=resolved_run_id_factory,
        generation_counter=generation_counter,
        reviewer_counter=reviewer_counter,
    )

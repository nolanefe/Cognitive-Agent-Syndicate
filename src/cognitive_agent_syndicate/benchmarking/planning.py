"""Dry-run benchmark planning without provider calls."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cognitive_agent_syndicate.benchmarking.display import format_dataset_label
from cognitive_agent_syndicate.benchmarking.schemas import BenchmarkMode


class BenchmarkPlan(BaseModel):
    """Dry-run plan describing an upcoming benchmark execution."""

    dataset_name: str
    dataset_version: str
    task_ids: list[str]
    modes: list[BenchmarkMode]
    repetitions: int = Field(..., ge=1, le=10)
    total_trials: int = Field(..., ge=0)
    min_provider_calls: int = Field(..., ge=0)
    max_provider_calls: int = Field(..., ge=0)
    repair_enabled: bool
    generation_provider: str
    generation_model: str
    reviewer_provider: str
    reviewer_model: str
    same_model_reviewer: bool
    pricing_configured: bool
    is_mock: bool
    is_live: bool
    live_safety_requirements: list[str] = Field(default_factory=list)


def provider_calls_for_mode(mode: BenchmarkMode) -> tuple[int, int]:
    """Return (min_calls, max_calls) for a benchmark mode."""
    if mode == BenchmarkMode.SINGLE_AGENT:
        return 2, 2
    if mode == BenchmarkMode.CONTRACT_NO_REPAIR:
        return 3, 3
    if mode == BenchmarkMode.CONTRACT_WITH_REPAIR:
        return 3, 5
    raise ValueError(f"Unsupported mode: {mode}")


def build_benchmark_plan(
    *,
    dataset_name: str,
    dataset_version: str,
    task_ids: list[str],
    modes: list[BenchmarkMode],
    repetitions: int,
    generation_provider: str,
    generation_model: str,
    reviewer_provider: str,
    reviewer_model: str,
    pricing_configured: bool,
    is_mock: bool,
    is_live: bool,
) -> BenchmarkPlan:
    """Build a dry-run benchmark plan."""
    total_trials = len(task_ids) * len(modes) * repetitions
    min_calls = 0
    max_calls = 0
    for mode in modes:
        mode_min, mode_max = provider_calls_for_mode(mode)
        min_calls += mode_min * len(task_ids) * repetitions
        max_calls += mode_max * len(task_ids) * repetitions

    repair_enabled = BenchmarkMode.CONTRACT_WITH_REPAIR in modes
    same_model = generation_provider == reviewer_provider and generation_model == reviewer_model

    live_requirements: list[str] = []
    if is_live:
        live_requirements = [
            "provider=openai must be explicitly selected",
            "RUN_LIVE_BENCHMARKS=1 environment variable required",
            "--confirm-live flag required",
            "--model must be explicitly provided",
            "repetitions capped at 5 for live runs",
            "API usage may incur cost",
        ]

    return BenchmarkPlan(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        task_ids=task_ids,
        modes=modes,
        repetitions=repetitions,
        total_trials=total_trials,
        min_provider_calls=min_calls,
        max_provider_calls=max_calls,
        repair_enabled=repair_enabled,
        generation_provider=generation_provider,
        generation_model=generation_model,
        reviewer_provider=reviewer_provider,
        reviewer_model=reviewer_model,
        same_model_reviewer=same_model,
        pricing_configured=pricing_configured,
        is_mock=is_mock,
        is_live=is_live,
        live_safety_requirements=live_requirements,
    )


def render_benchmark_plan(plan: BenchmarkPlan) -> str:
    """Render a human-readable benchmark plan."""
    mode_list = ", ".join(mode.value for mode in plan.modes)
    task_list = ", ".join(plan.task_ids)
    lines = [
        "Benchmark Plan (dry run — no provider calls)",
        f"Dataset: {format_dataset_label(plan.dataset_name, plan.dataset_version)}",
        f"Tasks: {task_list}",
        f"Modes: {mode_list}",
        f"Repetitions: {plan.repetitions}",
        f"Total trials: {plan.total_trials}",
        f"Min provider calls: {plan.min_provider_calls}",
        f"Max provider calls: {plan.max_provider_calls}",
        f"Repair enabled: {'yes' if plan.repair_enabled else 'no'}",
        f"Generation: {plan.generation_provider}/{plan.generation_model}",
        f"Reviewer: {plan.reviewer_provider}/{plan.reviewer_model}",
        f"Same model reviewer: {'yes' if plan.same_model_reviewer else 'no'}",
        f"Pricing configured: {'yes' if plan.pricing_configured else 'no'}",
        f"Execution: {'mock/offline' if plan.is_mock else 'live'}",
    ]
    if plan.is_live:
        lines.append("Live safety requirements:")
        lines.extend(f"  - {item}" for item in plan.live_safety_requirements)
    return "\n".join(lines)

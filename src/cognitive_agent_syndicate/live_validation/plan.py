"""Live validation plan rendering."""

from __future__ import annotations

from cognitive_agent_syndicate.benchmarking.display import format_dataset_label
from cognitive_agent_syndicate.benchmarking.planning import BenchmarkPlan, build_benchmark_plan
from cognitive_agent_syndicate.benchmarking.schemas import BenchmarkMode, PricingConfig
from cognitive_agent_syndicate.config import ProviderName
from cognitive_agent_syndicate.live_validation.preflight import GitMetadata


def build_live_validation_plan(
    *,
    dataset_name: str,
    dataset_version: str,
    task_ids: list[str],
    modes: list[BenchmarkMode],
    repetitions: int,
    model: str,
    reviewer_model: str,
    pricing: PricingConfig | None,
    git: GitMetadata,
    output_path: str,
) -> BenchmarkPlan:
    """Build the live validation benchmark plan."""
    del git, output_path
    return build_benchmark_plan(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        task_ids=task_ids,
        modes=modes,
        repetitions=repetitions,
        generation_provider=ProviderName.OPENAI.value,
        generation_model=model,
        reviewer_provider=ProviderName.OPENAI.value,
        reviewer_model=reviewer_model,
        pricing_configured=pricing is not None,
        is_mock=False,
        is_live=True,
    )


def render_live_validation_plan(
    plan: BenchmarkPlan,
    *,
    git: GitMetadata,
    output_path: str,
    pricing: PricingConfig | None = None,
) -> str:
    """Render the live validation plan for terminal display."""
    mode_list = ", ".join(mode.value for mode in plan.modes)
    task_list = ", ".join(plan.task_ids)
    pricing_label = "configured" if plan.pricing_configured else "not configured"
    if git.available and git.commit_sha:
        git_commit = git.commit_sha[:12]
    elif git.available:
        git_commit = "unavailable"
    else:
        git_commit = "unavailable (git not available)"
    if git.working_tree_clean is True:
        working_tree = "clean"
    elif git.working_tree_clean is False:
        working_tree = "dirty"
    else:
        working_tree = "unavailable"

    lines = [
        "LIVE VALIDATION PLAN",
        f"Dataset: {format_dataset_label(plan.dataset_name, plan.dataset_version)}",
        f"Tasks: {task_list}",
        f"Modes: {mode_list}",
        f"Repetitions: {plan.repetitions}",
        f"Trials: {plan.total_trials}",
        f"Model: {plan.generation_model}",
        f"Reviewer model: {plan.reviewer_model}",
        f"Min provider calls: {plan.min_provider_calls}",
        f"Max provider calls: {plan.max_provider_calls}",
        f"Pricing: {pricing_label}",
    ]
    if pricing is not None:
        lines.extend(
            [
                f"Pricing model: {pricing.model_label}",
                (
                    "Pricing rates (USD per 1M tokens): "
                    f"input {pricing.input_usd_per_million_tokens}, "
                    f"output {pricing.output_usd_per_million_tokens}"
                ),
                f"Pricing as-of: {pricing.effective_date.isoformat()}",
                "Pre-run cost estimate: unavailable until token usage is observed",
            ]
        )
    lines.extend(
        [
            "Generated code execution: disabled",
            f"Git commit: {git_commit}",
            f"Working tree: {working_tree}",
            f"Output: {output_path}",
            "",
            "API usage may incur cost.",
        ]
    )
    return "\n".join(lines)

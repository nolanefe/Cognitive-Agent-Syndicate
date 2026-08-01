"""Unit tests for live validation plan rendering."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from cognitive_agent_syndicate.benchmarking.ids import validate_benchmark_id
from cognitive_agent_syndicate.benchmarking.planning import build_benchmark_plan
from cognitive_agent_syndicate.benchmarking.schemas import BenchmarkMode, PricingConfig
from cognitive_agent_syndicate.live_validation.ids import generate_live_benchmark_id
from cognitive_agent_syndicate.live_validation.orchestrator import _total_estimated_cost
from cognitive_agent_syndicate.live_validation.plan import render_live_validation_plan
from cognitive_agent_syndicate.live_validation.preflight import GitMetadata


def test_plan_module_has_no_heuristic_token_constants() -> None:
    source = Path("src/cognitive_agent_syndicate/live_validation/plan.py").read_text(
        encoding="utf-8"
    )
    assert "avg_input_tokens" not in source
    assert "avg_output_tokens" not in source
    assert "estimate_plan_cost_bounds" not in source
    assert 'Decimal("500")' not in source
    assert 'Decimal("300")' not in source


def test_pre_run_plan_never_fabricates_monetary_estimate() -> None:
    plan = build_benchmark_plan(
        dataset_name="software_delivery",
        dataset_version="v1",
        task_ids=["task-url-shortener"],
        modes=[BenchmarkMode.SINGLE_AGENT],
        repetitions=1,
        generation_provider="openai",
        generation_model="gpt-test",
        reviewer_provider="openai",
        reviewer_model="gpt-test",
        pricing_configured=True,
        is_mock=False,
        is_live=True,
    )
    pricing = PricingConfig(
        model_label="example-model",
        input_usd_per_million_tokens=Decimal("1.50"),
        output_usd_per_million_tokens=Decimal("6.00"),
        source_or_note="Example pricing",
        effective_date=date(2026, 1, 1),
    )
    rendered = render_live_validation_plan(
        plan,
        git=GitMetadata(
            available=True,
            commit_sha="abc123",
            branch="main",
            working_tree_clean=True,
        ),
        output_path="benchmark_results/live-test",
        pricing=pricing,
    )
    assert "Pre-run cost estimate: unavailable until token usage is observed" in rendered
    assert "Estimated cost: $" not in rendered
    assert "$0." not in rendered


def test_post_run_observed_cost_still_computed_from_trials(tmp_path: Path) -> None:
    from cognitive_agent_syndicate.benchmarking.dataset import load_benchmark_dataset
    from cognitive_agent_syndicate.benchmarking.mock_fixtures import create_benchmark_mock_provider
    from cognitive_agent_syndicate.benchmarking.pricing import load_pricing_config
    from cognitive_agent_syndicate.benchmarking.runner import execute_benchmark
    from cognitive_agent_syndicate.config import build_settings

    async def _run() -> Decimal | None:
        dataset = load_benchmark_dataset(Path("benchmarks/datasets/software_delivery_v1.json"))
        task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")
        pricing = load_pricing_config(Path("benchmarks/pricing/example-pricing.json"))
        run, _ = await execute_benchmark(
            benchmark_id="live-cost-post-run",
            dataset=dataset,
            tasks=[task],
            modes=[BenchmarkMode.SINGLE_AGENT],
            repetitions=1,
            settings=build_settings(provider="mock"),
            output_dir=tmp_path / "benchmark_results",
            generation_provider_factory=lambda t, m: create_benchmark_mock_provider(t, m),
            pricing=pricing,
            is_mock=True,
        )
        return _total_estimated_cost(run)

    total = __import__("asyncio").run(_run())
    assert total is not None
    assert total > Decimal("0")


def test_generated_benchmark_id_passes_validator() -> None:
    generated = generate_live_benchmark_id(["task-url-shortener"], 2)
    assert validate_benchmark_id(generated) == generated

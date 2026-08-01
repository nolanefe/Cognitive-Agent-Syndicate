"""Tests for benchmark schemas and dataset validation."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from cognitive_agent_syndicate.benchmarking.dataset import (
    DatasetLoadError,
    filter_dataset_tasks,
    load_benchmark_dataset,
    parse_benchmark_modes,
    validate_repetitions,
)
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkDataset,
    BenchmarkMode,
    BenchmarkTask,
    PricingConfig,
    SingleAgentDelivery,
)
from cognitive_agent_syndicate.schemas import (
    AcceptanceCriterion,
    ArchitectureSpec,
    ArtifactBundle,
    ComponentSpec,
    GeneratedFile,
    SystemBrief,
)


def _minimal_brief() -> SystemBrief:
    return SystemBrief(
        title="Test",
        description="Test brief",
        acceptance_criteria=[
            AcceptanceCriterion(id="ac-1", description="Must pass", must_pass=True)
        ],
    )


def test_valid_dataset_loads() -> None:
    dataset = load_benchmark_dataset(Path("benchmarks/datasets/software_delivery_v1.json"))
    assert dataset.name == "software_delivery"
    assert dataset.version == "v1"
    assert len(dataset.tasks) == 6


def test_duplicate_task_ids_rejected() -> None:
    brief = _minimal_brief()
    task = BenchmarkTask(
        task_id="task-a",
        title="A",
        brief=brief,
        allowed_technologies=["python"],
        permitted_paths=["src"],
        implementation_constraints=["safe"],
        required_files=["pyproject.toml"],
    )
    with pytest.raises(ValidationError, match="Duplicate task IDs"):
        BenchmarkDataset(
            name="test",
            version="v1",
            description="test",
            tasks=[task, task.model_copy()],
            created_date=date(2026, 1, 1),
        )


def test_unsafe_required_path_rejected() -> None:
    with pytest.raises(ValidationError):
        BenchmarkTask(
            task_id="task-a",
            title="A",
            brief=_minimal_brief(),
            allowed_technologies=["python"],
            permitted_paths=["src"],
            implementation_constraints=["safe"],
            required_files=["../secrets.txt"],
        )


def test_notes_excluded_from_generation_context() -> None:
    task = BenchmarkTask(
        task_id="task-a",
        title="A",
        brief=_minimal_brief(),
        allowed_technologies=["python"],
        permitted_paths=["src"],
        implementation_constraints=["safe"],
        required_files=["pyproject.toml"],
        notes="Internal only",
    )
    context = task.generation_context()
    dumped = context.model_dump_json()
    assert "Internal only" not in dumped
    assert "notes" not in dumped


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid benchmark mode"):
        parse_benchmark_modes("not_a_mode")


def test_invalid_repetitions_rejected() -> None:
    with pytest.raises(ValueError, match="between 1 and 10"):
        validate_repetitions(0)
    with pytest.raises(ValueError, match="at most 5"):
        validate_repetitions(6, live=True)


def test_filter_unknown_task_ids() -> None:
    dataset = load_benchmark_dataset(Path("benchmarks/datasets/software_delivery_v1.json"))
    with pytest.raises(ValueError, match="Unknown task IDs"):
        filter_dataset_tasks(dataset, ["task-does-not-exist"])


def test_single_agent_delivery_schema() -> None:
    brief = _minimal_brief()
    delivery = SingleAgentDelivery(
        architecture=ArchitectureSpec(
            summary="s",
            components=[ComponentSpec(name="c", description="d", responsibilities=[])],
            acceptance_criteria=brief.acceptance_criteria,
        ),
        artifacts=ArtifactBundle(files=[GeneratedFile(path="src/a.py", content="x = 1\n")]),
    )
    assert delivery.architecture.summary == "s"


def test_pricing_config_decimal_rates() -> None:
    pricing = PricingConfig(
        model_label="m",
        input_usd_per_million_tokens=Decimal("1.0"),
        output_usd_per_million_tokens=Decimal("2.0"),
        source_or_note="test",
        effective_date=date(2026, 1, 1),
    )
    assert pricing.currency == "USD"


def test_invalid_dataset_file_raises() -> None:
    with pytest.raises(DatasetLoadError):
        load_benchmark_dataset(Path("does-not-exist.json"))


def test_parse_all_modes() -> None:
    modes = parse_benchmark_modes("single_agent,contract_no_repair,contract_with_repair")
    assert modes == [
        BenchmarkMode.SINGLE_AGENT,
        BenchmarkMode.CONTRACT_NO_REPAIR,
        BenchmarkMode.CONTRACT_WITH_REPAIR,
    ]

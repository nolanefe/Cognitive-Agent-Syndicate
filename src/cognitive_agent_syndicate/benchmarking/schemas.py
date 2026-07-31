"""Typed schemas for benchmark tasks, trials, and summaries."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from cognitive_agent_syndicate.benchmarking.ids import validate_benchmark_id
from cognitive_agent_syndicate.paths import normalize_relative_posix_path
from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    GateResult,
    ReviewStatus,
    SystemBrief,
    UsageMetrics,
)

MAX_BENCHMARK_TASKS = 50
MAX_BENCHMARK_TAGS = 10
MAX_BENCHMARK_STRING_LENGTH = 500
MAX_BENCHMARK_LIST_ITEMS = 20
MAX_BENCHMARK_NOTES_LENGTH = 2_000

_SECRET_PATTERNS = (
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*\S+"),
)


class BenchmarkMode(StrEnum):
    """Benchmark execution mode comparing pipeline strategies."""

    SINGLE_AGENT = "single_agent"
    CONTRACT_NO_REPAIR = "contract_no_repair"
    CONTRACT_WITH_REPAIR = "contract_with_repair"


class TrialStatus(StrEnum):
    """Outcome status for a single benchmark trial."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrialFailureCategory(StrEnum):
    """Typed failure category for benchmark trials."""

    PROVIDER_CONFIGURATION = "provider_configuration"
    PROVIDER_AUTHENTICATION = "provider_authentication"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_CONNECTION = "provider_connection"
    MALFORMED_STRUCTURED_OUTPUT = "malformed_structured_output"
    REVIEWER_REJECTED = "reviewer_rejected"
    DETERMINISTIC_GATE_FAILED = "deterministic_gate_failed"
    PERSISTENCE_FAILED = "persistence_failed"
    INTERNAL_ERROR = "internal_error"


class SingleAgentDelivery(BaseModel):
    """Combined architecture and artifacts from a single-agent generation call."""

    architecture: ArchitectureSpec
    artifacts: ArtifactBundle


class BenchmarkTask(BaseModel):
    """Bounded software-delivery task for benchmark evaluation."""

    task_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str = Field(..., min_length=1, max_length=200)
    tags: list[str] = Field(default_factory=list, max_length=MAX_BENCHMARK_TAGS)
    brief: SystemBrief
    allowed_technologies: list[str] = Field(..., min_length=1, max_length=MAX_BENCHMARK_LIST_ITEMS)
    permitted_paths: list[str] = Field(..., min_length=1, max_length=MAX_BENCHMARK_LIST_ITEMS)
    implementation_constraints: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BENCHMARK_LIST_ITEMS,
    )
    required_files: list[str] = Field(..., min_length=1, max_length=MAX_BENCHMARK_LIST_ITEMS)
    difficulty: str | None = Field(default=None, max_length=50)
    notes: str | None = Field(
        default=None,
        max_length=MAX_BENCHMARK_NOTES_LENGTH,
        description="Benchmark metadata excluded from agent context.",
    )

    @field_validator("tags", "allowed_technologies", "implementation_constraints")
    @classmethod
    def validate_bounded_strings(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value or len(value) > MAX_BENCHMARK_STRING_LENGTH:
                raise ValueError("List entries must be non-empty and bounded in length")
        return values

    @field_validator("permitted_paths", "required_files")
    @classmethod
    def validate_safe_paths(cls, values: list[str]) -> list[str]:
        return [normalize_relative_posix_path(value) for value in values]

    @model_validator(mode="after")
    def validate_acceptance_criteria(self) -> Self:
        if not self.brief.acceptance_criteria:
            raise ValueError("Task brief must contain at least one acceptance criterion")
        return self

    def generation_context(self) -> BenchmarkTaskGenerationContext:
        """Return task fields sent to generation agents (notes excluded)."""
        return BenchmarkTaskGenerationContext(
            task_id=self.task_id,
            title=self.title,
            tags=list(self.tags),
            brief=self.brief,
            allowed_technologies=list(self.allowed_technologies),
            permitted_paths=list(self.permitted_paths),
            implementation_constraints=list(self.implementation_constraints),
            required_files=list(self.required_files),
            difficulty=self.difficulty,
        )


class BenchmarkTaskGenerationContext(BaseModel):
    """Agent-visible subset of a benchmark task."""

    task_id: str
    title: str
    tags: list[str]
    brief: SystemBrief
    allowed_technologies: list[str]
    permitted_paths: list[str]
    implementation_constraints: list[str]
    required_files: list[str]
    difficulty: str | None = None


class BenchmarkDataset(BaseModel):
    """Versioned collection of benchmark tasks."""

    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., min_length=1, max_length=32)
    description: str = Field(..., min_length=1, max_length=2_000)
    tasks: list[BenchmarkTask] = Field(..., min_length=1, max_length=MAX_BENCHMARK_TASKS)
    created_date: date

    @model_validator(mode="after")
    def validate_unique_task_ids(self) -> Self:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs are not allowed")
        return self

    @model_validator(mode="after")
    def validate_no_secrets(self) -> Self:
        payload = self.model_dump_json()
        for pattern in _SECRET_PATTERNS:
            if pattern.search(payload):
                raise ValueError("Dataset must not contain secret-like values")
        return self


class PricingConfig(BaseModel):
    """User-supplied model pricing for cost estimation."""

    model_label: str = Field(..., min_length=1, max_length=100)
    input_usd_per_million_tokens: Decimal = Field(..., ge=0)
    output_usd_per_million_tokens: Decimal = Field(..., ge=0)
    source_or_note: str = Field(..., min_length=1, max_length=500)
    effective_date: date
    currency: Literal["USD"] = "USD"


class CostEstimate(BaseModel):
    """Estimated trial cost from user-supplied pricing."""

    input_cost_usd: Decimal
    output_cost_usd: Decimal
    total_cost_usd: Decimal
    pricing: PricingConfig


class BenchmarkTrial(BaseModel):
    """Result record for one benchmark trial."""

    benchmark_id: str = Field(..., min_length=1, max_length=64)
    dataset_version: str = Field(..., min_length=1, max_length=32)
    task_id: str = Field(..., min_length=1, max_length=64)
    mode: BenchmarkMode
    repetition: int = Field(..., ge=1, le=10)
    model_label: str = Field(..., min_length=1, max_length=100)
    reviewer_model_label: str = Field(..., min_length=1, max_length=100)
    status: TrialStatus
    success: bool
    reviewer_status: ReviewStatus | None = None
    gate_results: list[GateResult] = Field(default_factory=list)
    acceptance_criteria_passed: int = Field(default=0, ge=0)
    acceptance_criteria_total: int = Field(default=0, ge=0)
    required_files_gate_passed: bool | None = None
    syntax_gate_passed: bool | None = None
    forbidden_content_gate_passed: bool | None = None
    repair_attempted: bool = False
    repair_succeeded: bool = False
    provider_call_count: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    provider_latency_ms: float = Field(default=0.0, ge=0)
    wall_clock_duration_ms: float = Field(default=0.0, ge=0)
    estimated_cost: CostEstimate | None = None
    generated_file_count: int = Field(default=0, ge=0)
    failure_category: TrialFailureCategory | None = None
    failure_reason: str | None = Field(default=None, max_length=2_000)
    run_report_path: str | None = Field(default=None, max_length=500)

    @field_validator("benchmark_id")
    @classmethod
    def validate_trial_benchmark_id(cls, value: str) -> str:
        return validate_benchmark_id(value)


class ModeSummary(BaseModel):
    """Aggregated metrics for one benchmark mode."""

    mode: BenchmarkMode
    trial_count: int = Field(default=0, ge=0)
    attempted_trial_count: int = Field(default=0, ge=0)
    completed_trial_count: int = Field(default=0, ge=0)
    failed_trial_count: int = Field(default=0, ge=0)
    cancelled_trial_count: int = Field(default=0, ge=0)
    successful_trials: int = Field(default=0, ge=0)
    repair_attempt_count: int = Field(default=0, ge=0)
    repair_success_count: int = Field(default=0, ge=0)
    total_observed_provider_calls: int = Field(default=0, ge=0)
    success_rate: float | None = None
    reviewer_approval_rate: float | None = None
    required_gate_pass_rate: float | None = None
    acceptance_criterion_pass_rate: float | None = None
    syntax_pass_rate: float | None = None
    forbidden_content_pass_rate: float | None = None
    required_files_pass_rate: float | None = None
    repair_attempt_rate: float | None = None
    repair_success_rate: float | None = None
    avg_total_tokens: float | None = None
    median_total_tokens: float | None = None
    min_total_tokens: int | None = None
    max_total_tokens: int | None = None
    avg_provider_latency_ms: float | None = None
    median_provider_latency_ms: float | None = None
    min_provider_latency_ms: float | None = None
    max_provider_latency_ms: float | None = None
    avg_wall_clock_duration_ms: float | None = None
    median_wall_clock_duration_ms: float | None = None
    min_wall_clock_duration_ms: float | None = None
    max_wall_clock_duration_ms: float | None = None
    total_estimated_cost_usd: Decimal | None = None
    avg_estimated_cost_usd: Decimal | None = None
    failure_category_counts: dict[str, int] = Field(default_factory=dict)


class TaskSummary(BaseModel):
    """Aggregated metrics for one task across modes."""

    task_id: str
    title: str
    mode_summaries: list[ModeSummary] = Field(default_factory=list)


class BenchmarkSummary(BaseModel):
    """Overall benchmark run summary."""

    benchmark_id: str
    dataset_name: str
    dataset_version: str
    dataset_label: str
    modes: list[BenchmarkMode]
    repetitions: int
    model_label: str
    reviewer_model_label: str
    same_model_reviewer: bool
    pricing_configured: bool
    is_mock: bool
    total_trials: int
    attempted_trials: int = Field(default=0, ge=0)
    completed_trials: int
    failed_trials: int
    cancelled_trials: int
    successful_trials: int = Field(default=0, ge=0)
    total_observed_provider_calls: int = Field(default=0, ge=0)
    rate_rounding_note: str = Field(
        default=(
            "Rates use non-cancelled attempted trials as the denominator and are "
            "displayed rounded to two decimal places."
        )
    )
    mode_summaries: list[ModeSummary]
    task_summaries: list[TaskSummary]
    limitations: list[str] = Field(default_factory=list)


class BenchmarkRun(BaseModel):
    """Complete benchmark run with config, trials, and summary."""

    benchmark_id: str
    dataset: BenchmarkDataset
    modes: list[BenchmarkMode]
    repetitions: int
    model_label: str
    reviewer_model_label: str
    reviewer_provider_label: str
    generation_provider_label: str
    pricing: PricingConfig | None = None
    is_mock: bool = True
    trials: list[BenchmarkTrial] = Field(default_factory=list)
    summary: BenchmarkSummary | None = None


class BenchmarkConfigSnapshot(BaseModel):
    """Persisted benchmark configuration."""

    benchmark_id: str
    dataset_name: str
    dataset_version: str
    dataset_label: str
    task_ids: list[str]
    modes: list[BenchmarkMode]
    repetitions: int
    model_label: str
    reviewer_model_label: str
    reviewer_provider_label: str
    generation_provider_label: str
    pricing: PricingConfig | None = None
    is_mock: bool
    temperature: float
    limitations: list[str] = Field(default_factory=list)

    @field_validator("benchmark_id")
    @classmethod
    def validate_config_benchmark_id(cls, value: str) -> str:
        return validate_benchmark_id(value)


class TrialUsageSnapshot(BaseModel):
    """Usage metrics collected during a trial."""

    usage: UsageMetrics
    provider_call_count: int = Field(..., ge=0)

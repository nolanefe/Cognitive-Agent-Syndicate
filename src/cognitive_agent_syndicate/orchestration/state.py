"""Pipeline state models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    GateResult,
    PipelineAttempt,
    ReviewReport,
    SystemBrief,
    UsageMetrics,
)


class PipelineStage(StrEnum):
    INIT = "init"
    ARCHITECT = "architect"
    IMPLEMENTER = "implementer"
    REVIEWER = "reviewer"
    REPAIR = "repair"
    GATES = "gates"
    PERSISTENCE = "persistence"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineState(BaseModel):
    """In-memory record of a pipeline run."""

    run_id: str = Field(..., min_length=1, max_length=64)
    brief: SystemBrief
    architecture: ArchitectureSpec | None = None
    artifacts: ArtifactBundle | None = None
    review: ReviewReport | None = None
    gate_results: list[GateResult] = Field(default_factory=list)
    usage: UsageMetrics = Field(
        default_factory=lambda: UsageMetrics(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=0.0,
        )
    )
    stage: PipelineStage = PipelineStage.INIT
    success: bool = False
    failure_reason: str | None = None
    artifact_directory: str | None = None
    stages_completed: list[PipelineStage] = Field(default_factory=list)
    attempts: list[PipelineAttempt] = Field(default_factory=list, max_length=2)
    repair_attempted: bool = False
    repair_trigger: str | None = Field(default=None, max_length=2_000)
    final_artifacts: ArtifactBundle | None = None
    architect_usage: UsageMetrics = Field(
        default_factory=lambda: UsageMetrics(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=0.0,
        )
    )
    pipeline_started_at_ms: float = Field(default=0.0, ge=0)
    pipeline_ended_at_ms: float = Field(default=0.0, ge=0)
    wall_clock_duration_ms: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_usage_totals(self) -> PipelineState:
        if self.usage.total_tokens != self.usage.prompt_tokens + self.usage.completion_tokens:
            raise ValueError("Aggregated usage total_tokens must equal prompt + completion tokens")
        return self

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

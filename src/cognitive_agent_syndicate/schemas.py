"""Pydantic schemas for pipeline contracts and artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from cognitive_agent_syndicate.paths import canonical_path_key, normalize_relative_posix_path

MAX_GENERATED_FILE_CONTENT = 100_000
MAX_GENERATED_FILES = 100
MAX_ACCEPTANCE_CRITERIA = 50
MAX_ARCHITECTURE_STRINGS = 30
MAX_ARCHITECTURE_STRING_LENGTH = 500
MAX_REVIEW_STRINGS = 50
MAX_REVIEW_STRING_LENGTH = 1_000


class ReviewSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ReviewStatus(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class ReviewCategory(StrEnum):
    ACCEPTANCE_CRITERION = "acceptance_criterion"
    UNSUPPORTED_ASSUMPTION = "unsupported_assumption"
    CONTRACT_VIOLATION = "contract_violation"
    SECURITY = "security"
    QUALITY = "quality"


class GateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GateRepairability(StrEnum):
    REPAIRABLE = "repairable"
    NON_REPAIRABLE = "non_repairable"


class AttemptOutcome(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


class AcceptanceCriterion(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=500)
    must_pass: bool = True


class SystemBrief(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=5_000)
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        ...,
        min_length=1,
        max_length=MAX_ACCEPTANCE_CRITERIA,
    )


class ComponentSpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=2_000)
    responsibilities: list[str] = Field(default_factory=list, max_length=20)


class EndpointSpec(BaseModel):
    path: str = Field(..., min_length=1, max_length=200)
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    description: str = Field(..., min_length=1, max_length=1_000)
    request_model: str | None = Field(default=None, max_length=100)
    response_model: str | None = Field(default=None, max_length=100)


class DataModelSpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=2_000)
    fields: list[str] = Field(..., min_length=1, max_length=50)


class ArchitectureSpec(BaseModel):
    summary: str = Field(..., min_length=1, max_length=5_000)
    assumptions: list[str] = Field(default_factory=list, max_length=MAX_ARCHITECTURE_STRINGS)
    components: list[ComponentSpec] = Field(..., min_length=1, max_length=20)
    endpoints: list[EndpointSpec] = Field(default_factory=list, max_length=50)
    data_models: list[DataModelSpec] = Field(default_factory=list, max_length=50)
    dependencies: list[str] = Field(default_factory=list, max_length=MAX_ARCHITECTURE_STRINGS)
    security_constraints: list[str] = Field(
        default_factory=list,
        max_length=MAX_ARCHITECTURE_STRINGS,
    )
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        ...,
        min_length=1,
        max_length=MAX_ACCEPTANCE_CRITERIA,
    )
    implementation_risks: list[str] = Field(
        default_factory=list,
        max_length=MAX_ARCHITECTURE_STRINGS,
    )

    @field_validator(
        "assumptions",
        "dependencies",
        "security_constraints",
        "implementation_risks",
    )
    @classmethod
    def validate_bounded_strings(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value or len(value) > MAX_ARCHITECTURE_STRING_LENGTH:
                raise ValueError(
                    "Architecture list entries must be non-empty and bounded in length"
                )
        return values


class GeneratedFile(BaseModel):
    path: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1, max_length=MAX_GENERATED_FILE_CONTENT)
    language: str | None = Field(default=None, max_length=50)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return normalize_relative_posix_path(value)


class ArtifactBundle(BaseModel):
    files: list[GeneratedFile] = Field(..., min_length=1, max_length=MAX_GENERATED_FILES)

    @model_validator(mode="after")
    def check_unique_paths(self) -> Self:
        keys = [canonical_path_key(generated_file.path) for generated_file in self.files]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate generated file paths are not allowed")
        return self


class ReviewFinding(BaseModel):
    criterion_id: str = Field(..., min_length=1, max_length=64)
    category: ReviewCategory
    severity: ReviewSeverity
    message: str = Field(..., min_length=1, max_length=2_000)
    suggestion: str | None = Field(default=None, max_length=2_000)
    passed: bool | None = None

    @model_validator(mode="after")
    def validate_passed_for_acceptance_criterion(self) -> Self:
        if self.category == ReviewCategory.ACCEPTANCE_CRITERION and self.passed is None:
            raise ValueError("passed must be explicitly provided for ACCEPTANCE_CRITERION findings")
        return self


class ReviewReport(BaseModel):
    status: ReviewStatus
    findings: list[ReviewFinding] = Field(default_factory=list, max_length=100)
    summary: str = Field(..., min_length=1, max_length=5_000)
    unsupported_assumptions: list[str] = Field(
        default_factory=list,
        max_length=MAX_REVIEW_STRINGS,
    )
    contract_violations: list[str] = Field(default_factory=list, max_length=MAX_REVIEW_STRINGS)
    security_concerns: list[str] = Field(default_factory=list, max_length=MAX_REVIEW_STRINGS)
    recommended_repairs: list[str] = Field(default_factory=list, max_length=MAX_REVIEW_STRINGS)

    @field_validator(
        "unsupported_assumptions",
        "contract_violations",
        "security_concerns",
        "recommended_repairs",
    )
    @classmethod
    def validate_review_strings(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value or len(value) > MAX_REVIEW_STRING_LENGTH:
                raise ValueError("Review list entries must be non-empty and bounded in length")
        return values


class GateResult(BaseModel):
    gate_id: str = Field(..., min_length=1, max_length=100)
    gate_name: str = Field(..., min_length=1, max_length=100)
    status: GateStatus
    message: str = Field(..., min_length=1, max_length=2_000)
    duration_ms: float = Field(..., ge=0)
    required: bool = True
    repairable: GateRepairability = GateRepairability.REPAIRABLE


class RepairInstruction(BaseModel):
    source: Literal["gate", "reviewer"] = Field(...)
    gate_id: str | None = Field(default=None, max_length=100)
    message: str = Field(..., min_length=1, max_length=2_000)
    suggestion: str | None = Field(default=None, max_length=2_000)


class RepairRequest(BaseModel):
    """Bounded repair context passed to the implementer repair call."""

    brief: SystemBrief
    architecture: ArchitectureSpec
    current_bundle: ArtifactBundle
    gate_failures: list[GateResult] = Field(default_factory=list, max_length=20)
    reviewer_findings: list[ReviewFinding] = Field(default_factory=list, max_length=100)
    allowed_technologies: list[str] = Field(..., max_length=20)
    permitted_paths: list[str] = Field(..., max_length=20)
    implementation_constraints: list[str] = Field(..., max_length=20)
    permitted_file_changes: list[str] = Field(..., max_length=MAX_GENERATED_FILES)
    repair_instructions: list[RepairInstruction] = Field(default_factory=list, max_length=50)


class PipelineAttempt(BaseModel):
    attempt_number: int = Field(..., ge=1, le=2)
    artifacts: ArtifactBundle | None = None
    review: ReviewReport | None = None
    gate_results: list[GateResult] = Field(default_factory=list, max_length=30)
    usage: UsageMetrics = Field(
        default_factory=lambda: UsageMetrics(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=0.0,
        )
    )
    outcome: AttemptOutcome = AttemptOutcome.FAILED
    failure_reason: str | None = Field(default=None, max_length=2_000)
    reviewer_status: ReviewStatus | None = None
    started_at_ms: float = Field(default=0.0, ge=0)
    ended_at_ms: float = Field(default=0.0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0)
    gates_passed: bool = False
    reviewer_approved: bool = False


class AttemptSummary(BaseModel):
    attempt_number: int = Field(..., ge=1, le=2)
    outcome: AttemptOutcome
    reviewer_status: ReviewStatus | None = None
    gates_passed: bool
    reviewer_approved: bool
    failure_reason: str | None = Field(default=None, max_length=2_000)
    usage: UsageMetrics
    duration_ms: float = Field(..., ge=0)


class UsageMetrics(BaseModel):
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_total_tokens(self) -> Self:
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        return self


class RunReport(BaseModel):
    run_id: str = Field(..., min_length=1, max_length=64)
    brief_title: str = Field(..., min_length=1, max_length=200)
    gates: list[GateResult] = Field(default_factory=list, max_length=30)
    usage: UsageMetrics
    success: bool
    artifact_count: int = Field(..., ge=0)
    stages_completed: list[str] = Field(default_factory=list, max_length=30)
    reviewer_status: ReviewStatus | None = None
    failure_reason: str | None = Field(default=None, max_length=2_000)
    generated_files: list[str] = Field(default_factory=list, max_length=MAX_GENERATED_FILES)
    limitations: list[str] = Field(default_factory=list, max_length=20)
    repair_attempted: bool = False
    attempt_count: int = Field(default=1, ge=1, le=2)
    attempts: list[AttemptSummary] = Field(default_factory=list, max_length=2)
    repair_trigger: str | None = Field(default=None, max_length=2_000)
    wall_clock_duration_ms: float = Field(default=0.0, ge=0)
    provider_latency_ms: float = Field(
        default=0.0,
        ge=0,
        description="Sum of provider-reported latency across all agent calls.",
    )

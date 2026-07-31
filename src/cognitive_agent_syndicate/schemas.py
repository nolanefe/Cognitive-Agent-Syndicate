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
    gate_name: str = Field(..., min_length=1, max_length=100)
    status: GateStatus
    message: str = Field(..., min_length=1, max_length=2_000)
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
    gates: list[GateResult] = Field(default_factory=list, max_length=20)
    usage: UsageMetrics
    success: bool
    artifact_count: int = Field(..., ge=0)

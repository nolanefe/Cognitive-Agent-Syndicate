"""Typed pipeline failure categories and safe exception mapping."""

from __future__ import annotations

from enum import StrEnum

from pydantic import ValidationError

from cognitive_agent_syndicate.providers.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from cognitive_agent_syndicate.schemas import GateResult, GateStatus, ReviewStatus


class PipelineFailureCategory(StrEnum):
    """Provider-independent failure category for pipeline runs."""

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


EVALUABLE_PIPELINE_FAILURE_CATEGORIES = frozenset(
    {
        PipelineFailureCategory.REVIEWER_REJECTED,
        PipelineFailureCategory.DETERMINISTIC_GATE_FAILED,
    }
)

FATAL_PIPELINE_FAILURE_CATEGORIES = frozenset(
    category
    for category in PipelineFailureCategory
    if category not in EVALUABLE_PIPELINE_FAILURE_CATEGORIES
)


def categorize_pipeline_exception(exc: BaseException) -> PipelineFailureCategory:
    """Map an exception to a typed pipeline failure category."""
    from cognitive_agent_syndicate.reporting.artifacts import ArtifactPersistenceError

    if isinstance(exc, ProviderConfigurationError):
        return PipelineFailureCategory.PROVIDER_CONFIGURATION
    if isinstance(exc, ProviderAuthenticationError):
        return PipelineFailureCategory.PROVIDER_AUTHENTICATION
    if isinstance(exc, ProviderRateLimitError):
        return PipelineFailureCategory.PROVIDER_RATE_LIMIT
    if isinstance(exc, ProviderTimeoutError):
        return PipelineFailureCategory.PROVIDER_TIMEOUT
    if isinstance(exc, ProviderConnectionError):
        return PipelineFailureCategory.PROVIDER_CONNECTION
    if isinstance(exc, ProviderMalformedResponseError | ValidationError):
        return PipelineFailureCategory.MALFORMED_STRUCTURED_OUTPUT
    if isinstance(exc, ArtifactPersistenceError):
        return PipelineFailureCategory.PERSISTENCE_FAILED
    if isinstance(exc, ProviderError):
        return PipelineFailureCategory.INTERNAL_ERROR
    return PipelineFailureCategory.INTERNAL_ERROR


def infer_evaluable_failure_category(
    *,
    reviewer_status: ReviewStatus | None,
    gate_results: list[GateResult],
) -> PipelineFailureCategory:
    """Infer evaluable failure category from reviewer and gate outcomes."""
    if reviewer_status in {ReviewStatus.REJECTED, ReviewStatus.NEEDS_REVISION}:
        return PipelineFailureCategory.REVIEWER_REJECTED
    if any(gate.status == GateStatus.FAILED and gate.required for gate in gate_results):
        return PipelineFailureCategory.DETERMINISTIC_GATE_FAILED
    return PipelineFailureCategory.INTERNAL_ERROR

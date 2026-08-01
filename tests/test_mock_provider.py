"""Unit tests for the deterministic mock model provider."""

import pytest

from cognitive_agent_syndicate.providers.mock import (
    MockModelProvider,
    MockResponseNotConfiguredError,
)
from cognitive_agent_syndicate.schemas import (
    AcceptanceCriterion,
    ReviewReport,
    ReviewStatus,
    SystemBrief,
)


@pytest.mark.asyncio
async def test_mock_provider_returns_configured_response() -> None:
    provider = MockModelProvider()
    expected = SystemBrief(
        title="Mock Brief",
        description="Generated offline.",
        acceptance_criteria=[
            AcceptanceCriterion(id="ac-1", description="Runs without network access."),
        ],
    )
    provider.configure_response(SystemBrief, expected)

    result = await provider.generate(
        system_instructions="You are the architect.",
        user_content="Design a service.",
        response_type=SystemBrief,
    )

    assert result.response.title == "Mock Brief"
    assert result.usage.total_tokens == 20
    assert result.usage.latency_ms == 1.5


@pytest.mark.asyncio
async def test_mock_provider_records_calls() -> None:
    provider = MockModelProvider()
    provider.configure_response(
        ReviewReport,
        ReviewReport(status=ReviewStatus.APPROVED, summary="Looks good."),
    )

    await provider.generate(
        system_instructions="Review the artifacts.",
        user_content="Check bundle v1.",
        response_type=ReviewReport,
    )

    assert len(provider.calls) == 1
    assert provider.calls[0].system_instructions == "Review the artifacts."
    assert provider.calls[0].user_content == "Check bundle v1."
    assert provider.calls[0].response_type is ReviewReport


@pytest.mark.asyncio
async def test_mock_provider_raises_when_response_missing() -> None:
    provider = MockModelProvider()

    with pytest.raises(MockResponseNotConfiguredError) as exc_info:
        await provider.generate(
            system_instructions="Architect",
            user_content="No configured response.",
            response_type=SystemBrief,
        )

    assert exc_info.value.response_type is SystemBrief
    assert "No mock response configured" in str(exc_info.value)


def test_mock_provider_rejects_wrong_response_type() -> None:
    provider = MockModelProvider()

    wrong_response = ReviewReport(status=ReviewStatus.APPROVED, summary="x")

    with pytest.raises(TypeError):
        provider.configure_response(SystemBrief, wrong_response)


@pytest.mark.asyncio
async def test_mock_provider_isolates_configured_response_from_mutation() -> None:
    provider = MockModelProvider()
    configured = SystemBrief(
        title="Original",
        description="Generated offline.",
        acceptance_criteria=[
            AcceptanceCriterion(id="ac-1", description="Initial criterion."),
        ],
    )
    provider.configure_response(SystemBrief, configured)
    configured.title = "Mutated"

    result = await provider.generate(
        system_instructions="Architect",
        user_content="Design a service.",
        response_type=SystemBrief,
    )

    assert result.response.title == "Original"


@pytest.mark.asyncio
async def test_mock_provider_isolates_generation_results_from_mutation() -> None:
    provider = MockModelProvider()
    provider.configure_response(
        SystemBrief,
        SystemBrief(
            title="Stable",
            description="Generated offline.",
            acceptance_criteria=[
                AcceptanceCriterion(id="ac-1", description="Stable criterion."),
            ],
        ),
    )

    first = await provider.generate(
        system_instructions="Architect",
        user_content="First call.",
        response_type=SystemBrief,
    )
    first.response.title = "Changed"
    first.usage.prompt_tokens = 999

    second = await provider.generate(
        system_instructions="Architect",
        user_content="Second call.",
        response_type=SystemBrief,
    )

    assert second.response.title == "Stable"
    assert second.usage.prompt_tokens == 12


@pytest.mark.asyncio
async def test_mock_provider_prefers_exact_user_content_over_fallback() -> None:
    provider = MockModelProvider()
    provider.configure_response(
        SystemBrief,
        SystemBrief(
            title="Fallback",
            description="Fallback response.",
            acceptance_criteria=[
                AcceptanceCriterion(id="ac-1", description="Fallback criterion."),
            ],
        ),
    )
    provider.configure_response(
        SystemBrief,
        SystemBrief(
            title="Exact Match",
            description="Exact response.",
            acceptance_criteria=[
                AcceptanceCriterion(id="ac-2", description="Exact criterion."),
            ],
        ),
        user_content="specific prompt",
    )

    exact = await provider.generate(
        system_instructions="Architect",
        user_content="specific prompt",
        response_type=SystemBrief,
    )
    fallback = await provider.generate(
        system_instructions="Architect",
        user_content="other prompt",
        response_type=SystemBrief,
    )

    assert exact.response.title == "Exact Match"
    assert fallback.response.title == "Fallback"

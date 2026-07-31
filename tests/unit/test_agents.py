"""Unit tests for pipeline agents."""

import pytest

from cognitive_agent_syndicate.agents.architect import ArchitectAgent
from cognitive_agent_syndicate.agents.implementer import ImplementerAgent
from cognitive_agent_syndicate.agents.reviewer import ReviewerAgent
from cognitive_agent_syndicate.providers.mock import (
    MockModelProvider,
    MockResponseNotConfiguredError,
)
from cognitive_agent_syndicate.schemas import ArchitectureSpec, ArtifactBundle, ReviewReport
from tests.fixtures.pipeline_fixtures import (
    sample_architecture,
    sample_brief,
    sample_bundle,
    sample_review_approved,
    sample_usage,
)


@pytest.mark.asyncio
async def test_architect_agent_requests_architecture_spec() -> None:
    provider = MockModelProvider(usage=sample_usage(prompt=11, completion=4))
    provider.configure_response(ArchitectureSpec, sample_architecture())
    agent = ArchitectAgent(provider)

    result = await agent.run(sample_brief())

    assert isinstance(result.response, ArchitectureSpec)
    assert result.usage.total_tokens == 15
    assert provider.calls[0].response_type is ArchitectureSpec


@pytest.mark.asyncio
async def test_implementer_agent_requests_artifact_bundle() -> None:
    provider = MockModelProvider(usage=sample_usage(prompt=20, completion=10))
    provider.configure_response(ArtifactBundle, sample_bundle())
    agent = ImplementerAgent(provider)

    result = await agent.run(
        brief=sample_brief(),
        architecture=sample_architecture(),
        allowed_technologies=["python"],
        permitted_paths=["src", "tests"],
        implementation_constraints=["offline only"],
    )

    assert isinstance(result.response, ArtifactBundle)
    assert result.usage.prompt_tokens == 20
    assert provider.calls[0].response_type is ArtifactBundle


@pytest.mark.asyncio
async def test_reviewer_agent_requests_review_report() -> None:
    provider = MockModelProvider(usage=sample_usage(prompt=8, completion=2))
    provider.configure_response(ReviewReport, sample_review_approved())
    agent = ReviewerAgent(provider)

    result = await agent.run(
        brief=sample_brief(),
        architecture=sample_architecture(),
        bundle=sample_bundle(),
    )

    assert isinstance(result.response, ReviewReport)
    assert result.usage.completion_tokens == 2
    assert provider.calls[0].response_type is ReviewReport


@pytest.mark.asyncio
async def test_agent_provider_errors_propagate() -> None:
    provider = MockModelProvider()
    agent = ArchitectAgent(provider)

    with pytest.raises(MockResponseNotConfiguredError):
        await agent.run(sample_brief())


@pytest.mark.asyncio
async def test_architect_agent_does_not_include_system_prompt_in_response() -> None:
    provider = MockModelProvider()
    provider.configure_response(ArchitectureSpec, sample_architecture())
    agent = ArchitectAgent(provider)

    await agent.run(sample_brief())

    call = provider.calls[0]
    assert "sk-" not in call.system_instructions
    assert call.user_content.startswith("Design an architecture")

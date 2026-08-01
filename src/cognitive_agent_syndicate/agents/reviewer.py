"""Reviewer agent for evaluating ArtifactBundle against contracts."""

from __future__ import annotations

from cognitive_agent_syndicate.prompts import (
    build_reviewer_system_instructions,
    build_reviewer_user_content,
)
from cognitive_agent_syndicate.providers.base import GenerationResult, ModelProvider
from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    ReviewReport,
    SystemBrief,
)


class ReviewerAgent:
    """Generates a validated ReviewReport for an artifact bundle."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    async def run(
        self,
        *,
        brief: SystemBrief,
        architecture: ArchitectureSpec,
        bundle: ArtifactBundle,
    ) -> GenerationResult[ReviewReport]:
        system_instructions = build_reviewer_system_instructions()
        user_content = build_reviewer_user_content(
            brief=brief,
            architecture=architecture,
            bundle=bundle,
        )
        return await self._provider.generate(
            system_instructions=system_instructions,
            user_content=user_content,
            response_type=ReviewReport,
        )

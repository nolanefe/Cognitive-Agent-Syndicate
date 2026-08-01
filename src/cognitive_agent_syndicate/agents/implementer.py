"""Implementer agent for generating ArtifactBundle from architecture."""

from __future__ import annotations

from cognitive_agent_syndicate.prompts import (
    build_implementer_repair_system_instructions,
    build_implementer_repair_user_content,
    build_implementer_system_instructions,
    build_implementer_user_content,
)
from cognitive_agent_syndicate.providers.base import GenerationResult, ModelProvider
from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    RepairRequest,
    SystemBrief,
)


class ImplementerAgent:
    """Generates a validated ArtifactBundle from architecture and constraints."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    async def run(
        self,
        *,
        brief: SystemBrief,
        architecture: ArchitectureSpec,
        allowed_technologies: list[str],
        permitted_paths: list[str],
        implementation_constraints: list[str],
    ) -> GenerationResult[ArtifactBundle]:
        system_instructions = build_implementer_system_instructions()
        user_content = build_implementer_user_content(
            brief=brief,
            architecture=architecture,
            allowed_technologies=allowed_technologies,
            permitted_paths=permitted_paths,
            implementation_constraints=implementation_constraints,
        )
        return await self._provider.generate(
            system_instructions=system_instructions,
            user_content=user_content,
            response_type=ArtifactBundle,
        )

    async def repair(self, repair_request: RepairRequest) -> GenerationResult[ArtifactBundle]:
        """Perform a bounded repair attempt using exact failure context."""
        system_instructions = build_implementer_repair_system_instructions()
        user_content = build_implementer_repair_user_content(repair_request)
        return await self._provider.generate(
            system_instructions=system_instructions,
            user_content=user_content,
            response_type=ArtifactBundle,
        )

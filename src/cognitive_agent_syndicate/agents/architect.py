"""Architect agent for generating ArchitectureSpec from SystemBrief."""

from __future__ import annotations

from cognitive_agent_syndicate.prompts import (
    build_architect_system_instructions,
    build_architect_user_content,
)
from cognitive_agent_syndicate.providers.base import GenerationResult, ModelProvider
from cognitive_agent_syndicate.schemas import ArchitectureSpec, SystemBrief


class ArchitectAgent:
    """Generates a validated ArchitectureSpec from a SystemBrief."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    async def run(self, brief: SystemBrief) -> GenerationResult[ArchitectureSpec]:
        system_instructions = build_architect_system_instructions()
        user_content = build_architect_user_content(brief)
        return await self._provider.generate(
            system_instructions=system_instructions,
            user_content=user_content,
            response_type=ArchitectureSpec,
        )

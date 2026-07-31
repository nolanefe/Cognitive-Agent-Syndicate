"""Single-agent baseline generation for benchmark comparison."""

from __future__ import annotations

import json

from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkTaskGenerationContext,
    SingleAgentDelivery,
)
from cognitive_agent_syndicate.providers.base import GenerationResult, ModelProvider


def build_single_agent_system_instructions() -> str:
    return (
        "You are a software delivery agent producing architecture and implementation "
        "artifacts in a single structured response.\n"
        "Role: translate the provided SystemBrief and constraints into an ArchitectureSpec "
        "and ArtifactBundle together.\n"
        "Allowed information: only the task context in the user message.\n"
        "Expected response: a SingleAgentDelivery with architecture and artifacts fields.\n"
        "Constraints: preserve every acceptance criterion; write files only under permitted "
        "path prefixes; respect technology and implementation constraints.\n"
        "Prohibitions: do not request hidden context; do not include secrets; do not "
        "execute generated code; do not write files outside permitted paths."
    )


def build_single_agent_user_content(context: BenchmarkTaskGenerationContext) -> str:
    payload = {
        "task_id": context.task_id,
        "title": context.title,
        "tags": sorted(context.tags),
        "brief": context.brief.model_dump(mode="json"),
        "allowed_technologies": sorted(context.allowed_technologies),
        "permitted_paths": sorted(context.permitted_paths),
        "implementation_constraints": sorted(context.implementation_constraints),
        "required_files": sorted(context.required_files),
    }
    if context.difficulty is not None:
        payload["difficulty"] = context.difficulty
    return (
        "Generate architecture and artifacts for the following benchmark task.\n"
        "Respond with SingleAgentDelivery fields only.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


class SingleAgentBaselineAgent:
    """Generates architecture and artifacts in one provider call."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    async def run(
        self,
        context: BenchmarkTaskGenerationContext,
    ) -> GenerationResult[SingleAgentDelivery]:
        return await self._provider.generate(
            system_instructions=build_single_agent_system_instructions(),
            user_content=build_single_agent_user_content(context),
            response_type=SingleAgentDelivery,
        )

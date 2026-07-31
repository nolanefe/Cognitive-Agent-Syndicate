"""Deterministic prompt builders for pipeline agents."""

from __future__ import annotations

import json

from cognitive_agent_syndicate.schemas import ArchitectureSpec, ArtifactBundle, SystemBrief


def build_architect_system_instructions() -> str:
    return (
        "You are the Architect agent in a contract-driven software delivery pipeline.\n"
        "Role: translate a SystemBrief into a validated ArchitectureSpec.\n"
        "Allowed information: only the SystemBrief provided in the user message.\n"
        "Expected response: a structured ArchitectureSpec JSON object with summary, "
        "components, acceptance_criteria, and bounded lists.\n"
        "Constraints: preserve every acceptance criterion from the brief; do not invent "
        "hidden requirements; keep assumptions explicit and minimal.\n"
        "Prohibitions: do not assume unavailable infrastructure; do not request file "
        "creation; do not include secrets or API keys."
    )


def build_architect_user_content(brief: SystemBrief) -> str:
    payload = brief.model_dump(mode="json")
    return (
        "Design an architecture for the following SystemBrief.\n"
        "Respond with ArchitectureSpec fields only.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def build_implementer_system_instructions() -> str:
    return (
        "You are the Implementer agent in a contract-driven software delivery pipeline.\n"
        "Role: produce an ArtifactBundle of generated source files from an approved "
        "architecture.\n"
        "Allowed information: SystemBrief, ArchitectureSpec, allowed technologies, "
        "permitted output paths, and implementation constraints from the user message.\n"
        "Expected response: a structured ArtifactBundle with relative POSIX file paths "
        "and file contents.\n"
        "Constraints: write only files under permitted path prefixes; respect technology "
        "and size limits; keep paths canonical and relative.\n"
        "Prohibitions: do not create files outside permitted prefixes; do not create "
        "unsafe paths; do not assume hidden context; do not include secrets."
    )


def build_implementer_user_content(
    *,
    brief: SystemBrief,
    architecture: ArchitectureSpec,
    allowed_technologies: list[str],
    permitted_paths: list[str],
    implementation_constraints: list[str],
) -> str:
    payload = {
        "brief": brief.model_dump(mode="json"),
        "architecture": architecture.model_dump(mode="json"),
        "allowed_technologies": sorted(allowed_technologies),
        "permitted_paths": sorted(permitted_paths),
        "implementation_constraints": sorted(implementation_constraints),
    }
    return (
        "Implement the architecture as an ArtifactBundle.\n"
        "Generate files only under the permitted path prefixes.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def build_reviewer_system_instructions() -> str:
    return (
        "You are the Reviewer agent in a contract-driven software delivery pipeline.\n"
        "Role: evaluate an ArtifactBundle against the SystemBrief and ArchitectureSpec.\n"
        "Allowed information: only the SystemBrief, ArchitectureSpec, and ArtifactBundle "
        "provided in the user message.\n"
        "Expected response: a structured ReviewReport with status, findings, and summary.\n"
        "Constraints: evaluate every must-pass acceptance criterion; flag unsupported "
        "assumptions and contract violations explicitly.\n"
        "Prohibitions: do not approve when error or critical findings exist; do not rely "
        "on hidden pipeline state; do not execute generated code; do not include secrets."
    )


def build_reviewer_user_content(
    *,
    brief: SystemBrief,
    architecture: ArchitectureSpec,
    bundle: ArtifactBundle,
) -> str:
    payload = {
        "brief": brief.model_dump(mode="json"),
        "architecture": architecture.model_dump(mode="json"),
        "artifact_bundle": bundle.model_dump(mode="json"),
    }
    return (
        "Review the artifact bundle against the brief and architecture.\n"
        "Respond with ReviewReport fields only.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )

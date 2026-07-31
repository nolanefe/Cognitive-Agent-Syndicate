"""Deterministic prompt builders for pipeline agents."""

from __future__ import annotations

import json

from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    RepairRequest,
    SystemBrief,
)


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


def build_implementer_repair_system_instructions() -> str:
    return (
        "You are the Implementer agent performing a single bounded repair attempt.\n"
        "Role: revise an ArtifactBundle to address exact deterministic gate failures "
        "and reviewer findings.\n"
        "Allowed information: RepairRequest fields in the user message only.\n"
        "Expected response: a complete revised ArtifactBundle, not a patch description.\n"
        "Constraints: preserve the original architecture contract; make corrections "
        "traceable to specific failures; only one repair attempt is allowed.\n"
        "Prohibitions: no unrelated redesign; no changes outside permitted paths; "
        "no hidden assumptions; no secrets or API keys; do not execute generated code."
    )


def build_implementer_repair_user_content(repair_request: RepairRequest) -> str:
    payload = {
        "brief": repair_request.brief.model_dump(mode="json"),
        "architecture": repair_request.architecture.model_dump(mode="json"),
        "current_bundle": repair_request.current_bundle.model_dump(mode="json"),
        "gate_failures": [gate.model_dump(mode="json") for gate in repair_request.gate_failures],
        "reviewer_findings": [
            finding.model_dump(mode="json") for finding in repair_request.reviewer_findings
        ],
        "allowed_technologies": repair_request.allowed_technologies,
        "permitted_paths": repair_request.permitted_paths,
        "implementation_constraints": repair_request.implementation_constraints,
        "permitted_file_changes": repair_request.permitted_file_changes,
        "repair_instructions": [
            instruction.model_dump(mode="json")
            for instruction in repair_request.repair_instructions
        ],
    }
    return (
        "Perform one repair attempt on the artifact bundle.\n"
        "Address the exact gate failures and reviewer findings below.\n"
        "Return the complete revised ArtifactBundle.\n"
        "Only one repair attempt is allowed.\n\n"
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

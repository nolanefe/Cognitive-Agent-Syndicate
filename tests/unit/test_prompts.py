"""Unit tests for deterministic prompt builders."""

import json

from cognitive_agent_syndicate.prompts import (
    build_architect_system_instructions,
    build_architect_user_content,
    build_implementer_system_instructions,
    build_implementer_user_content,
    build_reviewer_system_instructions,
    build_reviewer_user_content,
)
from tests.fixtures.pipeline_fixtures import sample_architecture, sample_brief, sample_bundle


def test_architect_user_content_contains_only_brief() -> None:
    brief = sample_brief()
    content = build_architect_user_content(brief)

    payload = json.loads(content.split("\n\n", maxsplit=1)[1])
    assert set(payload.keys()) == {"title", "description", "acceptance_criteria"}
    assert "architecture" not in content.lower() or "ArchitectureSpec fields" in content
    assert "artifact_bundle" not in content
    assert "review" not in payload


def test_implementer_user_content_excludes_reviewer_data() -> None:
    brief = sample_brief()
    architecture = sample_architecture()
    content = build_implementer_user_content(
        brief=brief,
        architecture=architecture,
        allowed_technologies=["python"],
        permitted_paths=["src", "tests"],
        implementation_constraints=["no secrets"],
    )

    payload = json.loads(content.split("\n\n", maxsplit=1)[1])
    assert "artifact_bundle" not in payload
    assert "review" not in payload
    assert "brief" in payload
    assert "architecture" in payload


def test_reviewer_user_content_contains_intended_contracts_only() -> None:
    brief = sample_brief()
    architecture = sample_architecture()
    bundle = sample_bundle()
    content = build_reviewer_user_content(
        brief=brief,
        architecture=architecture,
        bundle=bundle,
    )

    payload = json.loads(content.split("\n\n", maxsplit=1)[1])
    assert set(payload.keys()) == {"brief", "architecture", "artifact_bundle"}
    assert "allowed_technologies" not in payload
    assert "permitted_paths" not in payload


def test_prompt_builders_are_deterministic() -> None:
    brief = sample_brief()
    architecture = sample_architecture()
    bundle = sample_bundle()

    assert build_architect_system_instructions() == build_architect_system_instructions()
    assert build_architect_user_content(brief) == build_architect_user_content(brief)
    assert build_implementer_system_instructions() == build_implementer_system_instructions()
    assert build_implementer_user_content(
        brief=brief,
        architecture=architecture,
        allowed_technologies=["python"],
        permitted_paths=["src"],
        implementation_constraints=["safe"],
    ) == build_implementer_user_content(
        brief=brief,
        architecture=architecture,
        allowed_technologies=["python"],
        permitted_paths=["src"],
        implementation_constraints=["safe"],
    )
    assert build_reviewer_system_instructions() == build_reviewer_system_instructions()
    assert build_reviewer_user_content(
        brief=brief, architecture=architecture, bundle=bundle
    ) == build_reviewer_user_content(brief=brief, architecture=architecture, bundle=bundle)


def test_system_instructions_define_role_and_prohibitions() -> None:
    for builder in (
        build_architect_system_instructions,
        build_implementer_system_instructions,
        build_reviewer_system_instructions,
    ):
        text = builder()
        assert "Role:" in text
        assert "Prohibitions:" in text
        assert "secrets" in text.lower()

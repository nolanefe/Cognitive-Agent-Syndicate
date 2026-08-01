"""Integration tests for OpenAI provider with injected fake client."""

from __future__ import annotations

import json

import httpx
import pytest
from openai import AuthenticationError

from cognitive_agent_syndicate.agents.architect import ArchitectAgent
from cognitive_agent_syndicate.agents.implementer import ImplementerAgent
from cognitive_agent_syndicate.agents.reviewer import ReviewerAgent
from cognitive_agent_syndicate.config import ProviderName, build_settings
from cognitive_agent_syndicate.demo import (
    MockScenario,
    canonical_url_shortener_brief,
    create_demo_provider,
)
from cognitive_agent_syndicate.orchestration.pipeline import ContractDrivenPipeline
from cognitive_agent_syndicate.providers.factory import create_model_provider
from cognitive_agent_syndicate.providers.openai_provider import OpenAIModelProvider
from cognitive_agent_syndicate.reporting.report_writer import build_run_report
from cognitive_agent_syndicate.schemas import ArchitectureSpec, ArtifactBundle, ReviewReport
from tests.fixtures.openai_provider_fixtures import (
    FakeAsyncOpenAIClient,
    FakeResponsesResource,
    build_parsed_response,
    sample_usage,
)


def _success_responses() -> list[object]:
    mock = create_demo_provider(scenario=MockScenario.SUCCESS)
    return [
        build_parsed_response(
            parsed=mock._responses[(ArchitectureSpec, None)],
            usage=sample_usage(input_tokens=10, output_tokens=5),
        ),
        build_parsed_response(
            parsed=mock._responses[(ArtifactBundle, None)],
            usage=sample_usage(input_tokens=12, output_tokens=6),
        ),
        build_parsed_response(
            parsed=mock._responses[(ReviewReport, None)],
            usage=sample_usage(input_tokens=8, output_tokens=4),
        ),
    ]


def _repair_success_responses() -> list[object]:
    mock = create_demo_provider(scenario=MockScenario.REPAIR_SUCCESS)
    architecture = mock._responses[(ArchitectureSpec, None)]
    return [
        build_parsed_response(parsed=architecture, usage=sample_usage()),
        build_parsed_response(
            parsed=mock._response_sequences[ArtifactBundle][0],
            usage=sample_usage(),
        ),
        build_parsed_response(
            parsed=mock._response_sequences[ReviewReport][0],
            usage=sample_usage(),
        ),
        build_parsed_response(
            parsed=mock._response_sequences[ArtifactBundle][1],
            usage=sample_usage(),
        ),
        build_parsed_response(
            parsed=mock._response_sequences[ReviewReport][1],
            usage=sample_usage(),
        ),
    ]


def _sequenced_client(responses: list[object]) -> FakeAsyncOpenAIClient:
    call_index = {"value": 0}
    resource = FakeResponsesResource()

    async def rotating_parse(**kwargs: object):
        resource.calls.append(type("Call", (), {"kwargs": kwargs})())
        idx = call_index["value"]
        call_index["value"] += 1
        return responses[idx]

    resource.parse = rotating_parse  # type: ignore[method-assign]
    return FakeAsyncOpenAIClient(responses=resource)


@pytest.mark.asyncio
async def test_openai_injected_provider_completes_pipeline(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    client = _sequenced_client(_success_responses())
    settings = build_settings(
        provider=ProviderName.OPENAI.value,
        model="gpt-4o-mini",
        openai_api_key="sk-test",
        artifact_output_dir="artifacts",
    )
    provider = create_model_provider(settings, client=client)
    assert isinstance(provider, OpenAIModelProvider)

    pipeline = ContractDrivenPipeline(
        architect=ArchitectAgent(provider),
        implementer=ImplementerAgent(provider),
        reviewer=ReviewerAgent(provider),
        settings=settings,
    )
    brief = canonical_url_shortener_brief()
    state = await pipeline.run(
        brief,
        allowed_technologies=["python", "pydantic"],
        permitted_paths=["pyproject.toml", "src", "tests"],
        implementation_constraints=["Keep generated files small and safe."],
        required_project_files=["pyproject.toml"],
    )

    assert state.success is True
    assert len(client.responses.calls) == 3


@pytest.mark.asyncio
async def test_provider_failure_produces_safe_pipeline_state_and_report(
    tmp_path, monkeypatch
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    response = httpx.Response(
        status_code=401,
        headers={"x-request-id": "req_fail"},
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    secret = "sk-secret-should-not-leak"
    fake = FakeResponsesResource(
        response=AuthenticationError("auth failed", response=response, body={}),
    )
    client = FakeAsyncOpenAIClient(responses=fake)
    settings = build_settings(
        provider=ProviderName.OPENAI.value,
        model="gpt-4o-mini",
        openai_api_key=secret,
        artifact_output_dir="artifacts",
    )
    provider = create_model_provider(settings, client=client)
    pipeline = ContractDrivenPipeline(
        architect=ArchitectAgent(provider),
        implementer=ImplementerAgent(provider),
        reviewer=ReviewerAgent(provider),
        settings=settings,
    )

    brief = canonical_url_shortener_brief()
    state = await pipeline.run(
        brief,
        allowed_technologies=["python"],
        permitted_paths=["pyproject.toml", "src", "tests"],
        implementation_constraints=["No secrets."],
        required_project_files=["pyproject.toml"],
    )

    assert state.success is False
    assert secret not in (state.failure_reason or "")
    assert secret not in state.model_dump_json()
    report = build_run_report(state, generated_files=[])
    report_json = json.dumps(report.model_dump(mode="json"))
    assert secret not in report_json


@pytest.mark.asyncio
async def test_bounded_repair_works_with_openai_protocol_provider(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    client = _sequenced_client(_repair_success_responses())
    settings = build_settings(
        provider=ProviderName.OPENAI.value,
        model="gpt-4o-mini",
        openai_api_key="sk-test",
        artifact_output_dir="artifacts",
        max_repair_attempts=1,
    )
    provider = create_model_provider(settings, client=client)
    pipeline = ContractDrivenPipeline(
        architect=ArchitectAgent(provider),
        implementer=ImplementerAgent(provider),
        reviewer=ReviewerAgent(provider),
        settings=settings,
    )

    state = await pipeline.run(
        canonical_url_shortener_brief(),
        allowed_technologies=["python", "pydantic"],
        permitted_paths=["pyproject.toml", "src", "tests"],
        implementation_constraints=["Keep generated files small and safe."],
        required_project_files=["pyproject.toml"],
    )

    assert state.success is True
    assert state.repair_attempted is True
    assert len(client.responses.calls) == 5

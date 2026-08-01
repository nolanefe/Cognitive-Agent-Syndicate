"""Unit tests for live provider smoke helper."""

from __future__ import annotations

import httpx
import pytest
from openai import AuthenticationError, RateLimitError

from cognitive_agent_syndicate.config import ProviderName, build_settings
from cognitive_agent_syndicate.live_validation.smoke import run_live_provider_smoke
from cognitive_agent_syndicate.schemas import SystemBrief
from tests.fixtures.openai_provider_fixtures import (
    FakeAsyncOpenAIClient,
    FakeResponsesResource,
    build_parsed_response,
)
from tests.fixtures.pipeline_fixtures import sample_brief


@pytest.mark.asyncio
async def test_smoke_success_makes_one_call() -> None:
    brief = sample_brief()
    fake_client = FakeAsyncOpenAIClient(
        responses=FakeResponsesResource(
            response=build_parsed_response(parsed=brief),
        ),
    )
    settings = build_settings(
        provider=ProviderName.OPENAI.value,
        model="gpt-test",
        openai_api_key="sk-test",
    )
    result = await run_live_provider_smoke(settings, client=fake_client)
    assert result.success is True
    assert result.provider == "openai"
    assert result.prompt_tokens is not None
    assert result.prompt_tokens >= 0
    assert len(fake_client.responses.calls) == 1
    assert fake_client.responses.calls[0].kwargs["store"] is False


@pytest.mark.asyncio
async def test_smoke_authentication_failure() -> None:
    auth_error = AuthenticationError(
        "auth failed",
        response=httpx.Response(
            status_code=401,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        ),
        body={},
    )
    fake_client = FakeAsyncOpenAIClient(responses=FakeResponsesResource(response=auth_error))
    settings = build_settings(
        provider=ProviderName.OPENAI.value,
        model="gpt-test",
        openai_api_key="sk-test",
    )
    result = await run_live_provider_smoke(settings, client=fake_client)
    assert result.success is False
    assert result.failure_category == "provider_authentication"


@pytest.mark.asyncio
async def test_smoke_rate_limit() -> None:
    rate_error = RateLimitError(
        "rate limited",
        response=httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        ),
        body={},
    )
    fake_client = FakeAsyncOpenAIClient(responses=FakeResponsesResource(response=rate_error))
    settings = build_settings(
        provider=ProviderName.OPENAI.value,
        model="gpt-test",
        openai_api_key="sk-test",
    )
    result = await run_live_provider_smoke(settings, client=fake_client)
    assert result.success is False
    assert result.failure_category == "provider_rate_limit"


@pytest.mark.asyncio
async def test_smoke_malformed_structured_output() -> None:
    fake_client = FakeAsyncOpenAIClient(
        responses=FakeResponsesResource(response=build_parsed_response(parsed=None)),
    )
    settings = build_settings(
        provider=ProviderName.OPENAI.value,
        model="gpt-test",
        openai_api_key="sk-test",
    )
    result = await run_live_provider_smoke(settings, client=fake_client)
    assert result.success is False
    assert result.failure_category == "malformed_structured_output"


@pytest.mark.asyncio
async def test_smoke_uses_system_brief_schema() -> None:
    brief = sample_brief()
    fake_client = FakeAsyncOpenAIClient(
        responses=FakeResponsesResource(
            response=build_parsed_response(parsed=brief),
        ),
    )
    settings = build_settings(
        provider=ProviderName.OPENAI.value,
        model="gpt-test",
        openai_api_key="sk-test",
    )
    await run_live_provider_smoke(settings, client=fake_client)
    assert fake_client.responses.calls[0].kwargs["text_format"] is SystemBrief

"""Unit tests for the OpenAI model provider."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from pydantic import BaseModel, Field

from cognitive_agent_syndicate.providers.errors import (
    ProviderAuthenticationError,
    ProviderCancelledResponseError,
    ProviderConnectionError,
    ProviderError,
    ProviderFailedResponseError,
    ProviderIncompleteResponseError,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderTimeoutError,
)
from cognitive_agent_syndicate.providers.openai_provider import OpenAIModelProvider
from cognitive_agent_syndicate.schemas import AcceptanceCriterion, SystemBrief
from tests.fixtures.openai_provider_fixtures import (
    FakeAsyncOpenAIClient,
    FakeResponsesResource,
    build_parsed_response,
    sample_usage,
)


class _TinySchema(BaseModel):
    title: str = Field(..., min_length=1)


def _sample_brief() -> SystemBrief:
    return SystemBrief(
        title="Sample",
        description="Offline provider test.",
        acceptance_criteria=[
            AcceptanceCriterion(id="ac-1", description="Works offline."),
        ],
    )


def _provider(
    *,
    response: Any,
    clock_values: list[float] | None = None,
) -> tuple[OpenAIModelProvider, FakeResponsesResource]:
    fake_responses = FakeResponsesResource(response=response)
    client = FakeAsyncOpenAIClient(responses=fake_responses)
    values = iter(clock_values or [0.0, 0.05])
    provider = OpenAIModelProvider(
        model="gpt-4o-mini",
        client=client,
        max_output_tokens=512,
        clock=lambda: next(values),
    )
    return provider, fake_responses


@pytest.mark.asyncio
async def test_openai_provider_success_sends_expected_request() -> None:
    parsed = _sample_brief()
    provider, fake = _provider(
        response=build_parsed_response(parsed=parsed, usage=sample_usage()),
        clock_values=[1.0, 1.02],
    )

    result = await provider.generate(
        system_instructions="System prompt.",
        user_content="User prompt.",
        response_type=SystemBrief,
    )

    assert result.response.title == "Sample"
    assert len(fake.calls) == 1
    call = fake.calls[0].kwargs
    assert call["model"] == "gpt-4o-mini"
    assert call["instructions"] == "System prompt."
    assert call["input"] == "User prompt."
    assert call["text_format"] is SystemBrief
    assert call["store"] is False
    assert call["background"] is False
    assert call["max_output_tokens"] == 512


@pytest.mark.asyncio
async def test_openai_provider_maps_usage_and_latency() -> None:
    parsed = _sample_brief()
    provider, _ = _provider(
        response=build_parsed_response(
            parsed=parsed,
            usage=sample_usage(input_tokens=11, output_tokens=7, total_tokens=18),
        ),
        clock_values=[2.0, 2.123],
    )

    result = await provider.generate(
        system_instructions="System",
        user_content="User",
        response_type=SystemBrief,
    )

    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 18
    assert result.usage.latency_ms == pytest.approx(123.0)


@pytest.mark.asyncio
async def test_openai_provider_reconciles_inconsistent_usage() -> None:
    parsed = _sample_brief()
    provider, _ = _provider(
        response=build_parsed_response(
            parsed=parsed,
            usage=sample_usage(input_tokens=4, output_tokens=2, total_tokens=99),
        ),
    )

    result = await provider.generate(
        system_instructions="System",
        user_content="User",
        response_type=SystemBrief,
    )

    assert result.usage.total_tokens == 6


@pytest.mark.asyncio
async def test_openai_provider_handles_missing_usage() -> None:
    parsed = _sample_brief()
    provider, _ = _provider(response=build_parsed_response(parsed=parsed, usage=None))

    result = await provider.generate(
        system_instructions="System",
        user_content="User",
        response_type=SystemBrief,
    )

    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert result.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_openai_provider_handles_refusal() -> None:
    provider, _ = _provider(
        response=build_parsed_response(parsed=None, refusal="Policy violation."),
    )

    with pytest.raises(ProviderRefusalError, match="refused") as exc_info:
        await provider.generate(
            system_instructions="System",
            user_content="User",
            response_type=SystemBrief,
        )

    assert exc_info.value.refusal_category == "Policy violation."


@pytest.mark.asyncio
async def test_openai_provider_handles_incomplete_response() -> None:
    provider, _ = _provider(
        response=build_parsed_response(
            parsed=_sample_brief(),
            status="incomplete",
            incomplete_reason="max_output_tokens",
        ),
    )

    with pytest.raises(ProviderIncompleteResponseError, match="max_output_tokens") as exc_info:
        await provider.generate(
            system_instructions="System",
            user_content="User",
            response_type=SystemBrief,
        )

    assert exc_info.value.incomplete_reason == "max_output_tokens"


@pytest.mark.asyncio
async def test_openai_provider_handles_failed_response() -> None:
    provider, _ = _provider(
        response=build_parsed_response(
            parsed=_sample_brief(),
            status="failed",
            error_code="server_error",
        ),
    )

    with pytest.raises(ProviderFailedResponseError, match="server_error"):
        await provider.generate(
            system_instructions="System",
            user_content="User",
            response_type=SystemBrief,
        )


@pytest.mark.asyncio
async def test_openai_provider_handles_missing_parsed_output() -> None:
    provider, _ = _provider(response=build_parsed_response(parsed=None))

    with pytest.raises(ProviderMalformedResponseError, match="missing parsed"):
        await provider.generate(
            system_instructions="System",
            user_content="User",
            response_type=SystemBrief,
        )


@pytest.mark.asyncio
async def test_openai_provider_handles_malformed_parsed_type() -> None:
    parsed = _TinySchema(title="Wrong type")
    provider, _ = _provider(response=build_parsed_response(parsed=parsed))

    with pytest.raises(ProviderMalformedResponseError, match="not SystemBrief"):
        await provider.generate(
            system_instructions="System",
            user_content="User",
            response_type=SystemBrief,
        )


def _http_response(status_code: int, request_id: str = "req_test") -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"x-request-id": request_id},
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )


def _sdk_error_cases() -> list[tuple[BaseException, type[ProviderError]]]:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return [
        (
            AuthenticationError("auth", response=_http_response(401), body={}),
            ProviderAuthenticationError,
        ),
        (
            RateLimitError("rate", response=_http_response(429), body={}),
            ProviderRateLimitError,
        ),
        (APITimeoutError(request), ProviderTimeoutError),
        (APIConnectionError(message="connection", request=request), ProviderConnectionError),
        (
            APIStatusError("status", response=_http_response(500), body={}),
            ProviderError,
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("exception", "expected_type"), _sdk_error_cases())
async def test_openai_provider_maps_sdk_errors(
    exception: BaseException,
    expected_type: type[ProviderError],
) -> None:
    provider, fake = _provider(response=exception)
    fake.response = exception

    with pytest.raises(expected_type):
        await provider.generate(
            system_instructions="System",
            user_content="User",
            response_type=SystemBrief,
        )


@pytest.mark.asyncio
async def test_openai_provider_retains_request_id_on_status_error() -> None:
    response = _http_response(500, request_id="req_safe123")
    provider, fake = _provider(
        response=APIStatusError("status", response=response, body={}),
    )
    fake.response = APIStatusError("status", response=response, body={})

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate(
            system_instructions="System",
            user_content="User",
            response_type=SystemBrief,
        )

    assert exc_info.value.request_id == "req_safe123"


@pytest.mark.asyncio
async def test_openai_provider_exception_does_not_include_api_key() -> None:
    secret = "sk-super-secret-key-value"
    response = _http_response(401, request_id="req_auth")
    provider, fake = _provider(
        response=AuthenticationError(
            f"Invalid key {secret}",
            response=response,
            body={},
        ),
    )
    fake.response = AuthenticationError(
        f"Invalid key {secret}",
        response=response,
        body={},
    )

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await provider.generate(
            system_instructions="System",
            user_content="User",
            response_type=SystemBrief,
        )

    assert secret not in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_provider_handles_cancelled_response() -> None:
    provider, _ = _provider(
        response=build_parsed_response(
            parsed=_sample_brief(),
            status="cancelled",
        ),
    )

    with pytest.raises(ProviderCancelledResponseError, match="cancelled"):
        await provider.generate(
            system_instructions="System",
            user_content="User",
            response_type=SystemBrief,
        )


@pytest.mark.asyncio
async def test_openai_provider_maps_api_response_validation_error() -> None:
    from openai import APIResponseValidationError

    response = _http_response(200, request_id="req_validate")
    provider, fake = _provider(
        response=APIResponseValidationError(response=response, body={"bad": "payload"}),
    )
    fake.response = APIResponseValidationError(response=response, body={"bad": "payload"})

    with pytest.raises(ProviderMalformedResponseError, match="SDK validation") as exc_info:
        await provider.generate(
            system_instructions="System",
            user_content="User",
            response_type=SystemBrief,
        )

    assert '{"bad": "payload"}' not in str(exc_info.value)

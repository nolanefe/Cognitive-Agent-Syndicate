"""OpenAI Responses API provider with structured Pydantic output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError as PydanticValidationError

from cognitive_agent_syndicate.orchestration.clock import MonotonicClock, default_monotonic_clock
from cognitive_agent_syndicate.providers.base import GenerationResult, T
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
from cognitive_agent_syndicate.providers.openai_types import OpenAIResponsesClient
from cognitive_agent_syndicate.schemas import UsageMetrics

if TYPE_CHECKING:
    from openai.types.responses import ParsedResponse
    from openai.types.responses.response_output_refusal import ResponseOutputRefusal
    from openai.types.responses.response_usage import ResponseUsage


@dataclass
class OpenAIModelProvider:
    """Production-conscious OpenAI provider using the Responses API."""

    model: str
    client: OpenAIResponsesClient
    max_output_tokens: int
    clock: MonotonicClock = field(default_factory=lambda: default_monotonic_clock)

    async def generate(
        self,
        *,
        system_instructions: str,
        user_content: str,
        response_type: type[T],
    ) -> GenerationResult[T]:
        from openai import (
            APIConnectionError,
            APIResponseValidationError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            RateLimitError,
        )

        start = self.clock()
        try:
            response = await self.client.responses.parse(
                model=self.model,
                instructions=system_instructions,
                input=user_content,
                text_format=response_type,
                max_output_tokens=self.max_output_tokens,
                store=False,
                background=False,
            )
        except AuthenticationError as exc:
            raise _map_authentication_error(exc) from exc
        except RateLimitError as exc:
            raise _map_rate_limit_error(exc) from exc
        except APITimeoutError as exc:
            raise _map_timeout_error(exc) from exc
        except APIConnectionError as exc:
            raise _map_connection_error(exc) from exc
        except APIResponseValidationError as exc:
            raise _map_response_validation_error(exc) from exc
        except APIStatusError as exc:
            raise _map_status_error(exc) from exc
        except ProviderError:
            raise
        except PydanticValidationError as exc:
            raise ProviderMalformedResponseError(
                f"OpenAI structured output failed validation for {response_type.__name__}."
            ) from exc
        except Exception as exc:
            raise ProviderError("OpenAI provider request failed.") from exc

        request_id = _safe_request_id(response)
        try:
            _validate_response_state(response, request_id=request_id)
            parsed = _extract_parsed_response(response, response_type, request_id=request_id)
            usage = _map_usage(response.usage, latency_ms=0.0)
        except ProviderError:
            raise
        except PydanticValidationError as exc:
            raise ProviderMalformedResponseError(
                f"OpenAI structured output failed validation for {response_type.__name__}."
            ) from exc

        latency_ms = max(0.0, (self.clock() - start) * 1000.0)
        usage = UsageMetrics(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
        )
        return GenerationResult(response=parsed, usage=usage)


def create_openai_client(*, api_key: str, timeout: float) -> Any:
    """Construct an AsyncOpenAI client with explicit timeout configuration."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=api_key, timeout=timeout)


def _validate_response_state(response: ParsedResponse[Any], *, request_id: str | None) -> None:
    if response.status == "failed":
        error_code = response.error.code if response.error is not None else "unknown"
        raise ProviderFailedResponseError(
            f"OpenAI response failed ({error_code}).",
            request_id=request_id,
        )

    if response.status == "cancelled":
        raise ProviderCancelledResponseError(
            "OpenAI response was cancelled.",
            request_id=request_id,
        )

    if response.status == "incomplete":
        reason = (
            response.incomplete_details.reason if response.incomplete_details is not None else None
        )
        message = "OpenAI response incomplete."
        if reason:
            message = f"OpenAI response incomplete ({reason})."
        raise ProviderIncompleteResponseError(
            message,
            incomplete_reason=reason,
            request_id=request_id,
        )

    refusal = _find_refusal(response)
    if refusal is not None:
        category = _refusal_category(refusal.refusal)
        message = "OpenAI model refused the request."
        if category:
            message = f"OpenAI model refused the request ({category})."
        raise ProviderRefusalError(
            message,
            refusal_category=category,
            request_id=request_id,
        )


def _extract_parsed_response(
    response: ParsedResponse[Any],
    response_type: type[T],
    *,
    request_id: str | None,
) -> T:
    parsed = response.output_parsed
    if parsed is None:
        raise ProviderMalformedResponseError(
            f"OpenAI response missing parsed {response_type.__name__} output.",
            request_id=request_id,
        )
    if not isinstance(parsed, response_type):
        raise ProviderMalformedResponseError(
            f"OpenAI parsed output is not {response_type.__name__}.",
            request_id=request_id,
        )
    return parsed


def _find_refusal(response: ParsedResponse[Any]) -> ResponseOutputRefusal | None:
    from openai.types.responses.response_output_refusal import ResponseOutputRefusal

    for output in response.output:
        if output.type != "message":
            continue
        for content in output.content:
            if isinstance(content, ResponseOutputRefusal):
                return content
            if getattr(content, "type", None) == "refusal":
                return content  # type: ignore[return-value]
    return None


def _refusal_category(refusal_text: str) -> str | None:
    stripped = refusal_text.strip()
    if not stripped:
        return None
    first_line = stripped.splitlines()[0]
    if len(first_line) > 120:
        return f"{first_line[:117]}..."
    return first_line


def _map_usage(usage: ResponseUsage | None, latency_ms: float) -> UsageMetrics:
    if usage is None:
        return UsageMetrics(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=latency_ms,
        )

    prompt_tokens = max(0, usage.input_tokens)
    completion_tokens = max(0, usage.output_tokens)
    total_tokens = max(0, usage.total_tokens)
    if total_tokens != prompt_tokens + completion_tokens:
        total_tokens = prompt_tokens + completion_tokens

    return UsageMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
    )


def _safe_request_id(response: ParsedResponse[Any]) -> str | None:
    response_id = getattr(response, "id", None)
    if isinstance(response_id, str) and response_id.strip():
        return response_id.strip()
    return None


def _map_authentication_error(exc: Exception) -> ProviderAuthenticationError:
    return ProviderAuthenticationError(
        "OpenAI authentication failed.",
        request_id=getattr(exc, "request_id", None),
        status_code=getattr(exc, "status_code", None),
    )


def _map_rate_limit_error(exc: Exception) -> ProviderRateLimitError:
    return ProviderRateLimitError(
        "OpenAI rate limit exceeded.",
        request_id=getattr(exc, "request_id", None),
        status_code=getattr(exc, "status_code", None),
    )


def _map_timeout_error(exc: Exception) -> ProviderTimeoutError:
    return ProviderTimeoutError(
        "OpenAI request timed out.",
        request_id=getattr(exc, "request_id", None),
        status_code=getattr(exc, "status_code", None),
    )


def _map_connection_error(exc: Exception) -> ProviderConnectionError:
    return ProviderConnectionError(
        "OpenAI connection failed.",
        request_id=getattr(exc, "request_id", None),
        status_code=getattr(exc, "status_code", None),
    )


def _map_status_error(exc: Exception) -> ProviderError:
    status_code = getattr(exc, "status_code", None)
    message = "OpenAI API request failed."
    if status_code is not None:
        message = f"OpenAI API request failed (HTTP {status_code})."
    return ProviderError(
        message,
        request_id=getattr(exc, "request_id", None),
        status_code=status_code,
    )


def _map_response_validation_error(exc: Exception) -> ProviderMalformedResponseError:
    return ProviderMalformedResponseError(
        "OpenAI response failed SDK validation.",
        request_id=getattr(exc, "request_id", None),
    )

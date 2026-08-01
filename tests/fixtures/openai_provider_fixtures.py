"""Shared fakes for offline OpenAI provider tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai.types.responses import ParsedResponse
from openai.types.responses.parsed_response import (
    ParsedResponseOutputMessage,
    ParsedResponseOutputText,
)
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from pydantic import BaseModel


def _default_token_detail_values(
    model_cls: type[BaseModel],
    *,
    defaults: dict[str, int],
) -> dict[str, int]:
    details: dict[str, int] = {}
    for key, value in defaults.items():
        if key in model_cls.model_fields:
            details[key] = value
    for name, field_info in model_cls.model_fields.items():
        if field_info.is_required() and name not in details:
            details[name] = 0
    return details


def build_input_tokens_details(**overrides: int) -> InputTokensDetails:
    details = _default_token_detail_values(
        InputTokensDetails,
        defaults={"cached_tokens": 0, "cache_write_tokens": 0},
    )
    details.update(overrides)
    return InputTokensDetails.model_validate(details)


def build_output_tokens_details(**overrides: int) -> OutputTokensDetails:
    details = _default_token_detail_values(
        OutputTokensDetails,
        defaults={"reasoning_tokens": 0},
    )
    details.update(overrides)
    return OutputTokensDetails.model_validate(details)


def build_response_usage(
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    total_tokens: int | None = None,
) -> ResponseUsage:
    resolved_total = total_tokens if total_tokens is not None else input_tokens + output_tokens
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=build_input_tokens_details(),
        output_tokens=output_tokens,
        output_tokens_details=build_output_tokens_details(),
        total_tokens=resolved_total,
    )


@dataclass
class FakeResponsesParseCall:
    kwargs: dict[str, Any]


@dataclass
class FakeResponsesResource:
    response: ParsedResponse[Any] | BaseException | None = None
    calls: list[FakeResponsesParseCall] = field(default_factory=list)

    async def parse(self, **kwargs: Any) -> ParsedResponse[Any]:
        self.calls.append(FakeResponsesParseCall(kwargs=kwargs))
        if isinstance(self.response, BaseException):
            raise self.response
        if self.response is None:
            raise RuntimeError("FakeResponsesResource response not configured.")
        return self.response


@dataclass
class FakeAsyncOpenAIClient:
    responses: FakeResponsesResource


def build_parsed_response(
    *,
    parsed: BaseModel | None,
    response_id: str = "resp_test123",
    status: str = "completed",
    usage: ResponseUsage | None = None,
    refusal: str | None = None,
    incomplete_reason: str | None = None,
    error_code: str | None = None,
) -> ParsedResponse[Any]:
    content: list[ParsedResponseOutputText[Any] | ResponseOutputRefusal]
    if refusal is not None:
        content = [ResponseOutputRefusal(refusal=refusal, type="refusal")]
    elif parsed is not None:
        content = [
            ParsedResponseOutputText(
                type="output_text",
                text=parsed.model_dump_json(),
                annotations=[],
                parsed=parsed,
            )
        ]
    else:
        content = [
            ParsedResponseOutputText(
                type="output_text",
                text="{}",
                annotations=[],
                parsed=None,
            )
        ]

    output = [
        ParsedResponseOutputMessage(
            id="msg_test",
            role="assistant",
            status="completed",
            type="message",
            content=content,
        )
    ]

    incomplete_details = None
    if incomplete_reason is not None:
        incomplete_details = {"reason": incomplete_reason}

    error = None
    if error_code is not None:
        error = {"code": error_code, "message": "Provider failed."}

    payload: dict[str, Any] = {
        "id": response_id,
        "created_at": 0.0,
        "model": "gpt-test",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
        "usage": usage,
        "incomplete_details": incomplete_details,
        "error": error,
    }
    return ParsedResponse.model_validate(payload)


def sample_usage(
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    total_tokens: int | None = None,
) -> ResponseUsage:
    return build_response_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )

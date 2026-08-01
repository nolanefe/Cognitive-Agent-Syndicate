"""Offline guard for OpenAI SDK request signature compatibility."""

from __future__ import annotations

import inspect

import pytest


@pytest.mark.parametrize(
    "keyword",
    [
        "model",
        "instructions",
        "input",
        "text_format",
        "max_output_tokens",
        "store",
        "background",
    ],
)
def test_openai_responses_parse_accepts_provider_keywords(keyword: str) -> None:
    pytest.importorskip("openai")
    from openai.resources.responses.responses import AsyncResponses

    signature = inspect.signature(AsyncResponses.parse)
    assert keyword in signature.parameters

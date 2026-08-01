"""Manually invoked live smoke test for the OpenAI provider."""

from __future__ import annotations

import os

import pytest

from cognitive_agent_syndicate.config import ProviderName, build_settings
from cognitive_agent_syndicate.live_validation.smoke import run_live_provider_smoke


@pytest.mark.live
@pytest.mark.asyncio
async def test_openai_provider_live_structured_output_smoke() -> None:
    if os.getenv("RUN_LIVE_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_TESTS=1 to run live provider smoke tests.")

    settings = build_settings()
    api_key_secret = settings.resolved_openai_api_key()
    if api_key_secret is None:
        pytest.skip("OPENAI_API_KEY is required for live provider smoke tests.")

    model = os.getenv("OPENAI_LIVE_MODEL", "gpt-4o-mini")
    settings = build_settings(
        provider=ProviderName.OPENAI.value,
        model=model,
        openai_api_key=api_key_secret.get_secret_value(),
    )

    result = await run_live_provider_smoke(settings)

    assert result.success is True
    assert result.model == model
    assert result.provider == ProviderName.OPENAI.value
    assert result.prompt_tokens is not None
    assert result.completion_tokens is not None
    assert result.total_tokens is not None
    assert result.latency_ms >= 0.0
    assert api_key_secret.get_secret_value() not in repr(result)

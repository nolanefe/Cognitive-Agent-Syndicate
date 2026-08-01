"""Shared pytest configuration for offline, secret-safe test runs."""

from __future__ import annotations

import pytest

_SENSITIVE_ENV_VARS = (
    "OPENAI_API_KEY",
    "API_KEY",
    "RUN_LIVE_TESTS",
    "RUN_LIVE_BENCHMARKS",
    "OPENAI_LIVE_MODEL",
)

_LIVE_TEST_PRESERVED_ENV_VARS = frozenset(
    {
        "OPENAI_API_KEY",
        "API_KEY",
        "RUN_LIVE_TESTS",
        "OPENAI_LIVE_MODEL",
    }
)


@pytest.fixture(autouse=True)
def _sanitize_provider_environment(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure normal tests never inherit live-provider secrets from the shell."""
    is_live_test = request.node.get_closest_marker("live") is not None
    for variable in _SENSITIVE_ENV_VARS:
        if is_live_test and variable in _LIVE_TEST_PRESERVED_ENV_VARS:
            continue
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture(autouse=True)
def _reset_openai_client_injection() -> None:
    """Clear test-only OpenAI client injection between tests."""
    from cognitive_agent_syndicate.providers.factory import set_openai_client_injection

    set_openai_client_injection(None)
    yield
    set_openai_client_injection(None)

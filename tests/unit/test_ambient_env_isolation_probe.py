"""Probe that autouse environment isolation clears ambient provider secrets."""

from __future__ import annotations

import os


def test_ambient_sensitive_env_is_cleared() -> None:
    assert os.getenv("RUN_LIVE_TESTS") is None
    assert os.getenv("OPENAI_API_KEY") is None
    assert os.getenv("API_KEY") is None
    assert os.getenv("RUN_LIVE_BENCHMARKS") is None
    assert os.getenv("OPENAI_LIVE_MODEL") is None

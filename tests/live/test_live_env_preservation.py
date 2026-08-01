"""Live opt-in environment checks that never touch the network."""

from __future__ import annotations

import os

import pytest


@pytest.mark.live
def test_live_explicit_opt_in_env_is_visible() -> None:
    if os.getenv("RUN_LIVE_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_TESTS=1 to run live provider smoke tests.")

    assert os.getenv("OPENAI_API_KEY") == "sk-fake-opt-in-key"
    assert os.getenv("OPENAI_LIVE_MODEL") == "gpt-fake-live-model"
    assert os.getenv("RUN_LIVE_BENCHMARKS") is None

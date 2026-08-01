"""Unit tests for benchmark progress instrumentation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cognitive_agent_syndicate.benchmarking.progress import (
    BenchmarkProgressEvent,
    BenchmarkProgressEventType,
    wrap_provider_for_progress,
)
from cognitive_agent_syndicate.benchmarking.provider_instrumentation import ProviderCallCounter
from cognitive_agent_syndicate.providers.base import GenerationResult
from cognitive_agent_syndicate.schemas import UsageMetrics
from tests.fixtures.pipeline_fixtures import sample_brief


@dataclass
class _StubProvider:
    async def generate(self, *, system_instructions: str, user_content: str, response_type: type):
        return GenerationResult(
            response=sample_brief(),
            usage=UsageMetrics(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                latency_ms=1.0,
            ),
        )


@pytest.mark.asyncio
async def test_progress_provider_wrapper_emits_events() -> None:
    events: list[BenchmarkProgressEvent] = []
    counter = ProviderCallCounter()

    def callback(event: BenchmarkProgressEvent) -> None:
        events.append(event)

    provider = wrap_provider_for_progress(
        _StubProvider(),
        counter,
        progress_callback=callback,
    )
    await provider.generate(
        system_instructions="test",
        user_content="test",
        response_type=type(sample_brief()),
    )
    event_types = [event.event_type for event in events]
    assert BenchmarkProgressEventType.PROVIDER_CALL_STARTED in event_types
    assert BenchmarkProgressEventType.PROVIDER_CALL_COMPLETED in event_types
    assert counter.count == 1

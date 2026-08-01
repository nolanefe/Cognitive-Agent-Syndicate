"""Typed progress events for benchmark execution."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from cognitive_agent_syndicate.providers.base import GenerationResult, ModelProvider, T

if TYPE_CHECKING:
    from cognitive_agent_syndicate.benchmarking.provider_instrumentation import ProviderCallCounter


class BenchmarkProgressEventType(StrEnum):
    """Benchmark lifecycle events suitable for CLI or test observers."""

    BENCHMARK_STARTED = "benchmark_started"
    TRIAL_STARTED = "trial_started"
    PROVIDER_CALL_STARTED = "provider_call_started"
    PROVIDER_CALL_COMPLETED = "provider_call_completed"
    TRIAL_COMPLETED = "trial_completed"
    TRIAL_FAILED = "trial_failed"
    REPAIR_STARTED = "repair_started"
    REPAIR_COMPLETED = "repair_completed"
    BENCHMARK_PERSISTENCE_STARTED = "benchmark_persistence_started"
    BENCHMARK_COMPLETED = "benchmark_completed"


@dataclass(frozen=True)
class BenchmarkProgressEvent:
    """Single benchmark progress notification."""

    event_type: BenchmarkProgressEventType
    trial_index: int | None = None
    total_trials: int | None = None
    task_id: str | None = None
    mode: str | None = None
    repetition: int | None = None
    provider_call_index: int | None = None
    provider_call_latency_ms: float | None = None
    trial_status: str | None = None
    failure_category: str | None = None
    repair_attempted: bool | None = None


ProgressCallback = Callable[[BenchmarkProgressEvent], None]


@dataclass
class ProgressReportingModelProvider:
    """Wrap a provider to emit progress events without altering behavior."""

    inner: ModelProvider
    counter: ProviderCallCounter
    progress_callback: ProgressCallback | None = None
    monotonic_clock: Callable[[], float] = time.monotonic

    async def generate(
        self,
        *,
        system_instructions: str,
        user_content: str,
        response_type: type[T],
    ) -> GenerationResult[T]:
        self.counter.count += 1
        call_index = self.counter.count
        if self.progress_callback is not None:
            self.progress_callback(
                BenchmarkProgressEvent(
                    event_type=BenchmarkProgressEventType.PROVIDER_CALL_STARTED,
                    provider_call_index=call_index,
                )
            )
        start = self.monotonic_clock()
        try:
            result = await self.inner.generate(
                system_instructions=system_instructions,
                user_content=user_content,
                response_type=response_type,
            )
        finally:
            latency_ms = max(0.0, (self.monotonic_clock() - start) * 1000.0)
            if self.progress_callback is not None:
                self.progress_callback(
                    BenchmarkProgressEvent(
                        event_type=BenchmarkProgressEventType.PROVIDER_CALL_COMPLETED,
                        provider_call_index=call_index,
                        provider_call_latency_ms=latency_ms,
                    )
                )
        return result


def wrap_provider_for_progress(
    provider: ModelProvider,
    counter: ProviderCallCounter,
    *,
    progress_callback: ProgressCallback | None = None,
    monotonic_clock: Callable[[], float] | None = None,
) -> ModelProvider:
    """Wrap a provider with call counting and optional progress reporting."""
    from cognitive_agent_syndicate.benchmarking.provider_instrumentation import (
        CountingModelProvider,
    )

    if isinstance(provider, ProgressReportingModelProvider):
        if provider.counter is counter and provider.progress_callback is progress_callback:
            return provider
        provider = provider.inner

    if isinstance(provider, CountingModelProvider):
        provider = provider.inner

    if progress_callback is None:
        from cognitive_agent_syndicate.benchmarking.provider_instrumentation import (
            wrap_provider_for_counting,
        )

        return wrap_provider_for_counting(provider, counter)

    clock = monotonic_clock or time.monotonic
    return ProgressReportingModelProvider(
        inner=provider,
        counter=counter,
        progress_callback=progress_callback,
        monotonic_clock=clock,
    )

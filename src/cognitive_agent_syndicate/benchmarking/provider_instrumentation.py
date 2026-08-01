"""Exact provider call counting for benchmark trials."""

from __future__ import annotations

from dataclasses import dataclass

from cognitive_agent_syndicate.providers.base import GenerationResult, ModelProvider, T


@dataclass
class ProviderCallCounter:
    """Counts attempted provider generate() calls for one trial."""

    count: int = 0


@dataclass
class CountingModelProvider:
    """Wrap a provider and count every attempted generate() call."""

    inner: ModelProvider
    counter: ProviderCallCounter

    async def generate(
        self,
        *,
        system_instructions: str,
        user_content: str,
        response_type: type[T],
    ) -> GenerationResult[T]:
        self.counter.count += 1
        return await self.inner.generate(
            system_instructions=system_instructions,
            user_content=user_content,
            response_type=response_type,
        )


def observed_provider_call_count(*counters: ProviderCallCounter) -> int:
    """Return the summed observed provider call count."""
    return sum(counter.count for counter in counters)


def wrap_provider_for_counting(
    provider: ModelProvider,
    counter: ProviderCallCounter,
) -> CountingModelProvider:
    """Wrap a provider with a shared trial-local counter."""
    if isinstance(provider, CountingModelProvider):
        if provider.counter is counter:
            return provider
        return CountingModelProvider(inner=provider.inner, counter=counter)
    return CountingModelProvider(inner=provider, counter=counter)

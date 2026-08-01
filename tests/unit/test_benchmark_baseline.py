"""Tests for benchmark baseline agent."""

from __future__ import annotations

import pytest

from cognitive_agent_syndicate.benchmarking.baseline import (
    SingleAgentBaselineAgent,
    build_single_agent_user_content,
)
from cognitive_agent_syndicate.benchmarking.dataset import load_benchmark_dataset
from cognitive_agent_syndicate.benchmarking.mock_fixtures import create_benchmark_mock_provider
from cognitive_agent_syndicate.benchmarking.schemas import BenchmarkMode, SingleAgentDelivery
from cognitive_agent_syndicate.providers.mock import MockModelProvider


@pytest.fixture
def url_task():
    dataset = load_benchmark_dataset(
        __import__("pathlib").Path("benchmarks/datasets/software_delivery_v1.json")
    )
    return next(task for task in dataset.tasks if task.task_id == "task-url-shortener")


@pytest.mark.asyncio
async def test_baseline_requests_single_agent_delivery(url_task) -> None:
    provider = create_benchmark_mock_provider(url_task, BenchmarkMode.SINGLE_AGENT)
    agent = SingleAgentBaselineAgent(provider)
    result = await agent.run(url_task.generation_context())
    assert isinstance(result.response, SingleAgentDelivery)


@pytest.mark.asyncio
async def test_baseline_no_file_writes(url_task, tmp_path) -> None:
    provider = create_benchmark_mock_provider(url_task, BenchmarkMode.SINGLE_AGENT)
    agent = SingleAgentBaselineAgent(provider)
    await agent.run(url_task.generation_context())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_baseline_returns_usage(url_task) -> None:
    provider = create_benchmark_mock_provider(url_task, BenchmarkMode.SINGLE_AGENT)
    agent = SingleAgentBaselineAgent(provider)
    result = await agent.run(url_task.generation_context())
    assert result.usage.total_tokens > 0


def test_baseline_context_excludes_notes(url_task) -> None:
    url_task.notes = "secret benchmark note"
    content = build_single_agent_user_content(url_task.generation_context())
    assert "secret benchmark note" not in content


@pytest.mark.asyncio
async def test_provider_error_categorized(url_task) -> None:
    from cognitive_agent_syndicate.benchmarking.adapters import categorize_exception
    from cognitive_agent_syndicate.providers.errors import ProviderConnectionError

    category = categorize_exception(ProviderConnectionError("offline"))
    assert category.value == "provider_connection"


@pytest.mark.asyncio
async def test_mock_provider_records_calls(url_task) -> None:
    wrapper = create_benchmark_mock_provider(url_task, BenchmarkMode.SINGLE_AGENT)
    agent = SingleAgentBaselineAgent(wrapper)
    await agent.run(url_task.generation_context())
    assert len(wrapper.inner.calls) >= 1


def test_baseline_uses_only_model_provider(url_task) -> None:
    provider = MockModelProvider()
    agent = SingleAgentBaselineAgent(provider)
    assert agent._provider is provider

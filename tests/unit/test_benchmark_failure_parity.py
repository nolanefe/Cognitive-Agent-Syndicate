"""Cross-mode failure classification parity tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_agent_syndicate.benchmarking.adapters import categorize_exception
from cognitive_agent_syndicate.benchmarking.dataset import load_benchmark_dataset
from cognitive_agent_syndicate.benchmarking.mock_fixtures import create_benchmark_mock_provider
from cognitive_agent_syndicate.benchmarking.runner import BenchmarkRunContext, run_benchmark_trial
from cognitive_agent_syndicate.benchmarking.schemas import (
    BenchmarkMode,
    TrialFailureCategory,
    TrialStatus,
)
from cognitive_agent_syndicate.config import build_settings
from cognitive_agent_syndicate.providers.errors import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderMalformedResponseError,
    ProviderTimeoutError,
)
from cognitive_agent_syndicate.providers.mock import MockModelProvider
from cognitive_agent_syndicate.schemas import ReviewStatus


class FakeClock:
    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        self._value += 0.01
        return self._value


class FailingProvider(MockModelProvider):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    async def generate(self, *, system_instructions: str, user_content: str, response_type):
        raise self._exc


_UNSAFE_REASON_MARKERS = (
    "traceback",
    "sk-",
    "api_key",
    "api key",
    "password",
    "secret",
    "token=",
    "system_instructions",
    'file "',
)


def _assert_safe_failure_reason(reason: str | None) -> None:
    assert reason is not None
    assert reason.strip()
    assert len(reason) <= 500
    lowered = reason.lower()
    for marker in _UNSAFE_REASON_MARKERS:
        assert marker not in lowered


@pytest.fixture
def dataset():
    return load_benchmark_dataset(Path("benchmarks/datasets/software_delivery_v1.json"))


async def _run_failing_trial(
    *,
    dataset,
    tmp_path,
    mode: BenchmarkMode,
    exc: Exception,
) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-url-shortener")
    context = BenchmarkRunContext(
        benchmark_id="fail-parity",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    failing = FailingProvider(exc)
    result = await run_benchmark_trial(
        task=task,
        mode=mode,
        repetition=1,
        context=context,
        generation_provider=failing,
        reviewer_provider=failing,
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
    )
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(BenchmarkMode))
async def test_provider_connection_parity_across_modes(dataset, tmp_path, mode) -> None:
    result = await _run_failing_trial(
        dataset=dataset,
        tmp_path=tmp_path,
        mode=mode,
        exc=ProviderConnectionError("connection failed"),
    )
    assert result.trial.status == TrialStatus.FAILED
    assert result.trial.failure_category == TrialFailureCategory.PROVIDER_CONNECTION
    _assert_safe_failure_reason(result.trial.failure_reason)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(BenchmarkMode))
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            ProviderAuthenticationError("authentication failed"),
            TrialFailureCategory.PROVIDER_AUTHENTICATION,
        ),
        (ProviderTimeoutError("request timed out"), TrialFailureCategory.PROVIDER_TIMEOUT),
        (
            ProviderMalformedResponseError("invalid structured response"),
            TrialFailureCategory.MALFORMED_STRUCTURED_OUTPUT,
        ),
    ],
)
async def test_typed_provider_failure_parity_across_modes(
    dataset,
    tmp_path,
    mode,
    exc,
    expected,
) -> None:
    result = await _run_failing_trial(
        dataset=dataset,
        tmp_path=tmp_path,
        mode=mode,
        exc=exc,
    )
    assert result.trial.status == TrialStatus.FAILED
    assert result.trial.failure_category == expected
    _assert_safe_failure_reason(result.trial.failure_reason)


@pytest.mark.asyncio
async def test_mock_fixture_provider_failure_parity(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-incident-summary")
    context = BenchmarkRunContext(
        benchmark_id="fail-parity",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    for mode in BenchmarkMode:
        result = await run_benchmark_trial(
            task=task,
            mode=mode,
            repetition=1,
            context=context,
            generation_provider=create_benchmark_mock_provider(task, mode),
            reviewer_provider=create_benchmark_mock_provider(task, mode),
            settings=build_settings(provider="mock"),
            trial_dir=tmp_path / "trial",
            clock=FakeClock(),
        )
        assert result.trial.status == TrialStatus.FAILED
        assert result.trial.failure_category == TrialFailureCategory.PROVIDER_CONNECTION


@pytest.mark.asyncio
async def test_reviewer_rejection_is_completed(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-support-ticket")
    context = BenchmarkRunContext(
        benchmark_id="fail-parity",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    result = await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        context=context,
        generation_provider=create_benchmark_mock_provider(task, BenchmarkMode.SINGLE_AGENT),
        reviewer_provider=create_benchmark_mock_provider(task, BenchmarkMode.SINGLE_AGENT),
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
    )
    assert result.trial.status == TrialStatus.COMPLETED
    assert result.trial.success is False
    assert result.trial.failure_category == TrialFailureCategory.REVIEWER_REJECTED
    assert result.trial.reviewer_status == ReviewStatus.REJECTED


@pytest.mark.asyncio
async def test_gate_failure_is_completed(dataset, tmp_path) -> None:
    task = next(t for t in dataset.tasks if t.task_id == "task-document-ingestion")
    context = BenchmarkRunContext(
        benchmark_id="fail-parity",
        dataset_version="v1",
        model_label="mock-model",
        reviewer_model_label="mock-model",
    )
    result = await run_benchmark_trial(
        task=task,
        mode=BenchmarkMode.SINGLE_AGENT,
        repetition=1,
        context=context,
        generation_provider=create_benchmark_mock_provider(task, BenchmarkMode.SINGLE_AGENT),
        reviewer_provider=create_benchmark_mock_provider(task, BenchmarkMode.SINGLE_AGENT),
        settings=build_settings(provider="mock"),
        trial_dir=tmp_path / "trial",
        clock=FakeClock(),
    )
    assert result.trial.status == TrialStatus.COMPLETED
    assert result.trial.failure_category == TrialFailureCategory.DETERMINISTIC_GATE_FAILED


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ProviderAuthenticationError("auth"), TrialFailureCategory.PROVIDER_AUTHENTICATION),
        (ProviderTimeoutError("timeout"), TrialFailureCategory.PROVIDER_TIMEOUT),
        (
            ProviderMalformedResponseError("bad json"),
            TrialFailureCategory.MALFORMED_STRUCTURED_OUTPUT,
        ),
    ],
)
def test_exception_category_mapping(exc: Exception, expected: TrialFailureCategory) -> None:
    assert categorize_exception(exc) == expected

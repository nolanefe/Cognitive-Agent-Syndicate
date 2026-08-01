"""Unit tests for benchmark progress reporting."""

from __future__ import annotations

from cognitive_agent_syndicate.benchmarking.progress import (
    BenchmarkProgressEvent,
    BenchmarkProgressEventType,
)
from cognitive_agent_syndicate.live_validation.progress_display import (
    LiveValidationProgressReporter,
)


def test_progress_reporter_trial_and_provider_events() -> None:
    reporter = LiveValidationProgressReporter()
    reporter(
        BenchmarkProgressEvent(
            event_type=BenchmarkProgressEventType.TRIAL_STARTED,
            trial_index=1,
            total_trials=3,
            mode="single_agent",
            repetition=1,
        )
    )
    reporter(
        BenchmarkProgressEvent(
            event_type=BenchmarkProgressEventType.PROVIDER_CALL_COMPLETED,
            provider_call_index=1,
            provider_call_latency_ms=8400.0,
        )
    )
    reporter(
        BenchmarkProgressEvent(
            event_type=BenchmarkProgressEventType.TRIAL_COMPLETED,
            trial_index=1,
            total_trials=3,
            trial_status="completed",
        )
    )
    joined = "\n".join(reporter.lines)
    assert "single_agent / repetition 1 started" in joined
    assert "provider call 1 completed in 8.4s" in joined
    assert "completed: completed" in joined
    assert "sk-" not in joined


def test_progress_reporter_failed_trial_event() -> None:
    reporter = LiveValidationProgressReporter()
    reporter(
        BenchmarkProgressEvent(
            event_type=BenchmarkProgressEventType.TRIAL_FAILED,
            trial_index=2,
            total_trials=3,
            failure_category="reviewer_rejected",
        )
    )
    assert "reviewer_rejected" in "\n".join(reporter.lines)

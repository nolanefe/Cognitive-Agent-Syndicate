"""CLI progress display for live validation benchmarks."""

from __future__ import annotations

from cognitive_agent_syndicate.benchmarking.progress import (
    BenchmarkProgressEvent,
    BenchmarkProgressEventType,
    ProgressCallback,
)


class LiveValidationProgressReporter:
    """Render concise benchmark progress to the terminal."""

    def __init__(self) -> None:
        self._current_trial_index: int | None = None
        self._current_mode: str | None = None
        self._current_repetition: int | None = None
        self.lines: list[str] = []

    def __call__(self, event: BenchmarkProgressEvent) -> None:
        if event.event_type == BenchmarkProgressEventType.TRIAL_STARTED:
            self._current_trial_index = event.trial_index
            self._current_mode = event.mode
            self._current_repetition = event.repetition
            line = (
                f"[{event.trial_index}/{event.total_trials}] "
                f"{event.mode} / repetition {event.repetition} started"
            )
            self._emit(line)
        elif event.event_type == BenchmarkProgressEventType.PROVIDER_CALL_COMPLETED:
            if event.provider_call_latency_ms is not None:
                seconds = event.provider_call_latency_ms / 1000.0
                self._emit(
                    f"      provider call {event.provider_call_index} completed in {seconds:.1f}s"
                )
        elif event.event_type in {
            BenchmarkProgressEventType.TRIAL_COMPLETED,
            BenchmarkProgressEventType.TRIAL_FAILED,
        }:
            suffix = event.trial_status or "completed"
            if event.failure_category:
                suffix = event.failure_category
            index = event.trial_index or self._current_trial_index or 0
            total = event.total_trials or "?"
            self._emit(f"[{index}/{total}] completed: {suffix}")
            self._emit("")

    def _emit(self, line: str) -> None:
        print(line, flush=True)
        self.lines.append(line)


def build_progress_callback(
    reporter: LiveValidationProgressReporter | None,
) -> ProgressCallback | None:
    """Return a progress callback when a reporter is configured."""
    if reporter is None:
        return None
    return reporter

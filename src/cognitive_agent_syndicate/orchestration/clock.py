"""Injectable monotonic clock for deterministic pipeline timing."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol


class MonotonicClock(Protocol):
    """Protocol for reading monotonic elapsed time in seconds."""

    def __call__(self) -> float:
        """Return monotonic time in seconds."""
        ...


def default_monotonic_clock() -> float:
    """Return the system monotonic clock in seconds."""
    return time.perf_counter()


MonotonicClockFactory = Callable[[], MonotonicClock]

"""Benchmark identifier validation."""

from __future__ import annotations

import re

_BENCHMARK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class InvalidBenchmarkIdError(ValueError):
    """Raised when a benchmark identifier is unsafe or invalid."""


def validate_benchmark_id(value: str) -> str:
    """Validate a single safe benchmark identifier/path segment."""
    if value is None:
        raise InvalidBenchmarkIdError("Benchmark ID must be non-empty")
    token = value.strip()
    if not token:
        raise InvalidBenchmarkIdError("Benchmark ID must be non-empty")
    if "\x00" in token:
        raise InvalidBenchmarkIdError("Benchmark ID must not contain null bytes")
    if "/" in token or "\\" in token:
        raise InvalidBenchmarkIdError("Benchmark ID must not contain path separators")
    if token in {".", ".."}:
        raise InvalidBenchmarkIdError("Benchmark ID must not be '.' or '..'")
    if not _BENCHMARK_ID_PATTERN.fullmatch(token):
        raise InvalidBenchmarkIdError("Benchmark ID must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    return token

"""Shared benchmark display and rate formatting helpers."""

from __future__ import annotations

RATE_ROUNDING_NOTE = "Displayed percentages use standard Python formatting to two decimal places."


def format_dataset_label(name: str, version: str) -> str:
    """Render dataset name and version exactly as stored."""
    return f"{name} {version}"


def format_rate_percent(rate: float | None) -> str:
    """Format a fractional rate as a percentage string."""
    if rate is None:
        return "n/a"
    return f"{rate:.2%}"


def format_success_fraction(*, successful: int, attempted: int) -> str:
    """Format success numerator and non-cancelled attempted denominator."""
    return f"{successful}/{attempted}"


def format_success_summary(*, successful: int, attempted: int, rate: float | None) -> str:
    """Format success count and rate with a shared denominator."""
    return (
        f"{format_success_fraction(successful=successful, attempted=attempted)} "
        f"success ({format_rate_percent(rate)})"
    )

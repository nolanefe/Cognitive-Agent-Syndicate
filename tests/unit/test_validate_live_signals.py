"""Regression tests for validate-live signal handling."""

from __future__ import annotations

import signal

from cognitive_agent_syndicate.validate_live_cli import (
    ValidateLiveCancellation,
    build_validate_live_signal_handler,
)


def test_double_signal_request_keeps_custom_handler_and_allows_cleanup() -> None:
    cancellation = ValidateLiveCancellation()
    handler = build_validate_live_signal_handler(cancellation)

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    try:
        handler(signal.SIGINT, None)
        assert cancellation.event.is_set()
        assert signal.getsignal(signal.SIGINT) is handler

        handler(signal.SIGINT, None)
        assert signal.getsignal(signal.SIGINT) is handler
        assert cancellation.event.is_set()
    finally:
        signal.signal(signal.SIGINT, previous)


def test_first_signal_marks_cancellation_without_sig_dfl() -> None:
    cancellation = ValidateLiveCancellation()
    handler = build_validate_live_signal_handler(cancellation)

    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, handler)
    try:
        handler(signal.SIGTERM, None)
        installed = signal.getsignal(signal.SIGTERM)
        assert installed is handler
        assert installed is not signal.SIG_DFL
        assert cancellation.event.is_set()
    finally:
        signal.signal(signal.SIGTERM, previous)

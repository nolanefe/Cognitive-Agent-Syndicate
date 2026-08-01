"""Validate-live CLI command."""

from __future__ import annotations

import asyncio
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import typer
from rich.console import Console

from cognitive_agent_syndicate.benchmarking.exit_codes import EXIT_CANCELLED, EXIT_FATAL
from cognitive_agent_syndicate.live_validation.orchestrator import run_live_validation

console = Console()


@dataclass
class ValidateLiveCancellation:
    """Thread-safe cancellation state for validate-live signal handling."""

    event: threading.Event = field(default_factory=threading.Event)
    first_message_printed: bool = False
    repeat_message_printed: bool = False

    def request(self) -> None:
        """Record a cancellation request without bypassing cleanup handlers."""
        if self.event.is_set():
            if not self.repeat_message_printed:
                console.print(
                    "[yellow]Cancellation already requested; waiting for cleanup.[/yellow]"
                )
                self.repeat_message_printed = True
            return
        self.event.set()
        if not self.first_message_printed:
            console.print(
                "[yellow]Cancellation requested; finishing current work and restoring "
                "state.[/yellow]"
            )
            self.first_message_printed = True


def build_validate_live_signal_handler(
    cancellation: ValidateLiveCancellation,
) -> Callable[[int, object | None], None]:
    """Build a signal handler that requests graceful cancellation."""

    def _request_cancel(_signum: int, _frame: object | None) -> None:
        cancellation.request()

    return _request_cancel


def register_validate_live_command(app: typer.Typer) -> None:
    """Register the validate-live command on the root Typer app."""

    @app.command(name="validate-live")
    def validate_live(
        dataset: str = typer.Option(
            "benchmarks/datasets/software_delivery_v1.json",
            "--dataset",
            help="Path to benchmark dataset JSON.",
        ),
        task_ids: str | None = typer.Option(
            None,
            "--task-ids",
            help="Comma-separated task IDs to include.",
        ),
        modes: str = typer.Option(
            "single_agent,contract_no_repair,contract_with_repair",
            "--modes",
            help="Comma-separated benchmark modes.",
        ),
        repetitions: int = typer.Option(1, "--repetitions", min=1, max=5),
        model: str | None = typer.Option(None, "--model", help="Model name (required)."),
        reviewer_model: str | None = typer.Option(
            None,
            "--reviewer-model",
            help="Optional reviewer model (defaults to generation model).",
        ),
        output_dir: str = typer.Option(
            "benchmark_results",
            "--output-dir",
            help="Relative output directory for benchmark artifacts.",
        ),
        benchmark_id: str | None = typer.Option(
            None,
            "--benchmark-id",
            help="Optional benchmark identifier.",
        ),
        pricing_file: str | None = typer.Option(
            None,
            "--pricing-file",
            help="Optional pricing JSON for cost estimation.",
        ),
        confirm_live: bool = typer.Option(
            False,
            "--confirm-live",
            help="Required confirmation flag for live validation.",
        ),
        smoke_only: bool = typer.Option(
            False,
            "--smoke-only",
            help="Run preflight and smoke test only; skip benchmark execution.",
        ),
        allow_dirty: bool = typer.Option(
            False,
            "--allow-dirty",
            help="Allow live validation when the git working tree is dirty.",
        ),
    ) -> None:
        """Run automated live validation: smoke test, plan, and benchmark."""
        cancellation = ValidateLiveCancellation()
        handler = build_validate_live_signal_handler(cancellation)

        previous_int = signal.getsignal(signal.SIGINT)
        previous_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        outcome = None
        try:
            outcome = asyncio.run(
                run_live_validation(
                    dataset=dataset,
                    task_ids=task_ids,
                    modes=modes,
                    repetitions=repetitions,
                    model=model,
                    reviewer_model=reviewer_model,
                    output_dir=output_dir,
                    benchmark_id=benchmark_id,
                    pricing_file=pricing_file,
                    confirm_live=confirm_live,
                    smoke_only=smoke_only,
                    allow_dirty=allow_dirty,
                    cancelled_check=cancellation.event.is_set,
                )
            )
        except KeyboardInterrupt:
            console.print("[yellow]Live validation cancelled.[/yellow]")
            console.print("Credential state restored.")
            raise typer.Exit(code=EXIT_CANCELLED) from None
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)

        if outcome is None:
            raise typer.Exit(code=EXIT_FATAL)

        if outcome.smoke is not None and not outcome.smoke.success:
            console.print(outcome.handoff_text or "Live smoke failed.")
            raise typer.Exit(code=EXIT_FATAL)

        if outcome.handoff_text and not outcome.smoke_only:
            console.print("")
            console.print(outcome.handoff_text)
        elif outcome.handoff_text and outcome.smoke_only:
            console.print(outcome.handoff_text)

        if outcome.cancelled:
            if outcome.results_path is not None:
                console.print(f"Partial results saved to: {outcome.results_path}")
            raise typer.Exit(code=EXIT_CANCELLED)

        if outcome.exit_code == EXIT_FATAL and outcome.handoff_text:
            _print_error(outcome.handoff_text)
            raise typer.Exit(code=EXIT_FATAL)

        raise typer.Exit(code=outcome.exit_code)


def _print_error(message: str) -> None:
    escaped = message.replace("[", "\\[")
    console.print(f"[red]{escaped}[/red]")

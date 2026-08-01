"""Live validation benchmark identifier generation."""

from __future__ import annotations

from datetime import datetime

from cognitive_agent_syndicate.benchmarking.ids import validate_benchmark_id


def generate_live_benchmark_id(
    task_ids: list[str],
    repetitions: int,
    *,
    now: datetime | None = None,
) -> str:
    """Generate a safe live-validation benchmark identifier."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    if len(task_ids) == 1:
        task_part = task_ids[0].removeprefix("task-")
    elif task_ids:
        task_part = "suite"
    else:
        task_part = "suite"
    candidate = f"live-{task_part}-r{repetitions}-{timestamp}"
    return validate_benchmark_id(candidate)

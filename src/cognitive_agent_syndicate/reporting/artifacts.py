"""Secure artifact persistence for pipeline runs."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from cognitive_agent_syndicate.orchestration.state import PipelineStage, PipelineState
from cognitive_agent_syndicate.paths import (
    canonical_path_key,
    find_path_hierarchy_collision,
    normalize_relative_posix_path,
    reject_symlink_artifact_root,
)
from cognitive_agent_syndicate.reporting.report_writer import (
    build_success_run_report_snapshot,
    write_run_reports,
)
from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    PipelineAttempt,
    ReviewReport,
    SystemBrief,
)


class ArtifactPersistenceError(Exception):
    """Raised when artifact persistence fails validation."""


def validate_generated_path_hierarchy(paths: list[str]) -> None:
    """Reject file/directory hierarchy collisions before persistence."""
    collision = find_path_hierarchy_collision(paths)
    if collision is not None:
        left, right = collision
        raise ArtifactPersistenceError(f"Path hierarchy collision between {left!r} and {right!r}.")


def _resolve_artifact_target(run_dir: Path, relative_path: str) -> Path:
    normalized = normalize_relative_posix_path(relative_path)
    artifacts_root = (run_dir / "artifacts").resolve()
    target = run_dir / "artifacts" / normalized

    if target.is_symlink():
        raise ArtifactPersistenceError(f"Symlink detected for {relative_path!r}.")

    for parent in target.parents:
        if parent.is_symlink():
            raise ArtifactPersistenceError(f"Symlink detected for {relative_path!r}.")
        if parent == artifacts_root:
            break

    resolved = target.resolve()
    if artifacts_root not in resolved.parents and resolved != artifacts_root:
        raise ArtifactPersistenceError(f"Path escape detected for {relative_path!r}.")

    return resolved


def _write_json(path: Path, payload: object) -> None:
    if path.exists():
        raise ArtifactPersistenceError(f"Refusing to overwrite existing file: {path}.")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run_contents(
    run_dir: Path,
    *,
    brief: SystemBrief,
    architecture: ArchitectureSpec,
    bundle: ArtifactBundle,
    review: ReviewReport,
) -> list[str]:
    """Write validated contracts and generated files into a run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    _write_json(run_dir / "brief.json", brief.model_dump(mode="json"))
    _write_json(run_dir / "architecture.json", architecture.model_dump(mode="json"))
    _write_json(run_dir / "review.json", review.model_dump(mode="json"))

    written_paths: list[str] = []
    seen_keys: set[str] = set()
    validate_generated_path_hierarchy([generated_file.path for generated_file in bundle.files])

    for generated_file in bundle.files:
        key = canonical_path_key(generated_file.path)
        if key in seen_keys:
            raise ArtifactPersistenceError(f"Duplicate generated path: {generated_file.path!r}.")
        seen_keys.add(key)

        target = _resolve_artifact_target(run_dir, generated_file.path)
        if target.exists():
            raise ArtifactPersistenceError(f"Refusing to overwrite existing file: {target}.")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(generated_file.content, encoding="utf-8")
        written_paths.append(generated_file.path)

    return written_paths


def _create_staging_directory(artifact_root: Path) -> Path:
    reject_symlink_artifact_root(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    staging_dir = artifact_root / f".staging-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=False, exist_ok=False)
    return staging_dir


def _finalize_staging_directory(*, staging_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise ArtifactPersistenceError(f"Run directory already exists: {final_dir}.")
    staging_dir.rename(final_dir)


def _cleanup_staging_directory(staging_dir: Path) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)


def persist_run_artifacts(
    *,
    artifact_root: Path,
    run_id: str,
    brief: SystemBrief,
    architecture: ArchitectureSpec,
    bundle: ArtifactBundle,
    review: ReviewReport,
    state: PipelineState,
    successful_attempt: PipelineAttempt,
    wall_clock_duration_ms: float,
) -> tuple[Path, list[str]]:
    """Persist a successful run atomically under the artifact root."""
    final_dir = artifact_root / run_id
    if final_dir.exists():
        raise ArtifactPersistenceError(f"Run directory already exists: {final_dir}.")

    staging_dir = _create_staging_directory(artifact_root)
    try:
        generated_files = _write_run_contents(
            staging_dir,
            brief=brief,
            architecture=architecture,
            bundle=bundle,
            review=review,
        )
        report = build_success_run_report_snapshot(
            state=state,
            successful_attempt=successful_attempt,
            generated_files=generated_files,
            wall_clock_duration_ms=wall_clock_duration_ms,
        )
        write_run_reports(
            run_dir=staging_dir,
            state=state,
            generated_files=generated_files,
            report=report,
            report_stage=PipelineStage.COMPLETED,
        )
        _finalize_staging_directory(staging_dir=staging_dir, final_dir=final_dir)
    except Exception:
        _cleanup_staging_directory(staging_dir)
        raise

    return final_dir, generated_files


def persist_failure_report(
    *,
    artifact_root: Path,
    run_id: str,
    state: PipelineState,
    generated_files: list[str] | None = None,
) -> Path:
    """Persist a minimal truthful failure report directory."""
    final_dir = artifact_root / run_id
    if final_dir.exists():
        raise ArtifactPersistenceError(f"Run directory already exists: {final_dir}.")

    staging_dir = _create_staging_directory(artifact_root)
    try:
        write_run_reports(
            run_dir=staging_dir,
            state=state,
            generated_files=generated_files or [],
        )
        _finalize_staging_directory(staging_dir=staging_dir, final_dir=final_dir)
    except Exception:
        _cleanup_staging_directory(staging_dir)
        raise

    return final_dir


def persist_run_artifacts_legacy(
    *,
    run_dir: Path,
    brief: SystemBrief,
    architecture: ArchitectureSpec,
    bundle: ArtifactBundle,
    review: ReviewReport,
) -> list[str]:
    """Persist validated contracts and generated files under an existing run directory."""
    return _write_run_contents(
        run_dir,
        brief=brief,
        architecture=architecture,
        bundle=bundle,
        review=review,
    )

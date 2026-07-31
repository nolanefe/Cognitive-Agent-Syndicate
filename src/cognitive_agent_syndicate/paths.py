"""Shared path normalization and validation utilities."""

from __future__ import annotations

import re
from pathlib import Path

_WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def normalize_relative_posix_path(path: str) -> str:
    """Normalize and validate a relative POSIX-style path."""
    if "\x00" in path:
        raise ValueError("Path must not contain null bytes")

    stripped = path.strip()
    if not stripped:
        raise ValueError("Path must be non-empty")

    normalized = stripped.replace("\\", "/")

    if normalized.startswith("//"):
        raise ValueError("Path must not be a UNC path")

    if normalized.startswith("/"):
        raise ValueError("Path must be relative")

    if _WINDOWS_DRIVE_PREFIX.match(normalized):
        raise ValueError("Path must not contain a Windows drive prefix")

    parts = normalized.split("/")
    if "" in parts:
        raise ValueError("Path must not contain empty segments")

    resolved: list[str] = []
    for part in parts:
        if part == ".":
            continue
        if part == "..":
            raise ValueError("Path must not contain '..' segments")
        resolved.append(part)

    if not resolved:
        raise ValueError("Path must be non-empty")

    return "/".join(resolved)


def canonical_path_key(path: str) -> str:
    """Return a case-folded canonical key for duplicate path detection."""
    return normalize_relative_posix_path(path).casefold()


class SymlinkArtifactRootError(ValueError):
    """Raised when an artifact output path crosses a symlink."""


def reject_symlink_artifact_root(relative_path: str | Path) -> None:
    """Reject artifact roots that are symlinks or traverse symlink components.

    Uses filesystem checks without resolving through untrusted symlinks.
    """
    path = Path(relative_path)
    if path.is_symlink():
        raise SymlinkArtifactRootError(f"Artifact output dir must not be a symlink: {path}")

    accumulated = Path(".")
    for part in path.parts:
        accumulated = accumulated / part
        if accumulated.is_symlink():
            raise SymlinkArtifactRootError(
                f"Artifact output dir crosses a symlink at {accumulated}"
            )

"""Shared path normalization and validation utilities."""

from __future__ import annotations

import re

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

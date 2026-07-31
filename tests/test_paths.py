"""Unit tests for shared path normalization utilities."""

import pytest

from cognitive_agent_syndicate.paths import canonical_path_key, normalize_relative_posix_path


def test_normalize_relative_posix_path_returns_canonical_forward_slashes() -> None:
    assert normalize_relative_posix_path("src\\main.py") == "src/main.py"
    assert normalize_relative_posix_path("  src/main.py  ") == "src/main.py"
    assert normalize_relative_posix_path("./src/main.py") == "src/main.py"


def test_normalize_relative_posix_path_rejects_posix_absolute_paths() -> None:
    with pytest.raises(ValueError, match="relative"):
        normalize_relative_posix_path("/etc/passwd")


def test_normalize_relative_posix_path_rejects_unc_paths() -> None:
    with pytest.raises(ValueError, match="UNC"):
        normalize_relative_posix_path("//server/share/file.txt")

    with pytest.raises(ValueError, match="UNC"):
        normalize_relative_posix_path("\\\\server\\share\\file.txt")


@pytest.mark.parametrize(
    "path",
    ["C:/Windows/system.ini", "C:file", "D:\\data\\file.txt"],
)
def test_normalize_relative_posix_path_rejects_windows_drive_prefixes(path: str) -> None:
    with pytest.raises(ValueError, match="Windows drive"):
        normalize_relative_posix_path(path)


@pytest.mark.parametrize(
    "path",
    ["src/../secrets.env", "src/..", ".."],
)
def test_normalize_relative_posix_path_rejects_parent_segments(path: str) -> None:
    with pytest.raises(ValueError, match="\\.\\."):
        normalize_relative_posix_path(path)


def test_normalize_relative_posix_path_rejects_empty_segments() -> None:
    with pytest.raises(ValueError, match="empty segments"):
        normalize_relative_posix_path("src//main.py")


def test_normalize_relative_posix_path_rejects_null_bytes() -> None:
    with pytest.raises(ValueError, match="null bytes"):
        normalize_relative_posix_path("src/main\x00.py")


@pytest.mark.parametrize("path", ["", "   ", "."])
def test_normalize_relative_posix_path_rejects_empty_paths(path: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_relative_posix_path(path)


def test_normalize_relative_posix_path_rejects_trailing_slash_only_prefix() -> None:
    with pytest.raises(ValueError, match="empty segments"):
        normalize_relative_posix_path("./")


def test_canonical_path_key_detects_case_only_differences() -> None:
    assert canonical_path_key("SRC/main.py") == canonical_path_key("src/main.py")


def test_canonical_path_key_detects_separator_and_prefix_variants() -> None:
    assert canonical_path_key("src\\main.py") == canonical_path_key("src/main.py")
    assert canonical_path_key("./src/main.py") == canonical_path_key("src/main.py")

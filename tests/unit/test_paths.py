"""Unit tests for artifact path protections."""

from pathlib import Path

import pytest

from cognitive_agent_syndicate.paths import SymlinkArtifactRootError, reject_symlink_artifact_root


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="Symlinks not supported")
def test_reject_symlink_artifact_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    real_root = tmp_path / "real-artifacts"
    real_root.mkdir()
    linked_root = tmp_path / "linked-artifacts"
    linked_root.symlink_to(real_root)

    with pytest.raises(SymlinkArtifactRootError):
        reject_symlink_artifact_root("linked-artifacts")


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="Symlinks not supported")
def test_reject_symlink_intermediate_parent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent)

    with pytest.raises(SymlinkArtifactRootError):
        reject_symlink_artifact_root("linked-parent/artifacts")

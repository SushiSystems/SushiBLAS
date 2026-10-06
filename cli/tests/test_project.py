# test_project.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests for the build service: an interrupted child and a build tree that must be replaced."""

from __future__ import annotations

from pathlib import Path

import pytest
import sushicore.proc

from sushiblas.services import project

K_CONFIGURE_FAILED = 9


class _Console:
    """Records every line the service prints, by kind."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def __getattr__(self, kind: str):
        """Returns a printer that records its text under *kind*."""
        return lambda text="": self.lines.append((kind, str(text)))

    def texts(self, kind: str) -> list[str]:
        """Returns the lines recorded under *kind*."""
        return [text for recorded, text in self.lines if recorded == kind]


class _Driver:
    """Stands in for the CMake driver: reports a tree to replace and runs nothing."""

    def __init__(self, console: _Console) -> None:
        self._console = console
        self.cleaned: list[Path] = []

    def cmake(self, cfg) -> str:
        """Returns the executable name the configure line starts with."""
        return "cmake"

    def needs_configure(self, build_dir: Path, generator: str, **_) -> bool:
        """Reports that the tree cannot be reused."""
        return True

    def clean_tree(self, build_dir: Path) -> None:
        """Records the tree it was asked to remove."""
        self._console.info(f"Removing {build_dir}...")
        self.cleaned.append(build_dir)

    def configure(self, args, cwd: Path, env=None) -> int:
        """Fails, so the build stops before it compiles."""
        return K_CONFIGURE_FAILED


class _Config:
    """Holds the few settings the build reads before it configures."""

    generator = "Ninja"
    is_windows = False

    def runtime_dir(self, root: Path) -> Path:
        """Returns the sibling checkout the test wrote."""
        return root.parent / "sushiruntime"

    def resolved_compiler(self, root: Path) -> str:
        """Returns a compiler name; nothing runs it."""
        return "clang++"

    def resolved_vcpkg(self, root: Path) -> str:
        """Returns no vcpkg tree."""
        return ""


@pytest.fixture
def service(tmp_path, monkeypatch):
    """Returns the recording console and driver the build service runs against."""
    root = tmp_path / "sushiblas"
    (root / "build").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("")
    (tmp_path / "sushiruntime").mkdir()
    (tmp_path / "sushiruntime" / "CMakeLists.txt").write_text("")
    console = _Console()
    driver = _Driver(console)
    monkeypatch.setattr(project, "console", console)
    monkeypatch.setattr(project, "_DRIVER", driver)
    monkeypatch.setattr(project, "find_project_root", lambda: root)
    monkeypatch.setattr(project, "load_config", _Config)
    monkeypatch.setattr(project, "load_build_env", lambda cfg, build_dir: {})
    return console, driver, root


def test_build_says_it_replaces_a_tree_it_cannot_reuse(service):
    """A tree configured for another build type is removed out loud, not silently."""
    console, driver, root = service
    assert project.build(project.BuildType.debug) == K_CONFIGURE_FAILED
    assert driver.cleaned == [root / "build"]
    said = console.texts("info")
    reason = next(text for text in said if "empty tree" in text)
    assert said.index(reason) < said.index(f"Removing {root / 'build'}...")


def test_build_removes_nothing_when_there_is_no_tree(service):
    """A first build has no tree to announce or remove."""
    console, driver, root = service
    (root / "build").rmdir()
    assert project.build(project.BuildType.debug) == K_CONFIGURE_FAILED
    assert driver.cleaned == []
    assert not [text for text in console.texts("info") if "empty tree" in text]


def test_an_interrupted_child_ends_in_exit_code_130(tmp_path, monkeypatch):
    """Ctrl+C during a child process becomes a warning and 130, not a traceback."""

    def interrupt(*args, **kwargs):
        """Raises what Ctrl+C raises while the child runs."""
        raise KeyboardInterrupt

    console = _Console()
    monkeypatch.setattr(project._RUNNER, "_console", console)
    monkeypatch.setattr(sushicore.proc.subprocess, "run", interrupt)
    assert project._RUNNER.run(["some-tool"], tmp_path, {}) == 130
    assert console.texts("warn") == ["Interrupted."]

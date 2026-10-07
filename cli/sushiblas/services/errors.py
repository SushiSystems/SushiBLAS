# errors.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Declares the errors the SushiBLAS services raise, each carrying its exit code."""

from __future__ import annotations

from pathlib import Path

K_EXIT_FAILURE = 1
K_EXIT_USAGE = 2


class ProjectError(Exception):
    """Base of every error a SushiBLAS service raises.

    Args:
        message: The one line the command prints.
        exit_code: The code the command exits with.
    """

    def __init__(self, message: str, exit_code: int = K_EXIT_FAILURE) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class InvalidDefineError(ProjectError):
    """Reports a -D entry that is not VAR=VALUE."""

    def __init__(self, entry: str) -> None:
        super().__init__(
            f"-D expects VAR=VALUE, got {entry!r}. Use VAR= to clear a variable.",
            K_EXIT_USAGE,
        )


class InvalidDeviceError(ProjectError):
    """Reports a --device value that names no device and is not a selector passthrough."""

    def __init__(self, device: str, names: list[str]) -> None:
        super().__init__(
            f"--device expects {', '.join(names)} or selector:<text>, got {device!r}.",
            K_EXIT_USAGE,
        )


class RuntimeMissingError(ProjectError):
    """Reports a SushiRuntime sibling checkout that holds no CMakeLists.txt."""

    def __init__(self, runtime: Path) -> None:
        super().__init__(
            f"SushiRuntime sibling not found at: {runtime}\n"
            "  - clone it next to this repo, or\n"
            "  - set sushiruntime_dir in cli/config.local.toml, or SUSHIRUNTIME_DIR in the env.",
        )


class BuildTreeMissingError(ProjectError):
    """Reports a command that needs a build tree run before any build."""

    def __init__(self) -> None:
        super().__init__("build/ not found. Run `sb build` first.")


class TreeNotRemovedError(ProjectError):
    """Reports a directory that still exists after its removal."""

    def __init__(self, tree: Path) -> None:
        super().__init__(f"{tree} could not be removed; a file in it may be in use.")


class SuiteOptionError(ProjectError):
    """Reports a ctest option given to a suite that runs no ctest."""

    def __init__(self) -> None:
        super().__init__(
            "--filter and --repeat do not apply to the package suite.",
            K_EXIT_USAGE,
        )


class ExecutableNotFoundError(ProjectError):
    """Reports a run target that matches no built executable."""

    def __init__(self, description: str) -> None:
        super().__init__(f"{description} not found.")


class StepFailedError(ProjectError):
    """Reports a child process that exited non-zero, with that process's exit code."""

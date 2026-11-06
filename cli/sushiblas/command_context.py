# command_context.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Holds what the `sb` commands print through and run on, and the context a run uses."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import console
from .config import find_project_root
from .services.project import ProjectService
from .services.project_factory import create_project_service


@dataclass(frozen=True, slots=True)
class CommandContext:
    """The three things a command reaches outside its own arguments.

    Args:
        service_factory: Builds the project service; called when a command needs it, so a
            command that fails on its arguments builds none.
        console: Receives the headers, the success lines and the errors a command prints.
        project_root: Returns the checkout root; called when a command runs.
    """

    service_factory: Callable[[], ProjectService]
    console: Any
    project_root: Callable[[], Path]


def default_context() -> CommandContext:
    """Returns the context a real `sb` run uses: the real service, console and checkout lookup."""
    return CommandContext(
        service_factory=lambda: create_project_service(console.current()),
        console=console,
        project_root=find_project_root,
    )


_current: ContextVar[CommandContext | None] = ContextVar("sb_command_context", default=None)


@contextlib.contextmanager
def using(context: CommandContext) -> Iterator[None]:
    """Makes *context* the one every command run inside the block uses, then restores the last."""
    token = _current.set(context)
    try:
        yield
    finally:
        _current.reset(token)


def context_of_run() -> CommandContext:
    """Returns the context set by :func:`using`, or the default context of a real `sb` run."""
    current = _current.get()
    return default_context() if current is None else current

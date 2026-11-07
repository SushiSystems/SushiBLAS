# test_console_lazy.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests that the console is built on first use and that building it needs no checkout.

See cli/tests/README.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sushicore.cli_console import LazyConsole

pytestmark = pytest.mark.small

K_CLI_DIR = Path("checkout") / "cli"


def _outside_a_checkout() -> Path:
    """Reports what `config_dir` reports outside a checkout."""
    raise SystemExit("Not inside a SushiBLAS project")


@pytest.mark.parametrize("config_dir", [lambda: K_CLI_DIR, _outside_a_checkout],
                         ids=["inside", "outside"])
def test_a_console_is_not_built_by_making_its_holder(config_dir):
    """Making the holder neither reads a file nor asks the checkout, inside a project or outside one."""
    assert LazyConsole(config_dir).built is False


def test_outside_a_checkout_there_is_no_config_file_to_layer():
    """A holder whose config folder is missing layers nothing and does not raise."""
    assert list(LazyConsole(_outside_a_checkout).sources()) == []


def test_the_first_print_outside_a_checkout_builds_a_console_without_config():
    """`console.info(...)` outside a project builds a config-less console and does not crash."""
    lazy = LazyConsole(_outside_a_checkout)
    lazy.get().info("hello from outside a checkout")
    assert lazy.built is True


def test_inside_a_checkout_the_console_layers_the_two_config_files_of_the_cli_folder():
    """Inside a checkout the console layers config.toml and config.local.toml of the cli folder."""
    assert list(LazyConsole(lambda: K_CLI_DIR).sources()) == [
        K_CLI_DIR / "config.toml", K_CLI_DIR / "config.local.toml"]


def test_the_raw_console_attribute_resolves_on_first_use():
    """The raw Rich `console` attribute resolves lazily too."""
    assert LazyConsole(_outside_a_checkout).attribute("console") is not None


def test_an_unknown_attribute_raises_attribute_error():
    """An attribute the console does not have is an AttributeError, as a module attribute must be."""
    with pytest.raises(AttributeError):
        LazyConsole(_outside_a_checkout).attribute("does_not_exist")

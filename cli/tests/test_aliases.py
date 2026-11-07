# test_aliases.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for the old spellings sb still answers to."""

from __future__ import annotations

from datetime import date

import pytest

from sushiblas.aliases import K_ALIASES

K_OLD_SPELLINGS = ["sb doxygen"]
K_LAST_DAY = date(2027, 3, 25)
K_DOCS_EXIT_CODE = 7


@pytest.mark.medium
@pytest.mark.parametrize("word", ["docs", "doxygen"])
def test_the_old_word_runs_the_new_command(commands, word):
    """`sb doxygen` reaches the same service call as `sb docs`."""
    commands.runner.script(K_DOCS_EXIT_CODE)
    assert commands.invoke(word).exit_code == K_DOCS_EXIT_CODE
    assert [command.argv[1:] for command in commands.runner.commands] == [
        (".config/doxygen/Doxyfile",)]


@pytest.mark.small
@pytest.mark.parametrize("old", K_OLD_SPELLINGS)
def test_every_row_replaces_a_sb_command_with_another_and_shares_the_date(old):
    """Each row is a `sb` spelling that points at another `sb` spelling and leaves on one day."""
    alias = K_ALIASES.alias_for(old)
    assert alias.old.startswith("sb ") and alias.new.startswith("sb ")
    assert alias.new != alias.old
    assert alias.removed_after == K_LAST_DAY


@pytest.mark.small
@pytest.mark.parametrize(("today", "expired"), [(K_LAST_DAY, False), (date(2027, 3, 26), True)])
def test_a_row_expires_the_day_after_its_last_day(today, expired):
    """A row past its date is listed as expired, which is the prompt to delete the old word."""
    assert bool(K_ALIASES.expired(today)) is expired

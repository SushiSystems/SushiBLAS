# test_aliases.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests for the old spellings sb still answers to."""

from __future__ import annotations

from datetime import date

import pytest

from sushiblas.aliases import K_ALIASES

pytestmark = pytest.mark.small

K_OLD_SPELLINGS = {"sb doxygen": "sb docs"}
K_DOCS_EXIT_CODE = 7


@pytest.mark.parametrize("word", ["docs", "doxygen"])
def test_the_old_word_runs_the_new_command(commands, word):
    """`sb doxygen` reaches the same service call as `sb docs`."""
    commands.runner.script(K_DOCS_EXIT_CODE)
    assert commands.invoke(word).exit_code == K_DOCS_EXIT_CODE
    assert [command.argv[1:] for command in commands.runner.commands] == [
        (".config/doxygen/Doxyfile",)]


@pytest.mark.parametrize("old", list(K_OLD_SPELLINGS))
def test_every_row_names_its_replacement_and_the_shared_date(old):
    """Each old spelling points at its new one and leaves on 2027-03-25."""
    alias = K_ALIASES.alias_for(old)
    assert alias.new == K_OLD_SPELLINGS[old]
    assert alias.removed_after == date(2027, 3, 25)


def test_no_row_has_expired_today():
    """A row past its date fails here, which is the prompt to delete the old word."""
    assert K_ALIASES.expired(date.today()) == []

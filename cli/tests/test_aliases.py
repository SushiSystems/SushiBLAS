# test_aliases.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests for the old spellings sb still answers to."""

from __future__ import annotations

import json
from datetime import date

import pytest
from typer.testing import CliRunner

from sushiblas import cli
from sushiblas.aliases import K_ALIASES

K_OLD_SPELLINGS = {"sb doxygen": "sb docs"}
K_DOCS_EXIT_CODE = 7


class _DocsService:
    """Stands in for the project service; docs returns a fixed code."""

    def docs(self) -> int:
        """Returns K_DOCS_EXIT_CODE."""
        return K_DOCS_EXIT_CODE


@pytest.mark.parametrize("word", ["docs", "doxygen"])
def test_the_old_word_runs_the_new_command(monkeypatch, word):
    """`sb doxygen` reaches the same service call as `sb docs`."""
    monkeypatch.setattr(cli, "_service", _DocsService)
    assert CliRunner().invoke(cli.app, [word]).exit_code == K_DOCS_EXIT_CODE


def test_the_old_word_is_absent_from_help():
    """The help screen lists docs and leaves doxygen out."""
    listed = [line.split()[0] for line in CliRunner().invoke(cli.app, ["--help"]).output.splitlines()
              if line.startswith("  ") and not line.startswith("   ")]
    assert "docs" in listed
    assert "doxygen" not in listed


def test_the_old_word_is_absent_from_describe():
    """The catalogue lists docs and leaves doxygen out."""
    document = json.loads(CliRunner().invoke(cli.app, ["--describe"]).output)
    names = [command["name"] for command in document["commands"]]
    assert "docs" in names
    assert "doxygen" not in names


@pytest.mark.parametrize("old", list(K_OLD_SPELLINGS))
def test_every_row_names_its_replacement_and_the_shared_date(old):
    """Each old spelling points at its new one and leaves on 2027-03-25."""
    alias = K_ALIASES.alias_for(old)
    assert alias.new == K_OLD_SPELLINGS[old]
    assert alias.removed_after == date(2027, 3, 25)


def test_no_row_has_expired_today():
    """A row past its date fails here, which is the prompt to delete the old word."""
    assert K_ALIASES.expired(date.today()) == []

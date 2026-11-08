# check_raw_colours.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Checks that no string of the sb package names a raw colour; output names a theme style.

A raw colour such as ``[green]`` bypasses the theme sushicore applies, so a user's theme cannot
change it. Every string literal of ``cli/sushiblas`` is read, f-string parts included.

Usage: python tools/styling/check_raw_colours.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402

#: The package whose strings are read, below the repository root.
K_PACKAGE_FOLDER = Path("cli") / "sushiblas"
#: An opening or closing Rich markup tag of a colour name, which the theme does not map.
K_RAW_COLOUR = re.compile(r"\[/?(green|red|yellow|cyan|blue|dim)\b")


@dataclass(frozen=True, slots=True)
class PythonSource:
    """One Python file of the package: where it is and what it holds."""

    path: Path
    text: str


def collect(paths: list[Path]) -> Iterable[PythonSource]:
    """Returns every Python file of the sb package under the repository root *paths* names."""
    package = paths[0] / K_PACKAGE_FOLDER
    return [PythonSource(path, path.read_text(encoding="utf-8"))
            for path in sorted(package.rglob("*.py"))]


def rule_raw_colour_strings(source: PythonSource) -> Iterator[Issue]:
    """Reports each string literal that opens or closes a raw colour tag, or a file that does not parse."""
    try:
        tree = ast.parse(source.text)
    except SyntaxError as error:
        yield Issue(source.path, error.lineno or 1, f"cannot be parsed: {error.msg}")
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            match = K_RAW_COLOUR.search(node.value)
            if match:
                yield Issue(source.path, node.lineno,
                            f"names the raw colour '{match.group(0)}]'; use a theme style")


K_CHECKER = Checker(
    description="Checks that no string of the sb package names a raw colour instead of a theme style.",
    collect=collect,
    rules={"rule_raw_colour_strings": rule_raw_colour_strings},
)


if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

# check_version.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks every copy of the version against project() in the root CMakeLists.txt.

Usage: python tools/versioning/check_version.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402

K_SOURCE_FILE = "CMakeLists.txt"
K_SOURCE_COMMENT = re.compile(r"^[ \t]*#.*$", re.MULTILINE)
K_SOURCE_VERSION = re.compile(
    r"\b(?i:project)\s*\(\s*\w+[^)]*?\bVERSION\s+(\d+)\.(\d+)\.(\d+)(?![.\d])"
)
K_SECTION_HEADER = re.compile(r"^\s*\[\[?([^\]]+)\]\]?\s*(?:#.*)?$")


@dataclass(frozen=True, slots=True)
class Version:
    """Holds the three numbers of the version the source of truth states."""

    major: int
    minor: int
    patch: int

    @property
    def dotted(self) -> str:
        """Returns the version as `major.minor.patch`."""
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds one repository root, the version project() states, when it states one, and its line."""

    root: Path
    version: Version | None
    source_line: int


@dataclass(frozen=True, slots=True)
class Copy:
    """Describes one place a file repeats the version, or one part of it.

    Attributes:
        pattern: Group 1 is the value the file states.
        section: The TOML table the line sits in; None accepts any line.
        prefix: A prefix the stated value may carry before the expected one.
    """

    name: str
    pattern: re.Pattern[str]
    expected: Callable[[Version], str]
    section: str | None = None
    prefix: str = ""


def _dotted(version: Version) -> str:
    """Returns the whole version, the value most copies state."""
    return version.dotted


K_HEADER_FILE = "modules/foundation/core/include/SushiBLAS/core/version.hpp"
K_TOML_VERSION = re.compile(r"""^\s*version\s*=\s*["']([^"']*)["']""")
K_HEADER_COPIES = (
    Copy(
        "SUSHIBLAS_VERSION_STRING",
        re.compile(r'^\s*#\s*define\s+SUSHIBLAS_VERSION_STRING\s+"([^"]*)"'),
        _dotted,
    ),
    Copy(
        "SUSHIBLAS_VERSION_MAJOR",
        re.compile(r"^\s*#\s*define\s+SUSHIBLAS_VERSION_MAJOR\s+(\S+)"),
        lambda version: str(version.major),
    ),
    Copy(
        "SUSHIBLAS_VERSION_MINOR",
        re.compile(r"^\s*#\s*define\s+SUSHIBLAS_VERSION_MINOR\s+(\S+)"),
        lambda version: str(version.minor),
    ),
    Copy(
        "SUSHIBLAS_VERSION_PATCH",
        re.compile(r"^\s*#\s*define\s+SUSHIBLAS_VERSION_PATCH\s+(\S+)"),
        lambda version: str(version.patch),
    ),
)
K_DOXYFILE = ".config/doxygen/Doxyfile"
K_DOXYFILE_COPIES = (
    Copy(
        "PROJECT_NUMBER",
        re.compile(r'^\s*PROJECT_NUMBER\s*=\s*"?([^"]*?)"?\s*$'),
        _dotted,
        prefix="v",
    ),
)
K_CLI_PACKAGE_FILE = "cli/pyproject.toml"
K_CLI_PACKAGE_COPIES = (
    Copy("[project] version", K_TOML_VERSION, _dotted, section="project"),
)
K_MODULE_MANIFEST_FILE = "sushi-module.toml"
K_MODULE_MANIFEST_COPIES = (
    Copy("[module] version", K_TOML_VERSION, _dotted, section="module"),
)
K_README_FILE = "README.md"
K_README_COPIES = (
    Copy("SushiBLAS x.y.z", re.compile(r"\bSushiBLAS\s+(v?\d+\.\d+\.\d+)\b"), _dotted, prefix="v"),
)
K_CHANGELOG_FILE = "docs/reference/CHANGELOG.md"
K_CHANGELOG_COPIES = (
    Copy(
        "release heading",
        re.compile(r"^## v(\d+\.\d+\.\d+)(?:-[0-9A-Za-z.]+)? — "),
        _dotted,
    ),
)


def read_source_version(root: Path) -> tuple[Version | None, int]:
    """Returns the version project() states and its line, or None and line 1 without one."""
    source = root / K_SOURCE_FILE
    if not source.is_file():
        return None, 1
    text = K_SOURCE_COMMENT.sub("", source.read_text(encoding="utf-8", errors="replace"))
    match = K_SOURCE_VERSION.search(text)
    if match is None:
        return None, 1
    version = Version(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return version, text.count("\n", 0, match.start(1)) + 1


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given repository folder with the version its source of truth states."""
    return [Repository(path, *read_source_version(path)) for path in paths if path.is_dir()]


def rule_source_version(repository: Repository) -> Iterator[Issue]:
    """Yields an issue when the root CMakeLists.txt is missing or states no version."""
    source = repository.root / K_SOURCE_FILE
    if not source.is_file():
        yield Issue(source, 1, "file is missing; project() in it is the version's source of truth")
    elif repository.version is None:
        yield Issue(source, 1, "no project(<Name> VERSION x.y.z); it is the source of truth")


def _stated_values(path: Path, copy: Copy) -> Iterator[tuple[int, str]]:
    """Yields the line and value of each place a file states one copy."""
    section: str | None = None
    text = path.read_text(encoding="utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), start=1):
        header = K_SECTION_HEADER.match(line)
        if header:
            section = header.group(1).strip()
        if copy.section is not None and section != copy.section:
            continue
        for match in copy.pattern.finditer(line):
            yield number, match.group(1)


def _copy_issues(path: Path, copy: Copy, repository: Repository, required: bool) -> Iterator[Issue]:
    """Yields an issue for each stated value that disagrees, or for a required value not stated."""
    expected = copy.expected(repository.version)
    source = f"{K_SOURCE_FILE}:{repository.source_line}"
    stated = list(_stated_values(path, copy))
    if required and not stated:
        yield Issue(path, 1, f"states no {copy.name}; {source} expects {expected}")
    for number, value in stated:
        if value not in (expected, copy.prefix + expected):
            yield Issue(path, number, f"{copy.name} is '{value}', {source} expects {expected}")


def _copy_rule(
    file: str, copies: tuple[Copy, ...], required: bool
) -> Callable[[Repository], Iterator[Issue]]:
    """Returns a rule that checks one file's copies, and its presence when the file is required."""
    def apply(repository: Repository) -> Iterator[Issue]:
        if repository.version is None:
            return
        path = repository.root / file
        if not path.is_file():
            if required:
                yield Issue(path, 1, f"file is missing; it must state {repository.version.dotted}")
            return
        for copy in copies:
            yield from _copy_issues(path, copy, repository, required)
    apply.__doc__ = f"Yields an issue for each version {file} states that disagrees with project()."
    return apply


rule_version_header = _copy_rule(K_HEADER_FILE, K_HEADER_COPIES, required=True)
rule_doxyfile = _copy_rule(K_DOXYFILE, K_DOXYFILE_COPIES, required=True)
rule_cli_package = _copy_rule(K_CLI_PACKAGE_FILE, K_CLI_PACKAGE_COPIES, required=True)
rule_module_manifest = _copy_rule(K_MODULE_MANIFEST_FILE, K_MODULE_MANIFEST_COPIES, required=False)
rule_readme = _copy_rule(K_README_FILE, K_README_COPIES, required=False)
rule_changelog = _copy_rule(K_CHANGELOG_FILE, K_CHANGELOG_COPIES, required=False)

K_CHECKER = Checker(
    description="Checks every copy of the version against project() in the root CMakeLists.txt.",
    collect=collect,
    rules={
        "rule_source_version": rule_source_version,
        "rule_version_header": rule_version_header,
        "rule_doxyfile": rule_doxyfile,
        "rule_cli_package": rule_cli_package,
        "rule_module_manifest": rule_module_manifest,
        "rule_readme": rule_readme,
        "rule_changelog": rule_changelog,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

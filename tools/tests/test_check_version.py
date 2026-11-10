# test_check_version.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the version checker: agreement, each disagreeing copy and a missing file."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Issue, run  # noqa: E402
from versioning.check_version import (  # noqa: E402
    K_CHANGELOG_COPIES,
    K_CHECKER,
    K_CLI_PACKAGE_COPIES,
    K_DOXYFILE_COPIES,
    K_HEADER_COPIES,
    K_MODULE_MANIFEST_COPIES,
    K_README_COPIES,
    Repository,
    Version,
    collect,
    file_copy_issues_in,
    read_source_version_text,
    rule_source_version,
)

K_CMAKE = "CMakeLists.txt"
K_HEADER = "modules/foundation/core/include/SushiBLAS/core/version.hpp"
K_DOXYFILE = ".config/doxygen/Doxyfile"
K_PYPROJECT = "cli/pyproject.toml"
K_MANIFEST = "sushi-module.toml"
K_README = "README.md"
K_CHANGELOG = "docs/reference/CHANGELOG.md"
K_SOURCE = "CMakeLists.txt:3 expects 1.2.3"
K_PROJECT_CALL = "cmake_minimum_required(VERSION 3.20)\n\nproject(SushiBLAS VERSION 1.2.3 LANGUAGES CXX C)\n"

K_HEADER_TEXT = (
    "#define SUSHIBLAS_VERSION_MAJOR 1\n"
    "#define SUSHIBLAS_VERSION_MINOR 2\n"
    "#define SUSHIBLAS_VERSION_PATCH 3\n"
    '#define SUSHIBLAS_VERSION_STRING "1.2.3"\n'
)
K_DOXYFILE_TEXT = 'PROJECT_NAME           = "SushiBLAS"\nPROJECT_NUMBER         = "1.2.3"\n'
K_PYPROJECT_TEXT = (
    '[build-system]\nrequires = ["setuptools>=61.0"]\n\n'
    '[project]\nname = "x"\nversion = "1.2.3"\n'
)
K_README_TEXT = "# SushiBLAS\n\nA tensor library.\n"
K_AGREEING = {
    K_CMAKE: K_PROJECT_CALL,
    K_HEADER: K_HEADER_TEXT,
    K_DOXYFILE: K_DOXYFILE_TEXT,
    K_PYPROJECT: K_PYPROJECT_TEXT,
    K_MANIFEST: '[module]\nname = "sushiblas"\n',
    K_README: K_README_TEXT,
}


def _copy_issues(path: str, text: str | None, copies, required: bool, version: Version = Version(1, 2, 3),
                 source_line: int = 3) -> list[Issue]:
    """Returns the issues of one file's text against a project() version stated at source_line."""
    repository = Repository(Path("."), version, source_line)
    return list(file_copy_issues_in(Path(path), text, copies, repository, required))


def _located(issues: list[Issue]) -> list[tuple[int, str]]:
    """Returns the line and message of each issue."""
    return [(issue.line, issue.message) for issue in issues]


def _repository(folder: str, changes: dict[str, str | None] | None = None) -> Repository:
    """Returns the agreeing repository at folder with files replaced, or removed when None."""
    root = Path(folder)
    for name, text in {**K_AGREEING, **(changes or {})}.items():
        if text is None:
            continue
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return next(iter(collect([root])))


class TestRegistration:
    """Checks the rule table."""

    K_TEST_SIZE = "small"

    def test_is_registered(self) -> None:
        """Lists the source rule and one rule per file that repeats the version."""
        assert list(K_CHECKER.rules) == [
            "rule_source_version",
            "rule_version_header",
            "rule_doxyfile",
            "rule_cli_package",
            "rule_module_manifest",
            "rule_readme",
            "rule_changelog",
        ]


class TestReadSourceVersion:
    """Checks the reader of project() on text."""

    K_TEST_SIZE = "small"

    def test_reads_the_version_and_its_line(self) -> None:
        """Reads the version from project(), not from cmake_minimum_required, with its line."""
        assert read_source_version_text(K_PROJECT_CALL) == (Version(1, 2, 3), 3)

    def test_reads_a_project_call_over_several_lines(self) -> None:
        """Reads a version that sits on a later line of the project() call."""
        text = "project(\n    SushiBLAS\n    VERSION 1.2.3\n    LANGUAGES CXX)\n"
        assert read_source_version_text(text) == (Version(1, 2, 3), 3)

    def test_skips_a_commented_project_call(self) -> None:
        """Reads the version from project(), not from a comment line above it."""
        assert read_source_version_text("# project(Old VERSION 9.9.9)\n" + K_PROJECT_CALL) == (Version(1, 2, 3), 4)

    @pytest.mark.parametrize("text", [
        pytest.param("project(SushiBLAS LANGUAGES CXX)\n", id="no-version"),
        pytest.param("project(SushiBLAS VERSION 1.2.3.4)\n", id="four-numbers"),
    ])
    def test_reads_no_version_from_an_unreadable_project_call(self, text: str) -> None:
        """Returns no version for a project() call that states none or states four numbers."""
        assert read_source_version_text(text) == (None, 1)


class TestAgreement:
    """Checks that agreeing copies report nothing."""

    K_TEST_SIZE = "small"

    @pytest.mark.parametrize("path, text, copies", [
        pytest.param(K_HEADER, K_HEADER_TEXT, K_HEADER_COPIES, id="header"),
        pytest.param(K_DOXYFILE, K_DOXYFILE_TEXT, K_DOXYFILE_COPIES, id="doxyfile"),
        pytest.param(K_PYPROJECT, K_PYPROJECT_TEXT, K_CLI_PACKAGE_COPIES, id="cli-package"),
        pytest.param(K_MANIFEST, '[module]\nname = "sushiblas"\nversion = "1.2.3"\n', K_MODULE_MANIFEST_COPIES,
                     id="manifest"),
        pytest.param(K_README, K_README_TEXT, K_README_COPIES, id="readme"),
    ])
    def test_accepts_agreeing_copies(self, path: str, text: str, copies) -> None:
        """Reports nothing when the file states the version of project()."""
        assert _copy_issues(path, text, copies, required=False) == []

    @pytest.mark.parametrize("path, text, copies", [
        pytest.param(K_DOXYFILE, "PROJECT_NUMBER = v1.2.3\n", K_DOXYFILE_COPIES, id="doxyfile"),
        pytest.param(K_README, "SushiBLAS v1.2.3 needs SushiRuntime v0.4.0.\n", K_README_COPIES, id="readme"),
    ])
    def test_accepts_a_v_prefix(self, path: str, text: str, copies) -> None:
        """Reports nothing for `v1.2.3` in the Doxyfile and in the README."""
        assert _copy_issues(path, text, copies, required=True) == []


class TestDisagreement:
    """Checks that each disagreeing copy is reported at its line."""

    K_TEST_SIZE = "small"

    @pytest.mark.parametrize("old, new, line, message", [
        pytest.param('"1.2.3"', '"1.2.4"', 4, f"SUSHIBLAS_VERSION_STRING is '1.2.4', {K_SOURCE}", id="the-string"),
        pytest.param("MINOR 2", "MINOR 5", 2, f"SUSHIBLAS_VERSION_MINOR is '5', {K_SOURCE}", id="a-component"),
    ])
    def test_reports_a_header_copy(self, old: str, new: str, line: int, message: str) -> None:
        """Reports the one macro of version.hpp that differs while the others agree."""
        issues = _copy_issues(K_HEADER, K_HEADER_TEXT.replace(old, new), K_HEADER_COPIES, required=True)
        assert _located(issues) == [(line, message)]

    def test_reports_the_doxyfile(self) -> None:
        """Reports a PROJECT_NUMBER that is not the version."""
        text = 'PROJECT_NAME = "SushiBLAS"\nPROJECT_NUMBER         = "v1 (unreleased)"\n'
        issues = _copy_issues(K_DOXYFILE, text, K_DOXYFILE_COPIES, required=True)
        assert _located(issues) == [(2, f"PROJECT_NUMBER is 'v1 (unreleased)', {K_SOURCE}")]

    def test_reports_the_cli_package(self) -> None:
        """Reports a package version that differs."""
        issues = _copy_issues(K_PYPROJECT, K_PYPROJECT_TEXT.replace("1.2.3", "0.9.0"), K_CLI_PACKAGE_COPIES, True)
        assert _located(issues) == [(6, f"[project] version is '0.9.0', {K_SOURCE}")]

    @pytest.mark.parametrize("table", [
        pytest.param('\n[tool.other]\nversion = "9.9.9"\n', id="a-table"),
        pytest.param('\n[[tool.other]]\nversion = "9.9.9"\n', id="an-array-table"),
    ])
    def test_ignores_a_version_key_outside_the_project_table(self, table: str) -> None:
        """Reports nothing for a `version` key of another table."""
        assert _copy_issues(K_PYPROJECT, K_PYPROJECT_TEXT + table, K_CLI_PACKAGE_COPIES, True) == []

    @pytest.mark.parametrize("text", [
        pytest.param('[module]\nname = "sushiblas"\nversion = "1.2.0"\n', id="double-quoted"),
        pytest.param("[module]  # what this module is\nname = 'sushiblas'\nversion = '1.2.0'\n",
                     id="single-quoted-below-a-commented-header"),
    ])
    def test_reports_the_module_manifest(self, text: str) -> None:
        """Reports a manifest version that differs, in either quote form."""
        issues = _copy_issues(K_MANIFEST, text, K_MODULE_MANIFEST_COPIES, required=False)
        assert _located(issues) == [(3, f"[module] version is '1.2.0', {K_SOURCE}")]

    def test_reports_the_readme(self) -> None:
        """Reports a README that names another SushiBLAS version."""
        issues = _copy_issues(K_README, "# SushiBLAS\n\nThis is SushiBLAS v1.0.0.\n", K_README_COPIES, False)
        assert _located(issues) == [(3, f"SushiBLAS x.y.z is 'v1.0.0', {K_SOURCE}")]


class TestMissing:
    """Checks what a missing file or a missing value reports."""

    K_TEST_SIZE = "small"

    @pytest.mark.parametrize("path, copies", [
        pytest.param(K_HEADER, K_HEADER_COPIES, id="header"),
        pytest.param(K_DOXYFILE, K_DOXYFILE_COPIES, id="doxyfile"),
        pytest.param(K_PYPROJECT, K_CLI_PACKAGE_COPIES, id="cli-package"),
    ])
    def test_reports_a_missing_required_file(self, path: str, copies) -> None:
        """Reports the header, the Doxyfile and the package file when each is absent."""
        issues = _copy_issues(path, None, copies, required=True)
        assert _located(issues) == [(1, "file is missing; it must state 1.2.3")]
        assert issues[0].path == Path(path)

    @pytest.mark.parametrize("copies", [
        pytest.param(K_MODULE_MANIFEST_COPIES, id="manifest"),
        pytest.param(K_README_COPIES, id="readme"),
        pytest.param(K_CHANGELOG_COPIES, id="changelog"),
    ])
    def test_accepts_a_missing_optional_file(self, copies) -> None:
        """Reports nothing when the manifest, the README or the changelog is absent."""
        assert _copy_issues("absent", None, copies, required=False) == []

    def test_reports_a_required_file_without_a_version(self) -> None:
        """Reports a Doxyfile that holds no PROJECT_NUMBER line."""
        issues = _copy_issues(K_DOXYFILE, 'PROJECT_NAME = "SushiBLAS"\n', K_DOXYFILE_COPIES, required=True)
        assert [issue.message for issue in issues] == [f"states no PROJECT_NUMBER; {K_SOURCE}"]

    def test_judges_no_copy_without_a_source_version(self) -> None:
        """Reports nothing for the copies when project() states no version."""
        repository = Repository(Path("."), None, 1)
        assert list(file_copy_issues_in(Path(K_HEADER), None, K_HEADER_COPIES, repository, True)) == []


class TestSourceFile:
    """Checks the rule that reads the root CMakeLists.txt from a folder."""

    K_TEST_SIZE = "medium"

    def test_reports_a_missing_source_once(self) -> None:
        """Reports the absent CMakeLists.txt once and nothing for the copies it cannot judge."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {K_CMAKE: None})
            issues = [issue for rule in K_CHECKER.rules.values() for issue in rule(repository)]
        assert repository.version is None
        assert [issue.message for issue in issues] == ["file is missing; project() in it is the version's source of truth"]

    @pytest.mark.parametrize("text", [
        pytest.param("project(SushiBLAS LANGUAGES CXX)\n", id="no-version"),
        pytest.param("project(SushiBLAS VERSION 1.2.3.4)\n", id="four-numbers"),
    ])
    def test_reports_a_source_without_a_readable_version(self, text: str) -> None:
        """Reports a project() call that names no version or one of four numbers."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {K_CMAKE: text})
            issues = list(rule_source_version(repository))
        assert [issue.message for issue in issues] == ["no project(<Name> VERSION x.y.z); it is the source of truth"]


class TestChangelog:
    """Checks the release heading of the changelog against project() 0.1.0."""

    K_TEST_SIZE = "small"

    def _issues(self, text: str) -> list[Issue]:
        """Returns what the changelog rule reports for one changelog text against project() 0.1.0."""
        return _copy_issues(K_CHANGELOG, text, K_CHANGELOG_COPIES, False, Version(0, 1, 0))

    def test_a_changelog_without_a_release_is_accepted(self) -> None:
        """Reports nothing for a changelog that holds only the Unreleased heading."""
        text = "# Changelog\n\n## Unreleased\n\n- 2026-10-07 — tools: Added a rule.\n"
        assert self._issues(text) == []

    @pytest.mark.parametrize("heading", [
        pytest.param("## v0.1.0 — 2026-10-20", id="release"),
        pytest.param("## v0.1.0-rc.1 — 2026-10-20", id="pre-release"),
    ])
    def test_a_release_heading_that_agrees_is_accepted(self, heading: str) -> None:
        """Reports nothing for a release or a pre-release heading of the project() version."""
        assert self._issues(f"# Changelog\n\n## Unreleased\n\n{heading}\n\n- An entry.\n") == []

    def test_a_release_heading_that_disagrees_is_reported_at_its_line(self) -> None:
        """Reports a release heading of another version at the line of the heading."""
        text = "# Changelog\n\n## Unreleased\n\n## v0.2.0 — 2026-11-01\n\n- An entry.\n"
        assert _located(self._issues(text)) == [
            (5, "release heading is '0.2.0', CMakeLists.txt:3 expects 0.1.0"),
        ]


class TestCommandLine:
    """Checks what the command line prints and returns over a folder."""

    K_TEST_SIZE = "medium"

    def _run(self, folder: str) -> tuple[int, str]:
        """Returns the exit code and the output of the checker over one folder."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = run(K_CHECKER, [folder])
        return code, output.getvalue()

    def test_exits_clean_on_agreement(self) -> None:
        """Prints nothing and returns 0 when every copy agrees."""
        with tempfile.TemporaryDirectory() as folder:
            _repository(folder)
            assert self._run(folder) == (0, "")

    def test_names_file_line_and_rule_on_disagreement(self) -> None:
        """Prints `file:line: rule: message` and returns 1 for a disagreeing copy."""
        with tempfile.TemporaryDirectory() as folder:
            _repository(folder, {K_DOXYFILE: "PROJECT_NUMBER = 1.0.0\n"})
            code, output = self._run(folder)
        assert code == 1
        assert output == f".config/doxygen/Doxyfile:1: rule_doxyfile: PROJECT_NUMBER is '1.0.0', {K_SOURCE}\n"

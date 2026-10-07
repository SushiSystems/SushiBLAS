# test_check_version.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the version checker: agreement, each disagreeing copy and a missing file."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.checker import Issue, run  # noqa: E402
from versioning.check_version import (  # noqa: E402
    K_CHECKER,
    Repository,
    Version,
    collect,
    rule_cli_package,
    rule_doxyfile,
    rule_module_manifest,
    rule_readme,
    rule_source_version,
    rule_version_header,
)

K_CMAKE = "CMakeLists.txt"
K_HEADER = "include/SushiBLAS/version.hpp"
K_DOXYFILE = "Doxyfile"
K_PYPROJECT = "cli/pyproject.toml"
K_MANIFEST = "sushi-module.toml"
K_README = "README.md"

K_AGREEING = {
    K_CMAKE: (
        "cmake_minimum_required(VERSION 3.20)\n\nproject(SushiBLAS VERSION 1.2.3 LANGUAGES CXX C)\n"
    ),
    K_HEADER: (
        "#define SUSHIBLAS_VERSION_MAJOR 1\n"
        "#define SUSHIBLAS_VERSION_MINOR 2\n"
        "#define SUSHIBLAS_VERSION_PATCH 3\n"
        '#define SUSHIBLAS_VERSION_STRING "1.2.3"\n'
    ),
    K_DOXYFILE: 'PROJECT_NAME           = "SushiBLAS"\nPROJECT_NUMBER         = "1.2.3"\n',
    K_PYPROJECT: (
        '[build-system]\nrequires = ["setuptools>=61.0"]\n\n'
        '[project]\nname = "x"\nversion = "1.2.3"\n'
    ),
    K_MANIFEST: '[module]\nname = "sushiblas"\n',
    K_README: "# SushiBLAS\n\nA tensor library.\n",
}


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


def _all_issues(repository: Repository) -> list[Issue]:
    """Returns the issues of every registered rule over one repository."""
    return [issue for rule in K_CHECKER.rules.values() for issue in rule(repository)]


class RegistrationTest(unittest.TestCase):
    """Checks the rule table."""

    def test_is_registered(self) -> None:
        """Lists the source rule and one rule per file that repeats the version."""
        self.assertEqual(
            list(K_CHECKER.rules),
            [
                "rule_source_version",
                "rule_version_header",
                "rule_doxyfile",
                "rule_cli_package",
                "rule_module_manifest",
                "rule_readme",
            ],
        )


class AgreementTest(unittest.TestCase):
    """Checks that agreeing copies report nothing."""

    def test_reads_the_version_and_its_line(self) -> None:
        """Reads the version from project(), not from cmake_minimum_required, with its line."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder)
            self.assertEqual(repository.version, Version(1, 2, 3))
            self.assertEqual(repository.source_line, 3)

    def test_reads_a_project_call_over_several_lines(self) -> None:
        """Reads a version that sits on a later line of the project() call."""
        text = "project(\n    SushiBLAS\n    VERSION 1.2.3\n    LANGUAGES CXX)\n"
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {K_CMAKE: text})
            self.assertEqual(repository.source_line, 3)
            self.assertEqual(_all_issues(repository), [])

    def test_skips_a_commented_project_call(self) -> None:
        """Reads the version from project(), not from a comment line above it."""
        text = "# project(Old VERSION 9.9.9)\n" + K_AGREEING[K_CMAKE]
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {K_CMAKE: text})
            self.assertEqual(repository.version, Version(1, 2, 3))
            self.assertEqual(repository.source_line, 4)

    def test_accepts_agreeing_copies(self) -> None:
        """Reports nothing when every copy states the version of project()."""
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(_all_issues(_repository(folder)), [])

    def test_accepts_a_v_prefix_in_doxyfile_and_readme(self) -> None:
        """Reports nothing for `v1.2.3` in the Doxyfile and in the README."""
        changes = {
            K_DOXYFILE: "PROJECT_NUMBER = v1.2.3\n",
            K_README: "SushiBLAS v1.2.3 needs SushiRuntime v0.4.0.\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(_all_issues(_repository(folder, changes)), [])

    def test_accepts_an_agreeing_manifest_version(self) -> None:
        """Reports nothing for a manifest whose module table states the version."""
        changes = {K_MANIFEST: '[module]\nname = "sushiblas"\nversion = "1.2.3"\n'}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_module_manifest(_repository(folder, changes))), [])


class DisagreementTest(unittest.TestCase):
    """Checks that each disagreeing copy is reported at its file and line."""

    def _assert_one(self, issues: list[Issue], name: str, line: int, fragment: str) -> None:
        """Asserts one issue at the given file and line whose message holds the fragment."""
        found = [(issue.path.as_posix().endswith(name), issue.line) for issue in issues]
        self.assertEqual(found, [(True, line)])
        self.assertIn(fragment, issues[0].message)
        self.assertIn("CMakeLists.txt:3 expects", issues[0].message)

    def test_reports_the_header_string(self) -> None:
        """Reports a version string in version.hpp that differs."""
        text = K_AGREEING[K_HEADER].replace('"1.2.3"', '"1.2.4"')
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_version_header(_repository(folder, {K_HEADER: text})))
            self._assert_one(issues, K_HEADER, 4, "SUSHIBLAS_VERSION_STRING is '1.2.4'")

    def test_reports_a_header_component(self) -> None:
        """Reports a minor macro that differs while the string agrees."""
        text = K_AGREEING[K_HEADER].replace("MINOR 2", "MINOR 5")
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_version_header(_repository(folder, {K_HEADER: text})))
            self._assert_one(issues, K_HEADER, 2, "SUSHIBLAS_VERSION_MINOR is '5'")

    def test_reports_the_doxyfile(self) -> None:
        """Reports a PROJECT_NUMBER that is not the version."""
        text = 'PROJECT_NAME = "SushiBLAS"\nPROJECT_NUMBER         = "v1 (unreleased)"\n'
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_doxyfile(_repository(folder, {K_DOXYFILE: text})))
            self._assert_one(issues, K_DOXYFILE, 2, "PROJECT_NUMBER is 'v1 (unreleased)'")

    def test_reports_the_cli_package(self) -> None:
        """Reports a package version that differs."""
        text = K_AGREEING[K_PYPROJECT].replace("1.2.3", "0.9.0")
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_cli_package(_repository(folder, {K_PYPROJECT: text})))
            self._assert_one(issues, K_PYPROJECT, 6, "[project] version is '0.9.0'")

    def test_ignores_a_version_key_outside_the_project_table(self) -> None:
        """Reports nothing for a `version` key of another table."""
        text = K_AGREEING[K_PYPROJECT] + '\n[tool.other]\nversion = "9.9.9"\n'
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_cli_package(_repository(folder, {K_PYPROJECT: text}))), [])

    def test_ignores_a_version_key_of_an_array_table(self) -> None:
        """Reports nothing for a `version` key below an array-of-tables header."""
        text = K_AGREEING[K_PYPROJECT] + '\n[[tool.other]]\nversion = "9.9.9"\n'
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_cli_package(_repository(folder, {K_PYPROJECT: text}))), [])

    def test_reports_a_single_quoted_value_below_a_commented_header(self) -> None:
        """Reports a manifest version in single quotes whose table header ends in a comment."""
        text = "[module]  # what this module is\nname = 'sushiblas'\nversion = '1.2.0'\n"
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_module_manifest(_repository(folder, {K_MANIFEST: text})))
            self._assert_one(issues, K_MANIFEST, 3, "[module] version is '1.2.0'")

    def test_reports_the_module_manifest(self) -> None:
        """Reports a manifest version that differs."""
        text = '[module]\nname = "sushiblas"\nversion = "1.2.0"\n'
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_module_manifest(_repository(folder, {K_MANIFEST: text})))
            self._assert_one(issues, K_MANIFEST, 3, "[module] version is '1.2.0'")

    def test_reports_the_readme(self) -> None:
        """Reports a README that names another SushiBLAS version."""
        text = "# SushiBLAS\n\nThis is SushiBLAS v1.0.0.\n"
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_readme(_repository(folder, {K_README: text})))
            self._assert_one(issues, K_README, 3, "SushiBLAS x.y.z is 'v1.0.0'")


class MissingTest(unittest.TestCase):
    """Checks what a missing file or a missing value reports."""

    def test_reports_a_missing_required_file(self) -> None:
        """Reports the header, the Doxyfile and the package file when each is absent."""
        cases = (
            (rule_version_header, K_HEADER),
            (rule_doxyfile, K_DOXYFILE),
            (rule_cli_package, K_PYPROJECT),
        )
        for rule, name in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                issues = list(rule(_repository(folder, {name: None})))
                self.assertEqual(len(issues), 1)
                self.assertTrue(issues[0].path.as_posix().endswith(name))
                self.assertIn("file is missing", issues[0].message)

    def test_accepts_a_missing_optional_file(self) -> None:
        """Reports nothing when the manifest or the README is absent."""
        changes = {K_MANIFEST: None, K_README: None}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(_all_issues(_repository(folder, changes)), [])

    def test_reports_a_required_file_without_a_version(self) -> None:
        """Reports a Doxyfile that holds no PROJECT_NUMBER line."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {K_DOXYFILE: 'PROJECT_NAME = "SushiBLAS"\n'})
            self.assertEqual(
                [issue.message for issue in rule_doxyfile(repository)],
                ["states no PROJECT_NUMBER; CMakeLists.txt:3 expects 1.2.3"],
            )

    def test_reports_a_missing_source_once(self) -> None:
        """Reports the absent CMakeLists.txt and nothing for the copies it cannot judge."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {K_CMAKE: None})
            self.assertIsNone(repository.version)
            issues = _all_issues(repository)
            self.assertEqual(len(issues), 1)
            self.assertIn("file is missing", issues[0].message)
            self.assertEqual(list(rule_source_version(repository)), issues)

    def test_reports_a_source_without_a_version(self) -> None:
        """Reports a project() call that names no version."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {K_CMAKE: "project(SushiBLAS LANGUAGES CXX)\n"})
            issues = _all_issues(repository)
            self.assertEqual(len(issues), 1)
            self.assertIn("no project(<Name> VERSION x.y.z)", issues[0].message)

    def test_reports_a_source_version_of_four_numbers(self) -> None:
        """Reports a project() version that is not three numbers."""
        text = "project(SushiBLAS VERSION 1.2.3.4)\n"
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {K_CMAKE: text})
            self.assertIsNone(repository.version)
            self.assertEqual(len(list(rule_source_version(repository))), 1)


class CommandLineTest(unittest.TestCase):
    """Checks what the command line prints and returns."""

    def _run(self, folder: str, *flags: str) -> tuple[int, str]:
        """Returns the exit code and the output of the checker over one folder."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = run(K_CHECKER, [folder, *flags])
        return code, output.getvalue()

    def test_exits_clean_on_agreement(self) -> None:
        """Prints nothing and returns 0 when every copy agrees."""
        with tempfile.TemporaryDirectory() as folder:
            _repository(folder)
            self.assertEqual(self._run(folder), (0, ""))

    def test_names_file_line_and_rule_on_disagreement(self) -> None:
        """Prints `file:line: rule: message` and returns 1 for a disagreeing copy."""
        with tempfile.TemporaryDirectory() as folder:
            _repository(folder, {K_DOXYFILE: "PROJECT_NUMBER = 1.0.0\n"})
            code, output = self._run(folder)
            self.assertEqual(code, 1)
            self.assertEqual(
                output,
                "Doxyfile:1: rule_doxyfile: PROJECT_NUMBER is '1.0.0', "
                "CMakeLists.txt:3 expects 1.2.3\n",
            )


if __name__ == "__main__":
    unittest.main()

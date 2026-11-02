# check_test_sizes.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that every test writes its size in its source, from `testing`.

Usage: python tools/testing/check_test_sizes.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import ast
import os
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402
from checking.ignored_paths import drop_ignored, ignored_folders  # noqa: E402

K_SIZES = ("small", "medium", "large")
K_TEST_FOLDERS = frozenset({"tests", "__tests__"})
K_SKIPPED_FOLDERS = frozenset({".git", "node_modules"})
K_BENCHMARK_PREFIX = "Benchmark_"
K_CPP_SUFFIXES = frozenset({".cpp", ".cc", ".cxx"})
K_CPP_SUITE = re.compile(r"^\s*(?:TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)\s*\(\s*(\w+)\s*,", re.MULTILINE)
K_CPP_PREFIXES = tuple(f"{size.capitalize()}_" for size in K_SIZES)
K_TS_DESCRIBE = re.compile(r"^describe\s*\(\s*([\"'`])(.*?)\1", re.MULTILINE)
K_TS_PREFIXES = tuple(f"{size}:" for size in K_SIZES)
K_CS_CLASS = re.compile(
    r"^[ \t]*((?:\[[^\]\n]*\][ \t]*\n?[ \t]*)*)((?:(?:public|internal|private|protected|sealed|static|abstract|partial)\s+)*)"
    r"class\s+(\w+)", re.MULTILINE)
K_CS_TEST_ATTRIBUTE = re.compile(r"\[\s*(?:Test|UnityTest|TestCase|TestCaseSource|Theory)\b")
K_CS_CATEGORY = re.compile(r"Category\s*\(\s*\"(\w+)\"\s*\)")


@dataclass(frozen=True, slots=True)
class TestTree:
    """Holds the test files of one repository that git tracks."""

    root: Path
    files: tuple[Path, ...]

    def of(self, test: callable) -> Iterator[Path]:
        """Yields the files whose name the predicate accepts."""
        return (path for path in self.files if test(path))


def _test_folders(root: Path) -> Iterator[Path]:
    """Yields every folder named like a test folder below a repository, outside ignored and nested ones."""
    ignored = ignored_folders(root)
    for current, names, _ in os.walk(root):
        names[:] = sorted(name for name in names
                          if name not in K_SKIPPED_FOLDERS and (Path(current) / name).resolve() not in ignored)
        found = [name for name in names if name in K_TEST_FOLDERS]
        for name in found:
            yield Path(current) / name
        names[:] = [name for name in names if name not in found]


def discover(root: Path) -> TestTree:
    """Returns the files under every test folder of a repository, at any depth."""
    found = sorted(path for folder in _test_folders(root) for path in folder.rglob("*") if path.is_file())
    return TestTree(root, tuple(drop_ignored(root, found)))


def collect(paths: list[Path]) -> Iterable[TestTree]:
    """Returns the test tree of each given folder."""
    return [discover(path) for path in paths if path.is_dir()]


def _line(text: str, offset: int) -> int:
    """Returns the line number of a character offset."""
    return text.count("\n", 0, offset) + 1


def rule_unsized_cpp_suite(tree: TestTree) -> Iterator[Issue]:
    """Yields an issue for each C++ test suite whose name opens with no size."""
    for path in tree.of(lambda path: path.suffix in K_CPP_SUFFIXES and path.name.startswith("test_")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in K_CPP_SUITE.finditer(text):
            if not match.group(1).startswith((*K_CPP_PREFIXES, K_BENCHMARK_PREFIX)):
                yield Issue(path, _line(text, match.start()),
                            f"suite {match.group(1)} opens with none of {', '.join(K_CPP_PREFIXES)}")


def _marks(node: ast.expr) -> set[str]:
    """Returns the pytest marks an expression names, such as `pytest.mark.small` or a list of them."""
    items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    return {ast.unparse(item).rsplit(".", 1)[-1] for item in items}


def _sizes_of(decorators: list[ast.expr]) -> set[str]:
    """Returns the sizes a list of decorators names."""
    return {name for decorator in decorators for name in _marks(decorator)} & set(K_SIZES)


def rule_unsized_python_test(tree: TestTree) -> Iterator[Issue]:
    """Yields an issue for each pytest test whose file, class or function carries no size."""
    for path in tree.of(lambda path: path.suffix == ".py" and path.name.startswith("test_")):
        module = ast.parse(path.read_text(encoding="utf-8"))
        file_sizes = {
            name
            for node in module.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets)
            for name in _marks(node.value)
        } & set(K_SIZES)
        if file_sizes:
            continue
        for node in module.body:
            is_test = (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")) \
                or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
            if is_test and not node.name.removeprefix("Test").startswith(K_BENCHMARK_PREFIX)                     and not _sizes_of(node.decorator_list):
                yield Issue(path, node.lineno, f"{node.name} carries no size, and the file sets none")


def rule_unsized_typescript_suite(tree: TestTree) -> Iterator[Issue]:
    """Yields an issue for each top-level TypeScript describe block whose name opens with no size."""
    for path in tree.of(lambda path: path.name.endswith((".test.ts", ".test.tsx"))):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in K_TS_DESCRIBE.finditer(text):
            if not match.group(2).startswith((*K_TS_PREFIXES, K_BENCHMARK_PREFIX)):
                yield Issue(path, _line(text, match.start()),
                            f"describe '{match.group(2)}' opens with none of {', '.join(K_TS_PREFIXES)}")


def rule_unsized_csharp_fixture(tree: TestTree) -> Iterator[Issue]:
    """Yields an issue for each NUnit fixture class that carries no `[Category("small|medium|large")]`."""
    for path in tree.of(lambda path: path.suffix == ".cs"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        declarations = list(K_CS_CLASS.finditer(text))
        for index, match in enumerate(declarations):
            end = declarations[index + 1].start() if index + 1 < len(declarations) else len(text)
            name = match.group(3)
            is_fixture = "TestFixture" in match.group(1) or K_CS_TEST_ATTRIBUTE.search(text, match.end(), end)
            if not is_fixture or "abstract" in match.group(2).split() or name.startswith(K_BENCHMARK_PREFIX):
                continue
            if not set(K_CS_CATEGORY.findall(match.group(1))) & set(K_SIZES):
                yield Issue(path, _line(text, match.start(1) if match.group(1) else match.start(3)),
                            f"fixture {name} carries no [Category(\"small\")], medium or large")


K_CHECKER = Checker(
    description="Checks that every test writes its size in its source.",
    collect=collect,
    rules={
        "rule_unsized_cpp_suite": rule_unsized_cpp_suite,
        "rule_unsized_python_test": rule_unsized_python_test,
        "rule_unsized_typescript_suite": rule_unsized_typescript_suite,
        "rule_unsized_csharp_fixture": rule_unsized_csharp_fixture,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))

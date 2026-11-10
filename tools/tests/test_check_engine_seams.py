# test_check_engine_seams.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the Engine seam rules: engine.hpp includes, view calls on an Engine and task ordering."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layering.check_engine_seams import (  # noqa: E402
    K_CHECKER,
    K_VIEW_ALLOWED_MEMBERS,
    removed_engine_members_in,
    rule_removed_engine_members,
    rule_seam_engine_includes,
    rule_task_dependencies,
    rule_view_engine_calls,
    seam_engine_includes_in,
)
from layering.check_tier_contents import KnownViolation, Repository, _engine_calls  # noqa: E402

K_ENGINE_HEADER_INCLUDE = "#include <SushiBLAS/engine/engine.hpp>\n"
K_TEST_ONLY_TIER = "if(SB_BUILD_TESTS)\n    add_subdirectory(engine_fixture)\nendif()\n"
K_TEST_ONLY_MODULE = "sushiblas_add_module(\n    NAME engine_fixture\n    LAYER engine\n    INTERNAL)\n"
K_FIXTURE = "modules/engine/engine_fixture/include/SushiBLAS/engine_fixture/sushiblas_test.hpp"
K_VIEW = "modules/operations/blas/include/SushiBLAS/blas/blas.hpp"
K_SEAM = "modules/operations/blas/include/SushiBLAS/blas/dispatch.hpp"
K_SOURCE = "modules/operations/blas/source/level1/dot.cpp"
K_ENGINE = "modules/engine/engine/include/SushiBLAS/engine/engine.hpp"
K_COMMAND_GROUP = (
    "return queue.submit([&](sycl::handler& h)\n"
    "{\n"
    "    {depends}h.parallel_for(range, body);\n"
    "});\n"
)
K_FINDINGS = [
    "layering violation: SushiBLAS/engine/engine.hpp",
    "layering violation: create_tensor",
    "layering violation: memcpy",
]


def _lines(path: str, text: str) -> list[tuple[Path, int, str]]:
    """Returns the numbered lines of a text, each with the path it belongs to."""
    return [(Path(path), number, line) for number, line in enumerate(text.splitlines(), 1)]


def _repository(folder: str, files: dict[str, str], known: tuple[KnownViolation, ...] = ()) -> Repository:
    """Returns a repository at folder that holds the given files and known table."""
    root = Path(folder)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return Repository(root, known)


def _messages(rule, files: dict[str, str]) -> list[tuple[str, str]]:
    """Returns the file name and message of each issue a rule reports on a repository of the files."""
    with tempfile.TemporaryDirectory() as folder:
        return [(issue.path.name, issue.message) for issue in rule(_repository(folder, files))]


class TestRegistration:
    """Checks the rule table of the Engine seam checker."""

    K_TEST_SIZE = "small"

    def test_is_registered(self) -> None:
        """Lists the four Engine seam rules."""
        assert list(K_CHECKER.rules) == [
            "rule_seam_engine_includes", "rule_view_engine_calls", "rule_task_dependencies",
            "rule_removed_engine_members",
        ]


class TestSeamEngineIncludes:
    """Checks that only the view of an operation family, `<family>/<family>.hpp`, includes engine.hpp."""

    K_TEST_SIZE = "small"

    def _issues(self, headers: dict[str, str], sources: dict[str, str]) -> list[str]:
        """Returns the paths of the includes reported for the given headers and sources, with the blas view exempt."""
        header_lines = [entry for path, text in headers.items() for entry in _lines(path, text)]
        source_lines = [entry for path, text in sources.items() for entry in _lines(path, text)]
        exempt = frozenset({K_VIEW, K_ENGINE})
        found = seam_engine_includes_in(header_lines, source_lines, exempt, lambda path: path.as_posix())
        return [violation.path.as_posix() for violation in found]

    def test_accepts_the_view_of_a_family(self) -> None:
        """Reports nothing for a view header that includes engine.hpp."""
        assert self._issues({K_VIEW: K_ENGINE_HEADER_INCLUDE}, {}) == []

    def test_reports_a_seam_header(self) -> None:
        """Reports a dispatch seam that includes engine.hpp."""
        assert self._issues({K_SEAM: K_ENGINE_HEADER_INCLUDE}, {}) == [K_SEAM]

    def test_reports_an_operation_source(self) -> None:
        """Reports an operation source that includes engine.hpp."""
        assert self._issues({}, {K_SOURCE: K_ENGINE_HEADER_INCLUDE}) == [K_SOURCE]


class TestViewEngineCalls:
    """Checks which Engine members a public header calls."""

    K_TEST_SIZE = "small"

    def test_accepts_rng(self) -> None:
        """Reports nothing for the one member a header may read."""
        lines = _lines("random.hpp", "RandomOps(recorder, engine.rng())\n")
        assert list(_engine_calls(lines, K_VIEW_ALLOWED_MEMBERS)) == []

    def test_reports_another_member(self) -> None:
        """Reports execute in a seam header."""
        lines = _lines("task.hpp", "engine.execute();\n")
        assert [violation.symbol for violation in _engine_calls(lines, K_VIEW_ALLOWED_MEMBERS)] == ["execute"]


class TestViewEngineCallsInEngineHeader:
    """Checks that the view rule does not scan engine.hpp itself."""

    K_TEST_SIZE = "medium"

    def test_skips_engine_hpp(self) -> None:
        """Reports the seam header and nothing in engine.hpp."""
        files = {
            "modules/engine/engine/include/SushiBLAS/engine/task.hpp": "engine.execute();\n",
            K_ENGINE: "engine.execute();\n",
        }
        assert _messages(rule_view_engine_calls, files) == [("task.hpp", "layering violation: execute")]


class TestRemovedEngineMembers:
    """Checks that no removed Engine member and no family constructor over Engine& comes back."""

    K_TEST_SIZE = "small"

    def _symbols(self, engine: str, header: str) -> list[str]:
        """Returns the symbol of each violation in the given engine.hpp and family header text."""
        found = removed_engine_members_in(_lines("engine.hpp", engine), _lines("blas.hpp", header))
        return [violation.symbol for violation in found]

    def test_accepts_the_engine_without_a_graph(self) -> None:
        """Reports nothing for an engine with a comment naming recorder() and a deleted copy constructor."""
        engine = (
            "/** @brief Returns the pool; no recorder() exists. */\n"
            "ScratchPool& scratch_pool() noexcept;\n"
            "Engine(const Engine&) = delete;\n"
        )
        assert self._symbols(engine, "") == []

    def test_accepts_a_family_built_over_a_recorder(self) -> None:
        """Reports nothing for a family constructor over a TaskRecorder or a fixture over an Engine."""
        header = "explicit BLASOps(TaskRecorder& recorder);\nexplicit RecordingFixture(Engine& engine);\n"
        assert self._symbols("", header) == []

    def test_reports_a_planted_execute(self) -> None:
        """Reports Engine::execute() declared in engine.hpp."""
        assert self._symbols("[[nodiscard]] SushiRuntime::Result<void> execute();\n", "") == ["execute"]

    def test_reports_a_planted_family_constructor(self) -> None:
        """Reports an XOps(Engine&) constructor."""
        assert self._symbols("", "explicit BLASOps(Engine& engine) noexcept;\n") == ["BLASOps(Engine&)"]

    def test_reports_each_other_removed_member(self) -> None:
        """Reports discard_recorded, recorder, the graph member and RecordedGraph in engine.hpp."""
        engine = "void discard_recorded();\nTaskRecorder& recorder() noexcept;\nRecordedGraph graph_;\n"
        assert self._symbols(engine, "") == ["discard_recorded", "TaskRecorder", "recorder", "RecordedGraph", "graph_"]


class TestRemovedEngineMembersOfARepository:
    """Checks the rule over a folder: it reads engine.hpp and the family headers."""

    K_TEST_SIZE = "medium"

    def test_reports_both_kinds_in_one_repository(self) -> None:
        """Reports execute in engine.hpp, then the family constructor."""
        files = {
            K_ENGINE: "[[nodiscard]] SushiRuntime::Result<void> execute();\n",
            K_VIEW: "explicit BLASOps(Engine& engine) noexcept;\n",
        }
        assert _messages(rule_removed_engine_members, files) == [
            ("engine.hpp", "layering violation: execute"),
            ("blas.hpp", "layering violation: BLASOps(Engine&)"),
        ]


class TestTaskDependencies:
    """Checks that a device task body orders its work behind the events it is given."""

    K_TEST_SIZE = "medium"

    def _file(self, depends: str) -> dict[str, str]:
        """Returns a device header whose command group holds the given depends_on line."""
        text = K_COMMAND_GROUP.replace("{depends}", depends)
        return {"modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/add.hpp": text}

    @pytest.mark.parametrize("depends", [
        pytest.param("h.depends_on(deps);\n    ", id="deps"),
        pytest.param("h.depends_on(staged);\n    ", id="a-second-stage"),
    ])
    def test_accepts_a_command_group_that_names_its_dependencies(self, depends: str) -> None:
        """Reports nothing for a submit with depends_on."""
        assert _messages(rule_task_dependencies, self._file(depends)) == []

    def test_reports_a_command_group_without_depends_on(self) -> None:
        """Reports a submit without depends_on."""
        assert _messages(rule_task_dependencies, self._file("")) == [("add.hpp", "layering violation: submit")]

    def test_accepts_a_shortcut_that_passes_deps(self) -> None:
        """Reports nothing for a fill that passes deps."""
        files = {"modules/operations/random/source/constant.cpp": "return q.fill(pT, value, size, deps);\n"}
        assert _messages(rule_task_dependencies, files) == []

    def test_reports_a_shortcut_that_drops_deps(self) -> None:
        """Reports a fill without deps."""
        files = {"modules/operations/random/source/constant.cpp": "return q.fill(pT, value, size);\n"}
        assert _messages(rule_task_dependencies, files) == [("constant.cpp", "layering violation: fill")]


class TestTestOnlyModule:
    """Checks that an INTERNAL module added only under a test or benchmark switch is exempt from the three rules."""

    K_TEST_SIZE = "medium"

    K_RULES = (rule_seam_engine_includes, rule_view_engine_calls, rule_task_dependencies)

    @pytest.mark.parametrize("switch, flags, expected", [
        pytest.param("SB_BUILD_TESTS", K_TEST_ONLY_MODULE, [], id="tests"),
        pytest.param("SB_BUILD_BENCHMARKS", K_TEST_ONLY_MODULE, [], id="benchmarks"),
        pytest.param("SB_BUILD_TESTS OR SB_BUILD_BENCHMARKS", K_TEST_ONLY_MODULE, [], id="tests-or-benchmarks"),
        pytest.param("SB_BUILD_TESTS", K_TEST_ONLY_MODULE.replace("    INTERNAL", ""), K_FINDINGS, id="not-internal"),
    ])
    def test_exempts_only_an_internal_module(self, switch: str, flags: str, expected: list[str]) -> None:
        """Reports nothing for an INTERNAL module under the switch and one finding per rule otherwise."""
        files = {
            "modules/engine/CMakeLists.txt": K_TEST_ONLY_TIER.replace("SB_BUILD_TESTS", switch),
            "modules/engine/engine_fixture/CMakeLists.txt": flags,
            K_FIXTURE: K_ENGINE_HEADER_INCLUDE + "engine->create_tensor({});\nq.memcpy(a, b, n).wait();\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, files)
            messages = [issue.message for rule in self.K_RULES for issue in rule(repository)]
        assert messages == expected

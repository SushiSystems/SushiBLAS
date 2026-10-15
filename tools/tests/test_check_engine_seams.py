# test_check_engine_seams.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the Engine seam rules: engine.hpp includes, view calls on an Engine and task ordering."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layering.check_engine_seams import (  # noqa: E402
    K_CHECKER,
    rule_seam_engine_includes,
    rule_task_dependencies,
    rule_view_engine_calls,
)
from layering.check_tier_contents import KnownViolation, Repository  # noqa: E402

K_ENGINE_HEADER_INCLUDE = "#include <SushiBLAS/engine/engine.hpp>\n"
K_TEST_ONLY_TIER = "if(SB_BUILD_TESTS)\n    add_subdirectory(engine_fixture)\nendif()\n"
K_TEST_ONLY_MODULE = "sushiblas_add_module(\n    NAME engine_fixture\n    LAYER engine\n    INTERNAL)\n"
K_FIXTURE = "modules/engine/engine_fixture/include/SushiBLAS/engine_fixture/sushiblas_test.hpp"


def _repository(folder: str, files: dict[str, str], known: tuple[KnownViolation, ...] = ()) -> Repository:
    """Returns a repository at folder that holds the given files and known table."""
    root = Path(folder)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return Repository(root, known)


class RegistrationTest(unittest.TestCase):
    """Checks the rule table of the Engine seam checker."""

    def test_is_registered(self) -> None:
        """Lists the three Engine seam rules."""
        self.assertEqual(
            list(K_CHECKER.rules),
            ["rule_seam_engine_includes", "rule_view_engine_calls", "rule_task_dependencies"],
        )


class SeamEngineIncludesTest(unittest.TestCase):
    """Checks that only the view of an operation family, `<family>/<family>.hpp`, includes engine.hpp."""

    def test_accepts_the_view_of_a_family(self) -> None:
        """Reports nothing for a view header that includes engine.hpp."""
        files = {
            "modules/operations/blas/include/SushiBLAS/blas/blas.hpp": K_ENGINE_HEADER_INCLUDE,
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_seam_engine_includes(_repository(folder, files))), [])

    def test_reports_a_seam_and_an_operation_source(self) -> None:
        """Reports a dispatch seam and an operation source that include engine.hpp."""
        files = {
            "modules/operations/blas/include/SushiBLAS/blas/dispatch.hpp": K_ENGINE_HEADER_INCLUDE,
            "modules/operations/blas/source/level1/dot.cpp": K_ENGINE_HEADER_INCLUDE,
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_seam_engine_includes(_repository(folder, files)))
            self.assertEqual([issue.path.name for issue in issues], ["dispatch.hpp", "dot.cpp"])


class ViewEngineCallsTest(unittest.TestCase):
    """Checks which Engine members a public header calls."""

    def test_accepts_recorder_and_rng(self) -> None:
        """Reports nothing for the two members a view constructor reads."""
        files = {"modules/operations/random/include/SushiBLAS/random/random.hpp": "RandomOps(engine.recorder(), engine.rng())\n"}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_view_engine_calls(_repository(folder, files))), [])

    def test_reports_another_member_and_skips_engine_hpp(self) -> None:
        """Reports execute in a seam and nothing in engine.hpp itself."""
        files = {
            "modules/engine/engine/include/SushiBLAS/engine/task.hpp": "engine.execute();\n",
            "modules/engine/engine/include/SushiBLAS/engine/engine.hpp": "engine.execute();\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_view_engine_calls(_repository(folder, files)))
            self.assertEqual([(issue.path.name, issue.message) for issue in issues],
                             [("task.hpp", "layering violation: execute")])


class TaskDependenciesTest(unittest.TestCase):
    """Checks that a device task body orders its work behind the events it is given."""

    def test_accepts_a_command_group_and_a_shortcut_that_name_their_dependencies(self) -> None:
        """Reports nothing for a command group with depends_on and a fill that passes deps."""
        files = {
            "modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/add.hpp": (
                "return queue.submit([&](sycl::handler& h)\n"
                "{\n"
                "    h.depends_on(deps);\n"
                "    h.parallel_for(range, body);\n"
                "});\n"
            ),
            "modules/operations/random/source/constant.cpp": "return q.fill(pT, value, size, deps);\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_task_dependencies(_repository(folder, files))), [])

    def test_reports_a_command_group_and_a_shortcut_that_drop_them(self) -> None:
        """Reports a command group without depends_on and a fill without deps."""
        files = {
            "modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/add.hpp": (
                "return queue.submit([&](sycl::handler& h)\n"
                "{\n"
                "    h.parallel_for(range, body);\n"
                "});\n"
            ),
            "modules/operations/random/source/constant.cpp": "return q.fill(pT, value, size);\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_task_dependencies(_repository(folder, files)))
            self.assertEqual([(issue.path.name, issue.message) for issue in issues],
                             [("add.hpp", "layering violation: submit"),
                              ("constant.cpp", "layering violation: fill")])

    def test_accepts_a_second_stage_that_waits_on_the_first(self) -> None:
        """Reports nothing for a command group that depends on an earlier stage's event."""
        files = {
            "modules/foundation/reduction/include/SushiBLAS/reduction/reduce_1d.hpp": (
                "return queue.submit([&](sycl::handler& h)\n"
                "{\n"
                "    h.depends_on(staged);\n"
                "    h.single_task(fold);\n"
                "});\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_task_dependencies(_repository(folder, files))), [])


class TestOnlyModuleTest(unittest.TestCase):
    """Checks that an INTERNAL module added only under SB_BUILD_TESTS is exempt from the three rules."""

    def _files(self, module_flags: str) -> dict[str, str]:
        """Returns a test-only fixture header that breaks all three rules, under the given module flags."""
        return {
            "modules/engine/CMakeLists.txt": K_TEST_ONLY_TIER,
            "modules/engine/engine_fixture/CMakeLists.txt": module_flags,
            K_FIXTURE: K_ENGINE_HEADER_INCLUDE + "engine->create_tensor({});\nq.memcpy(a, b, n).wait();\n",
        }

    def test_exempts_a_test_only_module(self) -> None:
        """Reports nothing for a header of an INTERNAL module the tier adds under SB_BUILD_TESTS."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, self._files(K_TEST_ONLY_MODULE))
            for rule in (rule_seam_engine_includes, rule_view_engine_calls, rule_task_dependencies):
                self.assertEqual(list(rule(repository)), [])

    def test_exempts_a_benchmark_only_module(self) -> None:
        """Reports nothing for a header of an INTERNAL module the tier adds under SB_BUILD_BENCHMARKS."""
        with tempfile.TemporaryDirectory() as folder:
            files = self._files(K_TEST_ONLY_MODULE)
            files["modules/engine/CMakeLists.txt"] = K_TEST_ONLY_TIER.replace("SB_BUILD_TESTS", "SB_BUILD_BENCHMARKS")
            repository = _repository(folder, files)
            for rule in (rule_seam_engine_includes, rule_view_engine_calls, rule_task_dependencies):
                self.assertEqual(list(rule(repository)), [])

    def test_checks_a_module_that_is_not_internal(self) -> None:
        """Reports all three findings when the module under the switch is not INTERNAL."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, self._files(K_TEST_ONLY_MODULE.replace("    INTERNAL", "")))
            messages = [issue.message for rule in (rule_seam_engine_includes, rule_view_engine_calls,
                                                   rule_task_dependencies) for issue in rule(repository)]
            self.assertEqual(messages, ["layering violation: SushiBLAS/engine/engine.hpp",
                                        "layering violation: create_tensor",
                                        "layering violation: memcpy"])


if __name__ == "__main__":
    unittest.main()

# test_check_layering.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the support, kernel, engine header and operation rules of the layering checker."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layering.check_layering import (  # noqa: E402
    K_CHECKER,
    KnownViolation,
    Repository,
    known_notes,
    rule_engine_header_includes,
    rule_kernel_includes,
    rule_operation_engine_calls,
    rule_support_independence,
)

K_KERNEL = "kernels/reference/add.hpp"
K_SUPPORT = "include/SushiBLAS/support/launch_1d.hpp"
K_ENGINE_INCLUDE = "#include <SushiBLAS/engine/task.hpp>\n"


def _repository(folder: str, files: dict[str, str], known: tuple[KnownViolation, ...] = ()) -> Repository:
    """Returns a repository at folder that holds the given files and known table."""
    root = Path(folder)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return Repository(root, None, known)


class RegistrationTest(unittest.TestCase):
    """Checks that the rule table keeps the old names and carries the new rules."""

    def test_is_registered(self) -> None:
        """Lists every module-tree rule and every repository layer rule."""
        self.assertEqual(
            list(K_CHECKER.rules),
            [
                "rule_declared_tiers",
                "rule_upward_includes",
                "rule_private_reach",
                "rule_support_independence",
                "rule_kernel_includes",
                "rule_engine_header_includes",
                "rule_operation_engine_calls",
            ],
        )


class SupportIndependenceTest(unittest.TestCase):
    """Checks which includes a support header may make."""

    def test_accepts_core_support_and_sycl(self) -> None:
        """Reports nothing for a support header that includes core, support and SYCL headers."""
        files = {
            K_SUPPORT: (
                "#include <sycl/sycl.hpp>\n"
                "#include <SushiBLAS/core/traits.hpp>\n"
                "#include <SushiBLAS/support/device_policy.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_support_independence(_repository(folder, files))), [])

    def test_reports_engine_kernel_and_tensor_includes(self) -> None:
        """Reports engine.hpp, engine/, kernels/ and tensor.hpp includes, each on its line."""
        files = {
            K_SUPPORT: (
                "#include <SushiBLAS/engine.hpp>\n"
                '#include "SushiBLAS/engine/task.hpp"\n'
                "#include <SushiBLAS/kernels/sycl/blas/level3/gemm.hpp>\n"
                "#include <SushiBLAS/tensor.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_support_independence(_repository(folder, files)))
            self.assertEqual([issue.line for issue in issues], [1, 2, 3, 4])


class KernelIncludesTest(unittest.TestCase):
    """Checks which includes a kernel may make."""

    def test_accepts_a_clean_kernel(self) -> None:
        """Reports nothing for a kernel that includes only core headers."""
        files = {K_KERNEL: "#include <SushiBLAS/core/common.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_kernel_includes(_repository(folder, files))), [])

    def test_reports_engine_and_tensor_includes(self) -> None:
        """Reports engine.hpp, an engine/ header and tensor.hpp, each on its line."""
        files = {
            "include/SushiBLAS/kernels/sycl/k.hpp": (
                "#include <SushiBLAS/engine.hpp>\n"
                '#include "SushiBLAS/engine/task.hpp"\n'
                "#include <SushiBLAS/tensor.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_kernel_includes(_repository(folder, files)))
            self.assertEqual([issue.line for issue in issues], [1, 2, 3])

    def test_reports_a_known_entry_as_a_note_only(self) -> None:
        """Reports nothing for a tabled violation and names it in the notes."""
        files = {K_KERNEL: K_ENGINE_INCLUDE}
        known = (KnownViolation("rule_kernel_includes", K_KERNEL, "SushiBLAS/engine/task.hpp", "r"),)
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, files, known)
            self.assertEqual(list(rule_kernel_includes(repository)), [])
            notes = list(known_notes(repository, ["rule_kernel_includes"]))
            self.assertEqual(len(notes), 1)
            self.assertIn(K_KERNEL, notes[0])

    def test_reports_a_stale_entry(self) -> None:
        """Reports a tabled violation that no longer occurs."""
        files = {K_KERNEL: "#include <SushiBLAS/core/common.hpp>\n"}
        known = (KnownViolation("rule_kernel_includes", K_KERNEL, "SushiBLAS/engine/task.hpp", "r"),)
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_kernel_includes(_repository(folder, files, known)))
            self.assertEqual(len(issues), 1)
            self.assertIn("stale", issues[0].message)


class EngineHeaderIncludesTest(unittest.TestCase):
    """Checks that engine.hpp includes no operation family."""

    def test_accepts_a_header_without_families(self) -> None:
        """Reports nothing for an engine.hpp that includes the tensor and core headers."""
        files = {"include/SushiBLAS/engine.hpp": "#include <SushiBLAS/tensor.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_engine_header_includes(_repository(folder, files))), [])

    def test_reports_a_family_include(self) -> None:
        """Reports an include of an operation-family header."""
        files = {"include/SushiBLAS/engine.hpp": "#include <SushiBLAS/engine/math/random.hpp>\n"}
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_engine_header_includes(_repository(folder, files)))
            self.assertEqual([issue.message for issue in issues],
                             ["layering violation: SushiBLAS/engine/math/random.hpp"])


class OperationEngineCallsTest(unittest.TestCase):
    """Checks which Engine members an operation source calls."""

    def test_accepts_get_graph(self) -> None:
        """Reports nothing for a call to get_graph or for a call in a comment."""
        files = {"src/engine/blas/a.cpp": "auto& g = engine_.get_graph();\n// engine_.get_mode();\n"}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_operation_engine_calls(_repository(folder, files))), [])

    def test_reports_another_member(self) -> None:
        """Reports a call to get_mode through a reference and through a pointer."""
        files = {"src/engine/blas/a.cpp": "engine_.get_mode();\nengine->create_tensor({});\n"}
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_operation_engine_calls(_repository(folder, files)))
            self.assertEqual([issue.message for issue in issues],
                             ["layering violation: get_mode", "layering violation: create_tensor"])


if __name__ == "__main__":
    unittest.main()

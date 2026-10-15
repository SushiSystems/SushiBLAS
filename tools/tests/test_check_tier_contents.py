# test_check_tier_contents.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the foundation, kernel, Engine, view and recorder rules of check_tier_contents.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layering.check_tier_contents import (  # noqa: E402
    K_CHECKER,
    KnownViolation,
    Repository,
    known_notes,
    rule_engine_header_includes,
    rule_graph_insertions,
    rule_kernel_includes,
    rule_operation_engine_calls,
    rule_foundation_independence,
)

K_KERNEL = "modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/reference/add.hpp"
K_FOUNDATION = "modules/foundation/launch/include/SushiBLAS/launch/launch_1d.hpp"
K_ENGINE_INCLUDE = "#include <SushiBLAS/engine/task.hpp>\n"
K_ENGINE_HEADER_INCLUDE = "#include <SushiBLAS/engine/engine.hpp>\n"


def _repository(folder: str, files: dict[str, str], known: tuple[KnownViolation, ...] = ()) -> Repository:
    """Returns a repository at folder that holds the given files and known table."""
    root = Path(folder)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return Repository(root, known)


class RegistrationTest(unittest.TestCase):
    """Checks that the rule table carries the five SushiBLAS layer rules."""

    def test_is_registered(self) -> None:
        """Lists every layer rule in its order."""
        self.assertEqual(
            list(K_CHECKER.rules),
            [
                "rule_foundation_independence",
                "rule_kernel_includes",
                "rule_engine_header_includes",
                "rule_operation_engine_calls",
                "rule_graph_insertions",
            ],
        )


class FoundationIndependenceTest(unittest.TestCase):
    """Checks which includes a foundation header may make."""

    def test_accepts_core_foundation_and_sycl(self) -> None:
        """Reports nothing for a foundation header that includes core, foundation and SYCL headers."""
        files = {
            K_FOUNDATION: (
                "#include <sycl/sycl.hpp>\n"
                "#include <SushiBLAS/core/traits.hpp>\n"
                "#include <SushiBLAS/launch/device_policy.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_foundation_independence(_repository(folder, files))), [])

    def test_reports_engine_kernel_and_tensor_includes(self) -> None:
        """Reports engine.hpp, engine/, modules/kernels/ and tensor.hpp includes, each on its line."""
        files = {
            K_FOUNDATION: (
                "#include <SushiBLAS/engine/engine.hpp>\n"
                '#include "SushiBLAS/engine/task.hpp"\n'
                "#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>\n"
                "#include <SushiBLAS/tensor/tensor.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_foundation_independence(_repository(folder, files)))
            self.assertEqual([issue.line for issue in issues], [1, 2, 3, 4])

    def test_reports_the_sparse_operand_headers(self) -> None:
        """Reports sparse_vector.hpp, sparse_matrix.hpp and sparse_builder.hpp, each on its line."""
        files = {
            K_FOUNDATION: (
                "#include <SushiBLAS/sparse_storage/sparse_vector.hpp>\n"
                "#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>\n"
                "#include <SushiBLAS/sparse_storage/sparse_builder.hpp>\n"
                "#include <SushiBLAS/core/logger.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_foundation_independence(_repository(folder, files)))
            self.assertEqual([issue.line for issue in issues], [1, 2, 3])


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
            "modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/sycl/k.hpp": (
                "#include <SushiBLAS/engine/engine.hpp>\n"
                '#include "SushiBLAS/engine/task.hpp"\n'
                "#include <SushiBLAS/tensor/tensor.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_kernel_includes(_repository(folder, files)))
            self.assertEqual([issue.line for issue in issues], [1, 2, 3])

    def test_reports_the_sparse_operand_headers(self) -> None:
        """Reports the three sparse operand headers and accepts a private kernel brick."""
        files = {
            "modules/kernels/sparse_kernels/include/SushiBLAS/sparse_kernels/sycl/k.hpp": (
                "#include <SushiBLAS/sparse_storage/sparse_vector.hpp>\n"
                "#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>\n"
                "#include <SushiBLAS/sparse_storage/sparse_builder.hpp>\n"
                '#include "kernels/sparse/csr_pattern.hpp"\n'
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
    """Checks that engine.hpp includes its five bricks and no view, seam or experimental header."""

    def test_accepts_the_bricks_engine_composes(self) -> None:
        """Reports nothing for the tensor header and the five engine bricks."""
        files = {
            "modules/engine/engine/include/SushiBLAS/engine/engine.hpp": (
                "#include <SushiBLAS/tensor/tensor.hpp>\n"
                "#include <SushiBLAS/tensor/tensor_factory.hpp>\n"
                "#include <SushiBLAS/engine/rng_stream.hpp>\n"
                "#include <SushiBLAS/engine/recorded_graph.hpp>\n"
                "#include <SushiBLAS/engine/scratch_pool.hpp>\n"
                "#include <SushiBLAS/engine/task_recorder.hpp>\n"
                "#include <SushiBLAS/engine/execution_mode.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_engine_header_includes(_repository(folder, files))), [])

    def test_reports_a_view_a_seam_io_and_experimental(self) -> None:
        """Reports a family view, a dispatch seam, io.hpp and an experimental header."""
        files = {
            "modules/engine/engine/include/SushiBLAS/engine/engine.hpp": (
                "#include <SushiBLAS/random/random.hpp>\n"
                "#include <SushiBLAS/engine/task.hpp>\n"
                "#include <SushiBLAS/io/io.hpp>\n"
                "#include <SushiBLAS/experimental/lapack/linalg.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_engine_header_includes(_repository(folder, files)))
            self.assertEqual([issue.message for issue in issues], [
                "layering violation: SushiBLAS/random/random.hpp",
                "layering violation: SushiBLAS/engine/task.hpp",
                "layering violation: SushiBLAS/io/io.hpp",
                "layering violation: SushiBLAS/experimental/lapack/linalg.hpp",
            ])


class OperationEngineCallsTest(unittest.TestCase):
    """Checks that an operation source calls nothing on an Engine."""

    def test_accepts_the_recorder_and_a_comment(self) -> None:
        """Reports nothing for a recorder call or for an engine call in a comment."""
        files = {"modules/operations/blas/source/a.cpp": "recorder_.record_pair(description, operands, family, body);\n// engine_.mode();\n"}
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_operation_engine_calls(_repository(folder, files))), [])

    def test_reports_every_member(self) -> None:
        """Reports recorder through a reference and create_tensor through a pointer."""
        files = {"modules/operations/blas/source/a.cpp": "engine_.recorder();\nengine->create_tensor({});\n"}
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_operation_engine_calls(_repository(folder, files)))
            self.assertEqual([issue.message for issue in issues],
                             ["layering violation: recorder", "layering violation: create_tensor"])


class GraphInsertionsTest(unittest.TestCase):
    """Checks that TaskRecorder and the graph primitives are the only callers of the recording verbs."""

    def test_accepts_the_recorder_header_a_primitive_and_a_comment(self) -> None:
        """Reports nothing for task in task_recorder.hpp, kernel in a graph primitive, or a comment."""
        files = {
            "modules/engine/engine/include/SushiBLAS/engine/task_recorder.hpp": "recorder_->task(access, work, options);\n",
            "modules/foundation/graph_primitives/include/SushiBLAS/graph_primitives/reduce.hpp": "recorder.kernel(count, access, body, options);\n",
            "modules/operations/logic/source/all.cpp": "// recorder.task(access, work) is the recorder's call.\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_graph_insertions(_repository(folder, files))), [])

    def test_accepts_a_record_call_and_a_free_function(self) -> None:
        """Reports nothing for TaskRecorder::record_pair or for a function named like a verb."""
        files = {
            "modules/operations/logic/source/all.cpp": "recorder_.record_pair(description, operands, family, body);\nkernel(q, deps);\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_graph_insertions(_repository(folder, files))), [])

    def test_reports_a_seam_and_a_source(self) -> None:
        """Reports task in a dispatch seam and host through a pointer in an operation source."""
        files = {
            "modules/operations/logic/include/SushiBLAS/logic/dispatch.hpp": "graph.recorder().task(access, body, options);\n",
            "modules/operations/logic/source/all.cpp": "recorder->host(access, work);\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_graph_insertions(_repository(folder, files)))
            self.assertEqual([issue.message for issue in issues],
                             ["layering violation: task", "layering violation: host"])

    def test_reports_every_recording_verb(self) -> None:
        """Reports each member of SushiRuntime::Recorder that records a node."""
        verbs = ("task", "kernel", "state_rule", "host", "untracked_kernel", "dynamic_kernel", "coupled_rule")
        files = {
            "modules/operations/logic/source/all.cpp": "".join(f"recorder.{verb}(arguments);\n" for verb in verbs),
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_graph_insertions(_repository(folder, files)))
            self.assertEqual([issue.message for issue in issues],
                             [f"layering violation: {verb}" for verb in verbs])


if __name__ == "__main__":
    unittest.main()

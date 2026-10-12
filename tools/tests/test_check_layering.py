# test_check_layering.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the support, kernel, Engine, view and recorder rules of the layering checker."""

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
    rule_graph_insertions,
    rule_kernel_includes,
    rule_operation_engine_calls,
    rule_seam_engine_includes,
    rule_support_independence,
    rule_task_dependencies,
    rule_view_engine_calls,
)

K_KERNEL = "modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/reference/add.hpp"
K_SUPPORT = "modules/foundation/support/include/SushiBLAS/support/launch_1d.hpp"
K_ENGINE_INCLUDE = "#include <SushiBLAS/engine/task.hpp>\n"
K_ENGINE_HEADER_INCLUDE = "#include <SushiBLAS/engine/engine.hpp>\n"


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
                "rule_seam_engine_includes",
                "rule_operation_engine_calls",
                "rule_view_engine_calls",
                "rule_graph_insertions",
                "rule_task_dependencies",
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
        """Reports engine.hpp, engine/, modules/kernels/ and tensor.hpp includes, each on its line."""
        files = {
            K_SUPPORT: (
                "#include <SushiBLAS/engine/engine.hpp>\n"
                '#include "SushiBLAS/engine/task.hpp"\n'
                "#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>\n"
                "#include <SushiBLAS/tensor/tensor.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_support_independence(_repository(folder, files)))
            self.assertEqual([issue.line for issue in issues], [1, 2, 3, 4])

    def test_reports_the_sparse_operand_headers(self) -> None:
        """Reports sparse_vector.hpp, sparse_matrix.hpp and sparse_builder.hpp, each on its line."""
        files = {
            K_SUPPORT: (
                "#include <SushiBLAS/sparse_storage/sparse_vector.hpp>\n"
                "#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>\n"
                "#include <SushiBLAS/sparse_storage/sparse_builder.hpp>\n"
                "#include <SushiBLAS/core/logger.hpp>\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_support_independence(_repository(folder, files)))
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


class SeamEngineIncludesTest(unittest.TestCase):
    """Checks that only the views the umbrella lists include engine.hpp."""

    K_UMBRELLA = "#include <SushiBLAS/engine/engine.hpp>\n#include <SushiBLAS/blas/blas.hpp>\n"

    def test_accepts_a_view_the_umbrella_lists(self) -> None:
        """Reports nothing for a view header that includes engine.hpp."""
        files = {
            "modules/library/sushiblas/include/SushiBLAS/SushiBLAS.h": self.K_UMBRELLA,
            "modules/operations/blas/include/SushiBLAS/blas/blas.hpp": K_ENGINE_HEADER_INCLUDE,
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_seam_engine_includes(_repository(folder, files))), [])

    def test_reports_a_seam_and_an_operation_source(self) -> None:
        """Reports a dispatch seam and an operation source that include engine.hpp."""
        files = {
            "modules/library/sushiblas/include/SushiBLAS/SushiBLAS.h": self.K_UMBRELLA,
            "modules/operations/blas/include/SushiBLAS/blas/dispatch.hpp": K_ENGINE_HEADER_INCLUDE,
            "modules/operations/blas/source/level1/dot.cpp": K_ENGINE_HEADER_INCLUDE,
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_seam_engine_includes(_repository(folder, files)))
            self.assertEqual([issue.path.name for issue in issues], ["dispatch.hpp", "dot.cpp"])


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
            "modules/foundation/support/include/SushiBLAS/support/reduce_1d.hpp": (
                "return queue.submit([&](sycl::handler& h)\n"
                "{\n"
                "    h.depends_on(staged);\n"
                "    h.single_task(fold);\n"
                "});\n"
            ),
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_task_dependencies(_repository(folder, files))), [])


if __name__ == "__main__":
    unittest.main()

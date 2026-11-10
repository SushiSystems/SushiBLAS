# test_check_tier_contents.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the foundation, kernel, Engine, view and recorder rules of check_tier_contents.py."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layering.check_tier_contents import (  # noqa: E402
    K_CHECKER,
    K_FOUNDATION_FORBIDDEN_INCLUDE,
    K_KERNEL_FORBIDDEN_INCLUDE,
    K_OPERATION_ALLOWED_MEMBERS,
    KnownViolation,
    Repository,
    _engine_calls,
    _include_matches,
    engine_header_includes_in,
    graph_insertions_in,
    known_notes,
    rule_kernel_includes,
    rule_operation_engine_calls,
    stale_known_keys,
)

K_KERNEL = "modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/reference/add.hpp"
K_ENGINE_INCLUDE = "#include <SushiBLAS/engine/task.hpp>\n"
K_RECORDER_HEADER = "modules/engine/engine/include/SushiBLAS/engine/task_recorder.hpp"
K_GRAPH_SINK = "modules/engine/engine/source/graph_task_sink.cpp"
K_PRIMITIVE = "modules/foundation/graph_primitives/include/SushiBLAS/graph_primitives/reduce.hpp"
K_SOURCE = "modules/operations/logic/source/all.cpp"
K_RECORDING_VERBS = (
    "task", "kernel", "state_rule", "host", "untracked_kernel", "dynamic_kernel", "coupled_rule",
    "raw_kernel", "captured_kernel", "untracked_raw_kernel",
)
K_BRICKS = (
    "#include <SushiBLAS/tensor/tensor.hpp>\n"
    "#include <SushiBLAS/tensor/tensor_factory.hpp>\n"
    "#include <SushiBLAS/engine/rng_stream.hpp>\n"
    "#include <SushiBLAS/engine/scratch_pool.hpp>\n"
    "#include <SushiBLAS/engine/execution_mode.hpp>\n"
    "#include <SushiBLAS/engine/mode_admission.hpp>\n"
    "#include <SushiBLAS/engine/scratch_block_source.hpp>\n"
)


def _lines(text: str, path: str = "file.hpp") -> list[tuple[Path, int, str]]:
    """Returns the numbered lines of a text, each with the path it belongs to."""
    return [(Path(path), number, line) for number, line in enumerate(text.splitlines(), 1)]


def _numbers(text: str, pattern) -> list[int]:
    """Returns the line of each include in the text that the pattern forbids."""
    return [violation.line for violation in _include_matches(_lines(text), pattern)]


def _insertions(text: str, path: str = K_SOURCE) -> list[str]:
    """Returns the verb of each recording call in the text of a file at path."""
    return [violation.symbol for violation in graph_insertions_in(_lines(text, path), lambda found: found.as_posix())]


class TestRegistration:
    """Checks that the rule table carries the five SushiBLAS layer rules."""

    K_TEST_SIZE = "small"

    def test_is_registered(self) -> None:
        """Lists every layer rule in its order."""
        assert list(K_CHECKER.rules) == [
            "rule_foundation_independence",
            "rule_kernel_includes",
            "rule_engine_header_includes",
            "rule_operation_engine_calls",
            "rule_graph_insertions",
        ]


class TestFoundationIndependence:
    """Checks which includes a foundation header may make."""

    K_TEST_SIZE = "small"

    def test_accepts_core_foundation_and_sycl(self) -> None:
        """Reports nothing for a foundation header that includes core, foundation and SYCL headers."""
        text = (
            "#include <sycl/sycl.hpp>\n"
            "#include <SushiBLAS/core/traits.hpp>\n"
            "#include <SushiBLAS/launch/device_policy.hpp>\n"
        )
        assert _numbers(text, K_FOUNDATION_FORBIDDEN_INCLUDE) == []

    @pytest.mark.parametrize("text, expected", [
        pytest.param(
            "#include <SushiBLAS/engine/engine.hpp>\n"
            '#include "SushiBLAS/engine/task.hpp"\n'
            "#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>\n"
            "#include <SushiBLAS/tensor/tensor.hpp>\n",
            [1, 2, 3, 4], id="engine-kernel-and-tensor"),
        pytest.param(
            "#include <SushiBLAS/sparse_storage/sparse_vector.hpp>\n"
            "#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>\n"
            "#include <SushiBLAS/sparse_storage/sparse_builder.hpp>\n"
            "#include <SushiBLAS/core/logger.hpp>\n",
            [1, 2, 3], id="sparse-operands"),
    ])
    def test_reports_the_forbidden_includes(self, text: str, expected: list[int]) -> None:
        """Reports each forbidden include on its line and leaves a core include."""
        assert _numbers(text, K_FOUNDATION_FORBIDDEN_INCLUDE) == expected


class TestKernelIncludes:
    """Checks which includes a kernel may make."""

    K_TEST_SIZE = "small"

    def test_accepts_a_clean_kernel(self) -> None:
        """Reports nothing for a kernel that includes only core headers."""
        assert _numbers("#include <SushiBLAS/core/common.hpp>\n", K_KERNEL_FORBIDDEN_INCLUDE) == []

    @pytest.mark.parametrize("text, expected", [
        pytest.param(
            "#include <SushiBLAS/engine/engine.hpp>\n"
            '#include "SushiBLAS/engine/task.hpp"\n'
            "#include <SushiBLAS/tensor/tensor.hpp>\n",
            [1, 2, 3], id="engine-and-tensor"),
        pytest.param(
            "#include <SushiBLAS/sparse_storage/sparse_vector.hpp>\n"
            "#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>\n"
            "#include <SushiBLAS/sparse_storage/sparse_builder.hpp>\n",
            [1, 2, 3], id="sparse-operands"),
    ])
    def test_reports_the_forbidden_includes(self, text: str, expected: list[int]) -> None:
        """Reports each forbidden include on its line."""
        assert _numbers(text, K_KERNEL_FORBIDDEN_INCLUDE) == expected

    def test_accepts_a_private_kernel_brick(self) -> None:
        """Reports nothing for an include of a brick the kernel owns."""
        assert _numbers('#include "kernels/sparse/csr_pattern.hpp"\n', K_KERNEL_FORBIDDEN_INCLUDE) == []

    def test_reports_a_stale_entry(self) -> None:
        """Names the tabled key that no violation matched."""
        known = [(K_KERNEL, "SushiBLAS/engine/task.hpp"), (K_KERNEL, "SushiBLAS/tensor/tensor.hpp")]
        seen = [(K_KERNEL, "SushiBLAS/tensor/tensor.hpp")]
        assert stale_known_keys(known, seen) == [(K_KERNEL, "SushiBLAS/engine/task.hpp")]


class TestKnownEntries:
    """Checks how a tabled violation is screened, over a folder."""

    K_TEST_SIZE = "medium"

    def _repository(self, folder: str, text: str) -> Repository:
        """Returns a repository whose kernel header holds the text and whose table lists the engine include."""
        path = Path(folder) / K_KERNEL
        path.parent.mkdir(parents=True)
        path.write_text(text, encoding="utf-8")
        known = (KnownViolation("rule_kernel_includes", K_KERNEL, "SushiBLAS/engine/task.hpp", "r"),)
        return Repository(Path(folder), known)

    def test_a_known_entry_reports_nothing(self) -> None:
        """Reports no issue for a tabled violation."""
        with tempfile.TemporaryDirectory() as folder:
            assert list(rule_kernel_includes(self._repository(folder, K_ENGINE_INCLUDE))) == []

    def test_a_known_entry_is_named_in_the_notes(self) -> None:
        """Names the tabled violation in exactly one note."""
        with tempfile.TemporaryDirectory() as folder:
            notes = list(known_notes(self._repository(folder, K_ENGINE_INCLUDE), ["rule_kernel_includes"]))
        assert len(notes) == 1 and K_KERNEL in notes[0]

    def test_a_stale_entry_is_an_issue(self) -> None:
        """Reports the tabled violation that no longer occurs."""
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_kernel_includes(self._repository(folder, "#include <SushiBLAS/core/common.hpp>\n")))
        assert len(issues) == 1 and "stale" in issues[0].message


class TestEngineHeaderIncludes:
    """Checks that engine.hpp includes only its bricks and no view, seam or experimental header."""

    K_TEST_SIZE = "small"

    def test_accepts_the_bricks_engine_composes(self) -> None:
        """Reports nothing for the tensor header and the engine bricks."""
        assert list(engine_header_includes_in(_lines(K_BRICKS))) == []

    def test_reports_a_view_a_seam_io_and_experimental(self) -> None:
        """Reports a family view, a dispatch seam, io.hpp and an experimental header."""
        text = (
            "#include <SushiBLAS/random/random.hpp>\n"
            "#include <SushiBLAS/engine/task.hpp>\n"
            "#include <SushiBLAS/io/io.hpp>\n"
            "#include <SushiBLAS/experimental/lapack/linalg.hpp>\n"
        )
        assert [violation.symbol for violation in engine_header_includes_in(_lines(text))] == [
            "SushiBLAS/random/random.hpp",
            "SushiBLAS/engine/task.hpp",
            "SushiBLAS/io/io.hpp",
            "SushiBLAS/experimental/lapack/linalg.hpp",
        ]


class TestOperationEngineCalls:
    """Checks that an operation source calls nothing on an Engine."""

    K_TEST_SIZE = "small"

    def _members(self, text: str) -> list[str]:
        """Returns the member of each engine call in the text."""
        return [violation.symbol for violation in _engine_calls(_lines(text), K_OPERATION_ALLOWED_MEMBERS)]

    def test_accepts_a_recorder_call(self) -> None:
        """Reports nothing for a call on the recorder."""
        assert self._members("recorder_.record_pair(description, operands, family, body);\n") == []

    def test_accepts_an_engine_call_in_a_comment(self) -> None:
        """Reports nothing for an engine call that sits in a comment."""
        assert self._members("// engine_.mode();\n") == []

    def test_reports_every_member(self) -> None:
        """Reports recorder through a reference and create_tensor through a pointer."""
        assert self._members("engine_.recorder();\nengine->create_tensor({});\n") == ["recorder", "create_tensor"]


class TestOperationEngineCallsOfARepository:
    """Checks the rule over a folder."""

    K_TEST_SIZE = "medium"

    def test_reports_the_call_of_a_source_file(self) -> None:
        """Reports the member an operation source calls on an engine."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "modules/operations/blas/source/a.cpp"
            path.parent.mkdir(parents=True)
            path.write_text("engine_.recorder();\n", encoding="utf-8")
            issues = list(rule_operation_engine_calls(Repository(Path(folder), ())))
        assert [issue.message for issue in issues] == ["layering violation: recorder"]


class TestGraphInsertions:
    """Checks that TaskRecorder, its graph sink and the graph primitives are the only callers of the recording verbs."""

    K_TEST_SIZE = "small"

    @pytest.mark.parametrize("path, text", [
        pytest.param(K_RECORDER_HEADER, "recorder_->task(access, work, options);\n", id="the-recorder-header"),
        pytest.param(K_GRAPH_SINK, "recorder_->task(access, work, options);\n", id="the-graph-sink"),
        pytest.param(K_PRIMITIVE, "recorder.kernel(count, access, body, options);\n", id="a-graph-primitive"),
    ])
    def test_accepts_a_file_that_may_record(self, path: str, text: str) -> None:
        """Reports nothing for a recording call in a file allowed to make it."""
        assert _insertions(text, path) == []

    def test_accepts_a_comment_naming_a_verb(self) -> None:
        """Reports nothing for a comment that names a verb."""
        assert _insertions("// recorder.task(access, work) is the recorder's call.\n") == []

    def test_accepts_a_record_call(self) -> None:
        """Reports nothing for TaskRecorder::record_pair."""
        assert _insertions("recorder_.record_pair(description, operands, family, body);\n") == []

    def test_accepts_a_free_function_named_like_a_verb(self) -> None:
        """Reports nothing for a function that is not a member call."""
        assert _insertions("kernel(q, deps);\n") == []

    @pytest.mark.parametrize("path, text, expected", [
        pytest.param("modules/operations/logic/include/SushiBLAS/logic/dispatch.hpp",
                     "graph.recorder().task(access, body, options);\n", ["task"], id="a-seam"),
        pytest.param(K_SOURCE, "recorder->host(access, work);\n", ["host"], id="a-source"),
    ])
    def test_reports_a_recording_call_outside_the_recorder(self, path: str, text: str, expected: list[str]) -> None:
        """Reports the verb called in a seam and in an operation source."""
        assert _insertions(text, path) == expected

    @pytest.mark.parametrize("verb", K_RECORDING_VERBS)
    def test_reports_every_recording_verb(self, verb: str) -> None:
        """Reports each member of SushiRuntime::Recorder that records a node."""
        assert _insertions(f"recorder.{verb}(arguments);\n") == [verb]

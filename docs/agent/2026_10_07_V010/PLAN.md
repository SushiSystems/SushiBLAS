# Plan: the v0.1.0 interface break

This file holds the three package tables of the work and their waves: the first round (`PLAN`,
waves 0 to 6), the second round (`PLAN_EXT`, waves 7 to 11) and the YOLO26 packages
(`DESIGN_YOLO26_PACKAGES`, which numbers its own waves 7 to 9).

The three texts are as their authors wrote them on 2026-10-07, before any package was
dispatched, so each status line still says "no package dispatched" and each wave table still
names the build that was to close it. No record shows any of those builds running. `REPORT.md` says
what landed, in which commit, and what did not.

Each text is unchanged apart from two things: its headings sit one level lower, and its title
line became a heading that carries the old file name with the title in parentheses. `SPEC.md`,
under "Names used in this folder", maps every old file name to its heading.

| Heading | Holds | Was |
| --- | --- | --- |
| `PLAN` | 32 packages in waves 0 to 6, the build-configuration lines per wave, the decisions taken on the designers' recommendation, what waits, what is open | `PLAN.md` before the merge |
| `PLAN_EXT` | 41 packages in waves 7 to 11, one owner per shared file, the build-configuration lines, the findings of the second design review | `PLAN_EXT.md` |
| `DESIGN_YOLO26_PACKAGES` | 12 packages YL-1 to YL-12, their waves and build-configuration lines | section 2 of `DESIGN_YOLO26.md` |

## PLAN (Plan: the v0.1.0 interface break)

**Status:** Open — no package dispatched. Written on 2026-10-07, after the owner answered the
open points of `DESIGN_REVIEW.md`.

This file is the landing order of the three designs in this folder: `DESIGN_RECORDER.md`,
`DESIGN_BLAS.md` and `DESIGN_SUPPORT_MEMORY.md`. Each package below is one dispatch. A package's
text is in the design section its row names; this file adds nothing to it but the order, the
waves and the build-configuration lines the orchestrator writes.

Nothing was built to write it. No `sb`, `cmake`, `ninja` or `ctest` process was started and no
source file was edited; another session was compiling the tree.

### 0. Ground rules

- Every dispatch opens with the seven lines of the owner's delegation block, verbatim, runs on
  `opus`, and carries the effort this plan gives it. A `high` dispatch needs the owner's word
  first. This plan has five: R4, R6, SM-7, BLAS-4 and BLAS-5.
- An implementer never builds and never edits a build configuration file. It checks its C++ with
  `clang -fsyntax-only` and the flags of a sibling entry of `build/compile_commands.json`. The
  orchestrator makes the edits of section 3 and runs `sb build` and `sb test` to close a wave.
- The owner chooses how the plan runs. Section 2 gives it in waves; the same order, read row by
  row, is the inline form.
- A package is done when the one thing its row of section 2 names holds.

Already in the tree, and not planned here: `tools/versioning/check_version.py` with its rows in
`AGENTS.md`, the CI workflow and `tools/README.md` and its changelog line (commits `5993a1e`,
`813676e`); the head of `NOTICE.md`; the Doxygen project number; `cmake/SyclTarget.cmake`
following the device targets SushiRuntime resolved (commit `535e83c`). A CUDA build of the tree
exists. On an RTX 3080 Ti 790 of its 792 tests pass, one is skipped by design in a release build
(`Regression_LoggerDeathTest.AssertWritesItsMessageBeforeAborting`) and one fails
(`Unit_Level2SyclParityTest.SymvMatchesReferenceLarge<float>`). BLAS-1, BLAS-4 and BLAS-10 carry
the fix for the failure.

Deleted from the plan: SM-10 (`sb` device gate) and SM-11 (GPU gate test). The owner had
`--device` removed in commit `358e746`, because `sr` and `se` have no such flag and leave the
device to the runtime and the build. What is left is three sentences of documentation in the
closing commit (`DESIGN_SUPPORT_MEMORY.md`, section 12.1).

### 1. Packages in landing order

Paths are relative to `D:\Projects\sushiblas` unless they begin with `sushiruntime/`. "New"
marks a file the package creates. A glob appears only where a design lists one. "Needs" names
what has to be on disk before the package is written, so that its syntax check can run.

| Wave | Id | Title | Files | Needs | Effort | Read |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | SR-1 | Aligned shared and device allocation, in SushiRuntime | `sushiruntime/include/SushiRuntime/execution/runtime_context.hpp`, `sushiruntime/src/execution/runtime_context.cpp`, `sushiruntime/tests/functional/unit/test_topology.cpp` | none | low: the text is given | support 3 |
| 0 | SM-2 | Scratch shelf and pool | New: `include/SushiBLAS/engine/scratch_shelf.hpp`, `include/SushiBLAS/engine/scratch_pool.hpp`, `tests/functional/unit/test_scratch_shelf.cpp` | none | low: both headers are given in full and parsed | support 5.1 to 5.3, 5.5 |
| 0 | SM-4 | Launch shapes and device profile | `include/SushiBLAS/support/device_policy.hpp`, `include/SushiBLAS/support/launch_1d.hpp`, `kernels/sycl/random/orthonormalize.hpp`. New: `tests/functional/unit/test_launch_shapes.cpp` | none | low: public headers, every changed function given verbatim | support 6 |
| 0 | SM-5 | Record-time bricks | `include/SushiBLAS/engine/operand_check.hpp`, `tests/functional/unit/test_operand_check.cpp`. New: `include/SushiBLAS/engine/math/elementwise/scalar_binding.hpp` | none | low: additions given verbatim | support 7 |
| 0 | SM-12 | Changelog row of the version checker | `tools/versioning/check_version.py`, `tools/tests/test_check_version.py` | none | low | support 12.3 |
| 0 | SM-15 | `sb docs` writes under the build tree | `cli/sushiblas/services/project.py`, `cli/tests/test_project.py` | none | low: one line and one test | support 12.2 |
| 0 | BLAS-1 | Value types and kernel bricks | `include/SushiBLAS/engine/math/safe_math.hpp`. New: `include/SushiBLAS/core/blas_types.hpp`, `include/SushiBLAS/support/complex_arithmetic.hpp`, `include/SushiBLAS/support/running_solution.hpp`, `include/SushiBLAS/support/reference_accumulator.hpp`, `include/SushiBLAS/kernels/blas/conjugation.hpp`, `include/SushiBLAS/kernels/blas/operand_access.hpp`, `include/SushiBLAS/engine/blas/scalar.hpp` | none | low: public headers, all seven given in full and parsed | BLAS 4 |
| 1 | R1 | Recorder bricks, `Storage` size, the scratch lease | `include/SushiBLAS/storage.hpp`, `include/SushiBLAS/tensor.hpp`, `include/SushiBLAS/engine/scratch.hpp`. New: `include/SushiBLAS/engine/execution_mode.hpp`, `include/SushiBLAS/engine/kernel_pair_dispatcher.hpp`, `include/SushiBLAS/engine/task_handle.hpp`, `include/SushiBLAS/engine/operand_set.hpp`, `include/SushiBLAS/engine/task_recorder.hpp`, `include/SushiBLAS/tensor_factory.hpp` | SM-2 | low: public headers, every one given in full and parsed | recorder 4, 5.2 to 5.4; support 5.4 for `scratch.hpp` |
| 1 | R7 | Layering checker | `tools/layering/check_layering.py`, `tools/tests/test_check_layering.py` | none | low: the text is given and was run | recorder 11 |
| 2 | R2 | `Engine`, umbrella, typed task | `include/SushiBLAS/engine.hpp`, `include/SushiBLAS/SushiBLAS.h`, `include/SushiBLAS/engine/task.hpp`, `src/engine.cpp`, `src/io.cpp` | R1 | low: three headers given in full and parsed | recorder 5.1, 5.5 to 5.7 |
| 2 | R3 | Elementwise, nonlinear and logic seams and sources | `include/SushiBLAS/engine/math/elementwise.hpp`, `include/SushiBLAS/engine/math/nonlinear.hpp`, `include/SushiBLAS/engine/math/elementwise/dispatch.hpp`, `include/SushiBLAS/engine/math/nonlinear/dispatch.hpp`, `include/SushiBLAS/engine/logic/logic.hpp`, `include/SushiBLAS/engine/logic/dispatch.hpp`, `src/engine/math/elementwise/**`, `src/engine/math/nonlinear/**`, `src/engine/logic/**` | R1 | low: one rule set, one worked seam, one worked source | recorder 6, 6.1, 7, 8, 8.2 |
| 2 | R4 | Reduction and random seams and sources | `include/SushiBLAS/engine/math/reductions.hpp`, `include/SushiBLAS/engine/math/random.hpp`, `include/SushiBLAS/engine/math/reductions/dispatch.hpp`, `include/SushiBLAS/engine/math/random/dispatch.hpp`, `src/engine/math/reductions/**`, `src/engine/math/random/**` | R1 | high: public seam headers derived from rules; six seams record inside dtype arms, where the intel-llvm host-kernel fault lives | recorder 6, 6.2 to 6.4, 7, 8 |
| 2 | R5 | Spatial and experimental | `include/SushiBLAS/engine/spatial.hpp`, `src/engine/spatial/**`, `include/SushiBLAS/experimental/lapack/linalg.hpp`, `include/SushiBLAS/experimental/signal/transforms.hpp` | R1 | low | recorder 7, 8 |
| 2 | R6 | BLAS, recorder migration only | `include/SushiBLAS/engine/blas.hpp`, `include/SushiBLAS/engine/blas/level1.hpp`, `include/SushiBLAS/engine/blas/level2.hpp`, `include/SushiBLAS/engine/blas/level3.hpp`, `include/SushiBLAS/engine/blas/dispatch.hpp`, `include/SushiBLAS/engine/blas/utils.hpp`, `src/engine/blas/**` | R1 | high: public headers derived from rules, and nineteen sources moved onto `OperandSet` | recorder 6.5, 6.6, 7, 8, 8.1 |
| 2 | R8 | Call-site sweep, hand edits, graph probe | New: `tests/common/graph_probe.hpp`. `tests/common/*.hpp`, `tests/common/*.cpp`, `tests/functional/**/*.cpp`, `tests/package/main.cpp`, `examples/*.cpp`, `benchmarks/*.cpp`, `benchmarks/*.hpp` | R1 | low: a script, a table of hand edits and one header given in full | recorder 12.1 to 12.3, 12.6 |
| 2 | BLAS-2 | BLAS record-time bricks | `include/SushiBLAS/engine/blas/matrix_operand.hpp`. New: `include/SushiBLAS/engine/blas/vector_operand.hpp`, `include/SushiBLAS/engine/blas/argument_check.hpp`, `include/SushiBLAS/engine/blas/task_metadata.hpp` | BLAS-1, SM-5 | low: given in full and parsed | BLAS 5 |
| 3 | R9 | Recorder tests | New: `tests/functional/unit/test_task_handle.cpp`, `tests/functional/unit/test_task_recorder.cpp`, `tests/functional/unit/test_tensor_factory.cpp`, `tests/functional/regression/test_operand_lifetime.cpp`, `tests/functional/regression/test_task_dependencies.cpp` | R2 to R8 | low; the reviewer reads `Regression_TaskDependencies` closely, it is the one test here that is not a table row | recorder 13.2, 6.7, 12.6 |
| 3 | SM-1 | Storage alignment and tensor costs | `include/SushiBLAS/storage.hpp`, `include/SushiBLAS/tensor.hpp`, `tests/functional/unit/test_storage_offset.cpp` | SR-1, R1, R8 | low: `storage.hpp` given in full | support 4 |
| 3 | SM-3 | Tests of the scratch lease | New: `tests/functional/unit/test_scratch_buffer.cpp` | R1, R2, R4, R8 | low | support 5.5 |
| 3 | SM-6 | Elementwise, nonlinear, logic and random executors | `include/SushiBLAS/engine/dispatch.hpp`, `include/SushiBLAS/engine/math/elementwise/dispatch.hpp`, `include/SushiBLAS/engine/math/nonlinear/dispatch.hpp`, `include/SushiBLAS/engine/logic/dispatch.hpp`, `include/SushiBLAS/engine/math/random/dispatch.hpp`, `src/engine/math/elementwise/clamp.cpp`, `src/engine/math/elementwise/pow.cpp`, `src/engine/math/elementwise/convert.cpp`, `src/engine/math/nonlinear/elu.cpp`, `src/engine/math/nonlinear/leaky_relu.cpp`, `src/engine/logic/all.cpp`, `src/engine/logic/any.cpp`, `src/engine/math/random/constant.cpp`, `src/engine/math/random/orthogonal.cpp`, `src/engine/math/random/shuffle.cpp`, `src/engine/math/random/truncated_normal.cpp`, `tests/functional/unit/test_integer_dtypes.cpp`, `tests/functional/unit/test_dispatch_complex.cpp`, `tests/functional/unit/test_dispatch_seam.cpp`, `tests/functional/unit/test_math_validation.cpp`, `tests/functional/unit/test_complex_nonlinear.cpp`, `tests/functional/unit/test_integer_arithmetic.cpp`, `tests/functional/unit/test_distributions.cpp`, `tests/functional/unit/test_orthogonal.cpp` | SM-5, R3, R4, R8 | low: hunks keyed on lines the recorder waves keep | support 8 |
| 3 | SM-7 | Reductions | `include/SushiBLAS/support/reduce_1d.hpp`, `include/SushiBLAS/engine/math/reductions.hpp`, `include/SushiBLAS/engine/math/reductions/dispatch.hpp`, `src/engine/math/reductions/argmax.cpp`, `src/engine/math/reductions/argmin.cpp`, `src/engine/math/reductions/asum.cpp`, `src/engine/math/reductions/count_nonfinite.cpp`, `src/engine/math/reductions/logsumexp.cpp`, `src/engine/math/reductions/max.cpp`, `src/engine/math/reductions/max_along.cpp`, `src/engine/math/reductions/mean.cpp`, `src/engine/math/reductions/mean_spatial.cpp`, `src/engine/math/reductions/min.cpp`, `src/engine/math/reductions/nrm2.cpp`, `src/engine/math/reductions/prod.cpp`, `src/engine/math/reductions/std.cpp`, `src/engine/math/reductions/sum.cpp`, `src/engine/math/reductions/sum_along.cpp`, `src/engine/math/reductions/var.cpp`, `tests/functional/unit/test_reductions.cpp`, `tests/functional/unit/test_axis_reductions.cpp`, `tests/functional/unit/test_count_nonfinite.cpp`, `tests/functional/unit/test_max_min_nan.cpp`, `tests/functional/unit/test_reduction_determinism.cpp`, `tests/functional/unit/test_sum_edge_cases.cpp`, `tests/functional/unit/test_mean_spatial.cpp`, `tests/functional/unit/test_empty_tensors.cpp`. New: `include/SushiBLAS/engine/math/reductions/centred_reduction.hpp`, `include/SushiBLAS/engine/math/reductions/arg_reduction.hpp`, `src/engine/math/reductions/min_along.cpp`, `tests/functional/unit/test_reduce_plans.cpp` | SM-5, R4, R8; shares its build with SM-6 | high: a numeric kernel (the two-stage moments), four accumulators and the zero-element contract | support 9 |
| 3 | SM-8 | Spatial record-time checks | `src/engine/spatial/broadcast.cpp`, `src/engine/spatial/channels.cpp`, `src/engine/spatial/col2im.cpp`, `src/engine/spatial/im2col.cpp`, `src/engine/spatial/pool_avg.cpp`, `src/engine/spatial/pool_max.cpp`, `src/engine/spatial/pool_max_backward.cpp`, `src/engine/spatial/upsample_nearest.cpp`, `include/SushiBLAS/engine/spatial/window.hpp`, `tests/functional/unit/test_spatial_channels.cpp` | SM-5, R5, R8 | low: a site table | support 10 |
| 3 | SM-9 | IO without full-size staging | `src/io.cpp`, `include/SushiBLAS/io.hpp`, `tests/functional/regression/test_io.cpp`, `tests/functional/unit/test_host_transfer.cpp` | R2, R8 | low: the code is given | support 11 |
| 4 | BLAS-3 | Level 1 kernels | `kernels/reference/blas/level1/level1.hpp`, `kernels/sycl/blas/level1/level1.hpp` | BLAS-1 | low: both `dot` kernels given in full, four one-line SYCL edits, and two one-sentence edits of the reference `asum` and `nrm2` | BLAS 7.1 to 7.3, 7.10 |
| 4 | BLAS-4 | Level 2 kernels | `kernels/reference/blas/level2/gemv.hpp`, `kernels/reference/blas/level2/ger.hpp`, `kernels/reference/blas/level2/symv.hpp`, `kernels/reference/blas/level2/syr.hpp`, `kernels/reference/blas/level2/syr2.hpp`, `kernels/reference/blas/level2/trmv.hpp`, `kernels/sycl/blas/level2/gemv.hpp`, `kernels/sycl/blas/level2/ger.hpp`, `kernels/sycl/blas/level2/symv.hpp`, `kernels/sycl/blas/level2/syr.hpp`, `kernels/sycl/blas/level2/syr2.hpp`, `kernels/sycl/blas/level2/trmv.hpp` | BLAS-1 | high: twelve numeric kernels rebuilt from tables: flags, conjugation policies and, on the reference side, the sums of the finding | BLAS 7.1, 7.4, 7.5, 7.10 |
| 4 | BLAS-5 | Level 3 kernels, with rider RD-5 | `include/SushiBLAS/kernels/reference/blas/level3/gemm.hpp`, `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`, `kernels/reference/blas/level3/syrk.hpp`, `kernels/reference/blas/level3/trsm.hpp`, `kernels/sycl/blas/level3/syrk.hpp`, `kernels/sycl/blas/level3/trsm.hpp` | BLAS-1, SM-4 | high: GEMM, SYRK and TRSM kernels, two of them public headers; the HALF fix, the fit check and the reference sums | BLAS 7.1, 7.6 to 7.8, 7.10; support 15, RD-5 |
| 5 | BLAS-6 | BLAS public surface and seam | `include/SushiBLAS/engine/blas.hpp`, `include/SushiBLAS/engine/blas/level1.hpp`, `include/SushiBLAS/engine/blas/level2.hpp`, `include/SushiBLAS/engine/blas/level3.hpp`, `include/SushiBLAS/engine/blas/dispatch.hpp`, `include/SushiBLAS/engine/blas/utils.hpp` | BLAS-2, BLAS-5, R6 | low: six public headers, each given in full and parsed; nothing is derived | BLAS 6 |
| 6 | BLAS-7 | Level 1 operations | `src/engine/blas/level1/asum.cpp`, `src/engine/blas/level1/axpy.cpp`, `src/engine/blas/level1/copy.cpp`, `src/engine/blas/level1/dot.cpp`, `src/engine/blas/level1/iamax.cpp`, `src/engine/blas/level1/nrm2.cpp`, `src/engine/blas/level1/rot.cpp`, `src/engine/blas/level1/scal.cpp`, `src/engine/blas/level1/swap.cpp` | BLAS-3, BLAS-6 | low | BLAS 8.1, 8.3 |
| 6 | BLAS-8 | Level 2 operations | `src/engine/blas/level2/gemv.cpp`, `src/engine/blas/level2/ger.cpp`, `src/engine/blas/level2/symv.cpp`, `src/engine/blas/level2/syr.cpp`, `src/engine/blas/level2/syr2.cpp`, `src/engine/blas/level2/trmv.cpp`, `src/engine/blas/level2/trsv.cpp` | BLAS-4, BLAS-5, BLAS-6 | low by the rule of this plan; the review asked for high, see section 6 | BLAS 8 |
| 6 | BLAS-9 | Level 3 operations | `src/engine/blas/level3/gemm.cpp`, `src/engine/blas/level3/syrk.cpp`, `src/engine/blas/level3/trsm.cpp` | BLAS-5, BLAS-6 | low | BLAS 8.1, 8.3 |
| 6 | BLAS-10 | Existing BLAS tests, and the parity test on a truth | New: `tests/common/blas_flags.hpp`, `tests/common/blas_truth.hpp`. `tests/functional/regression/test_regression.cpp`, `tests/functional/unit/test_blas_half_accumulation.cpp`, `tests/functional/unit/test_blas_validation.cpp`, `tests/functional/unit/test_channel_bias_epilogue.cpp`, `tests/functional/unit/test_gemm.cpp`, `tests/functional/unit/test_level2_reference.cpp`, `tests/functional/unit/test_level2_sycl_parity.cpp`, `tests/functional/unit/test_symv.cpp`, `tests/functional/unit/test_syr.cpp`, `tests/functional/unit/test_syr2.cpp`, `tests/functional/unit/test_syrk.cpp`, `tests/functional/unit/test_syrk_trsm_parity.cpp`, `tests/functional/unit/test_trmv.cpp`, `tests/functional/unit/test_trsm.cpp`, `tests/functional/unit/test_trsm_reference.cpp`, `tests/functional/unit/test_trsv.cpp`, `tests/functional/unit/test_reference_mode_guard.cpp`, `tests/package/main.cpp`, and one line each of `tests/functional/unit/test_storage_offset.cpp`, `tests/functional/unit/test_axis_reductions.cpp` and the new `tests/functional/regression/test_operand_lifetime.cpp` | BLAS-1, BLAS-5, BLAS-6, SM-1, SM-7, R9 | low: a mapping table, two headers given in full and the test table of the parity file | BLAS 9, 10.1, 10.2 |
| 6 | BLAS-11 | New BLAS tests | New: `tests/functional/unit/test_blas_types.cpp`, `tests/functional/unit/test_blas_scalar.cpp`, `tests/functional/unit/test_complex_arithmetic.cpp`, `tests/functional/unit/test_operand_access.cpp`, `tests/functional/unit/test_running_solution.cpp`, `tests/functional/unit/test_reference_accumulator.cpp`, `tests/functional/unit/test_blas_complex.cpp`, `tests/functional/regression/test_blas_symmetric_bits.cpp` | BLAS-1, BLAS-3, BLAS-4, BLAS-5, BLAS-6 | low | BLAS 10.3 |
| after | R11 | SushiAI migration, by the sushiai session, once | `D:\Projects\sushiai` | wave 6 built | not this session's | recorder 12.5; BLAS 9.4; support 16 |
| after | SM-13 | Build configuration for the release, by the orchestrator | `src/CMakeLists.txt`, `tests/CMakeLists.txt`, `tests/package/CMakeLists.txt`, `cmake/SushiBLASConfig.cmake.in`, `cmake/Runtime.cmake`, `CMakePresets.json` | SushiRuntime v0.4.0 tagged | the orchestrator's | support 13 |
| after | R10, SM-14, BLAS 11 | One closing commit of records, by the orchestrator | the files of recorder 12.4, support 14 and BLAS 11 | every package above | the orchestrator's | recorder 12.4; support 12.1, 14; BLAS 11 |

Effort follows the owner's rule: `low` is mechanical work or a text given verbatim, and `high`
is kept for a numeric kernel or a public header that an implementer has to derive. A public
header whose whole text is in the design and was parsed there is `low`, and its row says so.

What the owner's answers changed in the order, against the table of `DESIGN_REVIEW.md`:

- SM-2 moved to the front. `TaskRecorder` takes a `ScratchPool&` from R1 on, so the pool has to
  exist before R1.
- The lease (`engine/scratch.hpp`) is written once, by R1, from the support design's text. The
  first plan changed its constructor in R1 and again in SM-3.
- SM-3 shrank to the lease's tests. Rider RD-1, the one token SM-3 changed in
  `engine/blas/utils.hpp`, and the joint build of SM-3 with SM-7 are gone.
- R6 writes the scratch line once, and BLAS-6 no longer waits on a support package.
- SM-10 and SM-11 are deleted. SM-15 keeps the one line of SM-10 that the documentation layout
  needs.
- SM-4, SM-5 and BLAS-1 moved to the front: none of them needs anything the recorder break
  changes, and landing them first gives them a build of their own.
- SM-7 joined the wave of SM-6, and BLAS-11 the wave of BLAS-7 to BLAS-10. In both cases the
  later package needs none of the earlier one's files on disk; it only has to be in the same
  build.

### 2. Waves

Inside a wave the file sets are pairwise disjoint and every "Needs" of every package is in an
earlier wave; section 7 has the script that checked both against the working tree. Seven waves,
four of which end with a build.

| Wave | Packages, with effort | Closes with | Tree compiles at the end |
| --- | --- | --- | --- |
| 0 | SR-1 (low), SM-2 (low), SM-4 (low), SM-5 (low), SM-12 (low), SM-15 (low), BLAS-1 (low) | `sr build` and `sr test` in sushiruntime for SR-1; `sb build`, `sb test`, the Python tests and the five checks of `AGENTS.md` here | yes |
| 1 | R1 (low), R7 (low) | no build; `python -m unittest discover -s tools/tests` for R7 | no |
| 2 | R2 (low), R3 (low), R4 (high), R5 (low), R6 (high), R8 (low), BLAS-2 (low) | `sb build`, `sb test`, the five checks | yes |
| 3 | R9 (low), SM-1 (low), SM-3 (low), SM-6 (low), SM-7 (high), SM-8 (low), SM-9 (low) | `sb build`, `sb test`, the five checks | yes |
| 4 | BLAS-3 (low), BLAS-4 (high), BLAS-5 (high) | no build | no |
| 5 | BLAS-6 (low) | no build | no |
| 6 | BLAS-7 (low), BLAS-8 (low), BLAS-9 (low), BLAS-10 (low), BLAS-11 (low) | `sb build`, `sb test`, the five checks; then the same on the CUDA build | yes |

Three waves end with a tree that does not compile. Each is forced, and each is followed by the
wave that closes it.

**Wave 1.** R1 gives `ScratchBuffer` its pool-taking constructor and makes `Storage`'s size a
call. Three call sites of the constructor (`engine/math/reductions/dispatch.hpp`, twice;
`engine/blas/utils.hpp`, once) and two test lines that read the size field then no longer
compile, and those five lines belong to R4, R6 and R8. R1 cannot move into wave 2 beside them:
the packages of wave 2 parse their seams against `task_recorder.hpp` and `operand_set.hpp`, so
the bricks have to be on disk first. Keeping the old constructor alive for one wave would avoid
the gap and would be the two-step spelling the owner ruled out.

**Wave 4.** BLAS-3, BLAS-4 and BLAS-5 change kernel signatures from `bool` to the flag
enumerations under their callers.

**Wave 5.** BLAS-6 changes the public BLAS signatures under the operation sources and the tests.
It cannot share wave 4, because `engine/blas/dispatch.hpp` calls the GEMM kernels and parses
only against their new signatures. The sources of wave 6 cannot share wave 5 for the same
reason, one level up.

Waves 4 to 6 are one build unit, as waves 1 and 2 are. No build is asked for inside either.

What "done" means for each package. An implementer reports the syntax check; the orchestrator
judges the row at the build that closes the wave.

| Package | Done when |
| --- | --- |
| SR-1 | The three new `Unit_TopologyTest` cases pass in sushiruntime |
| SM-2 | `Unit_ScratchShelf` passes |
| SM-4 | `Unit_LaunchShapes` passes and every existing test still does |
| SM-5 | The eight new cases of `test_operand_check.cpp` pass |
| SM-12 | The four new cases of `tools/tests/test_check_version.py` pass |
| SM-15 | `cd cli && python -m pytest tests -q` passes with the new case |
| BLAS-1 | The tree builds with the new block of `safe_math.hpp`, and each new header parses alone |
| R1 | Each new header, and `scratch.hpp`, parses alone under the project's warning flags |
| R7 | `python -m unittest discover -s tools/tests` passes |
| R2 to R6, R8 | The wave-2 build passes; every existing test passes, with `SetModeRefusesVerifyAndKeepsTheMode` the one test deleted; `python tools/layering/check_layering.py .` prints no finding and no note; the sweep script, run again, leaves no `get_graph` line |
| BLAS-2 | The wave-2 build passes with the fp64 check in `describe_matrix_operand` |
| R9 | The five new test files pass |
| SM-1 | The four new cases of `test_storage_offset.cpp` pass |
| SM-3 | `Unit_ScratchBuffer` passes |
| SM-6 | The flipped and the new cases of its eight test files pass; `grep -rn "std::vector<Param>" include/SushiBLAS/engine` prints nothing |
| SM-7 | The cases of support 9.8 pass; the six `rule_block_ceiling` findings of its files are gone |
| SM-8 | `grep -n SB_THROW_IF src/engine/spatial include/SushiBLAS/engine/spatial` prints nothing and the spatial tests pass |
| SM-9 | The seven new cases of `test_io.cpp` and `test_host_transfer.cpp` pass |
| BLAS-3 to BLAS-5 | `grep -nE "std::conj\|\) \* W\(\|\) / W\(" kernels/sycl/blas include/SushiBLAS/kernels/sycl/blas` prints nothing, and each header parses alone |
| BLAS-6 | The six headers parse in one scratch translation unit |
| BLAS-7 to BLAS-9 | `grep -rn SB_THROW_IF include/SushiBLAS/engine/blas src/engine/blas` prints nothing and the wave-6 build passes |
| BLAS-10 | Every existing BLAS test passes; both kernels are inside their bounds in `Unit_Level2SyclParityTest` on the CPU build and on the CUDA build |
| BLAS-11 | The eight new test files pass |

The wave-6 build repeated on the CUDA build is part of closing that wave. It is what shows the
finding of `DESIGN_BLAS.md`, section 7.10, closed, and it is the first compile of
`Kernels::multiply` and `Kernels::divide` for an ahead-of-time target.

### 3. Build-configuration edits, per wave

The orchestrator makes these. No implementer does. Lines are verbatim, with their four leading
spaces; none was run.

#### Wave 0

`tests/CMakeLists.txt`, in `add_executable(sushiblas_tests ...)`, a new group directly above
the line `    # --- Integration ---`:

```cmake
    # --- Support layer and scratch ---
    functional/unit/test_scratch_shelf.cpp
    functional/unit/test_launch_shapes.cpp

```

#### Wave 1

None. R1's headers are compiled for the first time by the wave-2 build.

#### Wave 2

None. `tests/common/graph_probe.hpp` is a header, and BLAS-2 adds headers only.

#### Wave 3

`tests/CMakeLists.txt`, appended to the group wave 0 added:

```cmake
    functional/unit/test_scratch_buffer.cpp

    # --- Recorder, handles and the tensor factory ---
    functional/unit/test_task_handle.cpp
    functional/unit/test_task_recorder.cpp
    functional/unit/test_tensor_factory.cpp
```

In the group `# --- Reductions ---`, after `    functional/unit/test_count_nonfinite.cpp`:

```cmake
    functional/unit/test_reduce_plans.cpp
```

In the group `# --- Regression ---`, after `    functional/regression/test_logger.cpp`:

```cmake
    functional/regression/test_operand_lifetime.cpp
    functional/regression/test_task_dependencies.cpp
```

`src/CMakeLists.txt`, in `SUSHIBLAS_SOURCES`, after `    engine/math/reductions/max_along.cpp`:

```cmake
    engine/math/reductions/min_along.cpp
```

#### Waves 4 and 5

None.

#### Wave 6

`tests/CMakeLists.txt`, in the group `# --- Level 3 ---`, after
`    functional/unit/test_trsm.cpp`:

```cmake
    functional/unit/test_blas_types.cpp
    functional/unit/test_blas_scalar.cpp
    functional/unit/test_complex_arithmetic.cpp
    functional/unit/test_operand_access.cpp
    functional/unit/test_running_solution.cpp
    functional/unit/test_reference_accumulator.cpp
    functional/unit/test_blas_complex.cpp
```

In the group `# --- Regression ---`, after the two lines of wave 3:

```cmake
    functional/regression/test_blas_symmetric_bits.cpp
```

No `gpu` label and no fourth `gtest_discover_tests` block: SM-11 is deleted. The three existing
blocks pick the new suites up by their `Unit_` and `Regression_` prefixes.

#### After the last wave: SM-13

The install rule. `src/CMakeLists.txt:234-236` installs all of `include/`. Replace the rule:

```cmake
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
        PATTERN "experimental" EXCLUDE
)
```

The proof of it. `tests/package/CMakeLists.txt`, after `find_package(SushiBLAS REQUIRED)`:

```cmake
get_target_property(_sb_include_dirs SushiBLAS::sushiblas INTERFACE_INCLUDE_DIRECTORIES)
foreach(_sb_include_dir IN LISTS _sb_include_dirs)
    if(EXISTS "${_sb_include_dir}/SushiBLAS/experimental")
        message(FATAL_ERROR "The installed package ships SushiBLAS/experimental; it must not.")
    endif()
    file(GLOB_RECURSE _sb_stray_documents "${_sb_include_dir}/SushiBLAS/*.md")
    if(_sb_stray_documents)
        message(FATAL_ERROR "The installed package ships documents among its headers: ${_sb_stray_documents}")
    endif()
endforeach()
```

In the same file the stale path `docs/INTEGRATION.md` becomes `docs/guides/INTEGRATION.md`.

The SushiRuntime pin, once v0.4.0 is tagged:

| File | Old | New |
| --- | --- | --- |
| `cmake/SushiBLASConfig.cmake.in:26` | `find_dependency(SushiRuntime)` | `find_dependency(SushiRuntime 0.4)` |
| `cmake/Runtime.cmake`, mode 2 | `find_package(SushiRuntime QUIET)` | `find_package(SushiRuntime 0.4 QUIET)` |
| `cmake/Runtime.cmake`, mode 3, before `add_subdirectory` | nothing | the block below |

```cmake
    file(STRINGS "${SUSHIRUNTIME_DIR}/CMakeLists.txt" _sb_runtime_project
         REGEX "^project\\(SushiRuntime VERSION ")
    string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" _sb_runtime_version "${_sb_runtime_project}")
    if(NOT "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}" STREQUAL "0.4")
        message(FATAL_ERROR
            "SushiBLAS 0.1 builds against SushiRuntime 0.4; the checkout at "
            "${SUSHIRUNTIME_DIR} declares ${_sb_runtime_version}.")
    endif()
```

`CMakePresets.json` waits for the owner's answer (section 6).

SM-1 builds only against a SushiRuntime that has SR-1. Wave 0 lands SR-1 in the sibling checkout
this tree builds against, so the wave-3 build finds it; the pin above is what stops a consumer
with an older runtime.

### 4. Decisions taken on the designers' recommendation

The owner asked that every remaining small point take its design's own recommendation. They are
collected here so they can be read in one place. Each can be reversed by saying so before the
package that carries it is dispatched.

The recorder (`DESIGN_RECORDER.md`, section 15):

| Decision | Package |
| --- | --- |
| The seams keep the name `execute_*` although they record | R3 to R6 |
| `create_tensor` is `[[nodiscard]]`; four test expectations gain a `(void)` | R2, R8 |
| The factory accessor is `Engine::get_tensor_factory()` | R2 |
| `Engine::rng()` becomes `get_rng()`, by the accessor rule; it had no caller outside one seam header | R2 |
| `TaskRecorder` holds no `RuntimeContext`: with the pool passed in, nothing reads one | R1 |
| `Engine::execute()` stays a member of `Engine`; no separate graph runner | R2 |
| `Engine` declares its pool before its graph, so tasks return their blocks before the pool frees them | R2 |
| The test helper is `Test::GraphProbe` in `tests/common/graph_probe.hpp`, a class a test builds, not a fixture it inherits | R8 |
| The dependency contract is held by a text rule in the layering checker and by one regression test over seven operations | R7, R9 |

The support layer and memory (`DESIGN_SUPPORT_MEMORY.md`, section 18):

| Decision | Package |
| --- | --- |
| `min_along` is added | SM-7 |
| The storage alignment is 128 bytes, `SushiRuntime::Core::DEFAULT_ALIGNMENT`; the manual's "64-byte" sentences change | SM-1 |
| A load whose read fails after the length check leaves a partly written tensor and says so; a short file is still refused before any element changes | SM-9 |
| `logsumexp` of an infinite maximum returns that infinity where it returned NaN | SM-7 |
| The seams take `params` as `std::initializer_list`, no longer as `std::vector` | SM-6 |
| The ordered reductions reject an empty tensor; the identity reductions accept one without storage | SM-7 |
| `var` and `std` no longer equal the Welford loop in the last bits | SM-7 |
| The pool keeps six size classes up to 4 KiB and at most 64 idle blocks per class and device | SM-2 |
| File payloads of device storage are staged in 4 MiB chunks | SM-9 |
| No record-time check for HALF on a device without `aspect::fp16`; a known-issues row instead | SM-5 |
| SYRK refuses a device that cannot host its tile with `std::runtime_error` from inside the task | BLAS-5 |
| A rank-0 tensor keeps printing as `[]`, and `to_vector` keeps zero-filling; both recorded | SM-9 |
| `Tensor::reshape` keeps both signatures; `Tensor`'s own errors stay `std::runtime_error` | SM-1 |
| `sb docs` keeps its one line from the deleted SM-10, as SM-15 | SM-15 |
| The runtime's version bump lands with SR-1 or the version check lands with the tag; the orchestrator picks | SM-13 |

BLAS (`DESIGN_BLAS.md`, section 12):

| Decision | Package |
| --- | --- |
| The flag enumerators carry the CBLAS numbers, so a value-initialised flag is invalid | BLAS-1 |
| `dotu`, `dotc`, `geru`, `gerc` on a real tensor forward to `dot` and `ger` | BLAS-7, BLAS-8 |
| `ger`, `her`, `herk` and `rot` take their real scalars as `double` | BLAS-6 |
| `BlasScalar` is in namespace `SushiBLAS`; `std::complex<float>` does not convert to it implicitly | BLAS-1 |
| One source file per kernel pair, so `hemv` lives in `symv.cpp` | BLAS-8 |
| `engine/blas/sparse.hpp` stays where it is, no longer included | none |
| BLAS-6 is dispatched at `low`: its six headers are given in full | BLAS-6 |

Designed in this revision at the owner's request, and not yet read by the owner in detail
(`DESIGN_BLAS.md`, sections 4.8, 7.10 and 10.2). These are the choices inside the fix for the
SYMV finding:

| Decision | Package |
| --- | --- |
| A reference sum of HALF, FLOAT32 or COMPLEX32 terms is formed in double | BLAS-1, BLAS-3 to BLAS-5 |
| A reference sum of FLOAT64 or COMPLEX64 terms is a compensated double sum, not `long double`, which is 64 bits on one toolchain and 80 on the other | BLAS-1 |
| The brick is a new header, `support/reference_accumulator.hpp`; `support/accumulator.hpp` and every SYCL kernel stay as they are | BLAS-1 |
| The reference rank updates (`ger`, `syr`, `syr2`) stay in `Accumulator<T>`: they have no sum | BLAS-4 |
| The reference `nrm2` runs its recurrence in double for FLOAT32 and is not compensated for FLOAT64 | BLAS-3 |
| The reference TRSM widens each step's sum and keeps carrying the solution in `Accumulator<T>` | BLAS-5 |
| The parity test holds each kernel to the classical worst-case bound for its own accumulation type, and adds an exact-data case per shape for what that bound is too wide to catch | BLAS-10 |
| The truth helper is real-only | BLAS-10 |

### 5. What waits until after 0.1.0

From the recorder design:

- A "run after" on `TaskHandle`, which needs a public edge call in SushiRuntime.
- `add_task` overloads that take a span of regions and a small-buffer callable; each record still
  pays two region vectors and one `std::function`.
- A statement from SushiRuntime of what concurrent `add_task` guarantees.
- The recorder's identity counter in a source file, so two binaries cannot hand out one identity.
- Narrowing `IO` off `Engine&`, with the IO split of the backlog (C4).
- A fix for the reference path's host access to `Shared` storage beside a running kernel. The
  release records it as a known issue.

From the support design:

- A record-once plan: `Engine::execute` still compiles and discards the graph on every call.
- A parallel `shuffle`, which changes the permutation a seed gives; 0.1.0 says the permutation
  is not frozen.
- CPU GEMM packing and k-blocking; parallel TRMV, TRSV and TRSM.
- A record-time check for HALF on a device without `aspect::fp16`.
- A SYRK variant without local memory.
- The rank-0 print and the zero-fill of `to_vector`.
- `abs1` out of the reference Level 1 header, with the split of that header per routine.
- The layout moves: `src/` to `source/`, homes for `kernels/`, `examples/` and `benchmarks/`.
- Proof of the 128-byte alignment on Level Zero.

From the BLAS design:

- `hemm` and `her2k`, by the owner's decision, with `symm`, `syr2k` and `trmm`.
- The complex operators of the elementwise and nonlinear kernels moved onto `safe_mul` and
  `safe_div`.
- An implicit `BlasScalar` from `std::complex<float>`.
- The reference rank updates in `ReferenceAccumulator<T>`, and a complex truth for the parity
  tests.
- `sparse.hpp`: deleting it or moving it under `experimental/` needs the owner's word.

Not wanted at all, so not waiting: `sb --device`, `sb bench` and a `gpu` test label.

### 6. Open for the owner

1. **The `high` dispatches.** R4, R6, SM-7, BLAS-4 and BLAS-5 each need a yes before they go.
2. **BLAS-8.** The review rated it `high`, because sixteen methods are built from a table. This
   plan's rule keeps `high` for a kernel or a public header, and BLAS-8 is neither, so it is
   `low` here. Say so if it should be `high`.
3. **How the plan runs**: in the waves of section 2, or inline in the row order of section 1.
4. **Two merges this plan made to save a wave each.** SM-7 shares wave 3 with SM-6 and five
   other packages, so the riskiest numeric package of the support design gets its first build in
   the widest wave. BLAS-11 shares wave 6, so new tests and migrated tests fail or pass in one
   run. Either can be split off into a wave of its own.
5. **`CMakePresets.json`** describes a toolchain the project left. Delete it or rewrite it. No
   design recommends one, so the rule of section 4 does not settle it.
6. **Is a green CI run on the tagged commit a release condition?**
7. **Who tags SushiRuntime v0.4.0, and when.** SR-1 has to be in it, and so does the
   `Graph::add_emitter` addition, which is uncommitted in the runtime's working tree today.
8. **SM-12.** The owner's note says the version checker has landed with its changelog line. The
   checker has; its `rule_changelog`, which checks the release heading of the changelog against
   `project()`, has not, and SM-12 is that rule. If the note meant the rule is not wanted,
   SM-12 goes.
9. **SM-15.** It was not in the owner's answers. It is the one line of the deleted SM-10 that
    moves the generated API site under `build/`, which the documentation layout checker asks
    for. Say no and the `Doxyfile`'s output folder stays where it is.
10. **`IO` and `Shared` storage.** SM-9 reads `Shared` storage through its host pointer for a
    print, as a save already does. On Windows with CUDA that faults if a kernel of another graph
    is running on the same GPU. Routing `Shared` through the queue would close it for one
    staging copy per run. The plan keeps SM-9 as designed and records the exposure.
11. **The choices inside the reference accumulation and the parity bound** (the last table of
    section 4).
12. **SR-1 lands in a tree another session is working in.** The sushiruntime working tree holds
    uncommitted changes to `api/graph/graph.hpp`, `api/graph/region.hpp`, two READMEs, the
    changelog, `tests/CMakeLists.txt` and one new integration test. None of SR-1's three files
    is among them, but its changelog line touches a file that is.
13. **Names this revision picked**: `get_tensor_factory()`, `Test::GraphProbe`,
    `ReferenceAccumulator`, `ReferenceSum`, `rule_task_dependencies`.

### 7. Verification of this plan

Disjointness and order were checked mechanically. A script read the table of section 1 out of
this file, expanded every path and glob against the working tree, and reported any file claimed
by two packages of one wave, any "Needs" that is not in an earlier wave, and any path that is
neither on disk nor marked new by the package that names it or by an earlier one. Its output:

```
python -B check_plan.py
wave 0: SR-1 (3 files), SM-2 (3 files), SM-4 (4 files), SM-5 (3 files), SM-12 (2 files), SM-15 (2 files), BLAS-1 (8 files); 25 distinct files
wave 1: R1 (9 files), R7 (2 files); 11 distinct files
wave 2: R2 (5 files), R3 (59 files), R4 (36 files), R5 (11 files), R6 (25 files), R8 (140 files), BLAS-2 (4 files); 280 distinct files
wave 3: R9 (5 files), SM-1 (3 files), SM-3 (1 files), SM-6 (24 files), SM-7 (31 files), SM-8 (10 files), SM-9 (4 files); 78 distinct files
wave 4: BLAS-3 (2 files), BLAS-4 (12 files), BLAS-5 (6 files); 20 distinct files
wave 5: BLAS-6 (6 files); 6 distinct files
wave 6: BLAS-7 (9 files), BLAS-8 (7 files), BLAS-9 (3 files), BLAS-10 (23 files), BLAS-11 (8 files); 50 distinct files
32 packages, 470 file claims, 0 problems
```

The script was then given a fault to see that it reports one: a package in wave 2 that claims
`src/io.cpp`, names a file that does not exist and needs BLAS-6. It printed three problems, one
for each, and was deleted after the run.

What the script cannot see: a dependency this plan failed to write down. The "Needs" column was
derived by reading each package's includes and hunks, not by a tool. It also expands the globs
against today's tree, so a test file created in wave 0 is not yet among the files R8 sweeps in
wave 2; R8's sweep covers it by its glob when the wave runs, and no other package of wave 2
touches a test.

The C++ blocks the revision changed were parsed with the project's compiler;
`DESIGN_REVIEW.md`, section 7, has the command and its output.

## PLAN_EXT (Plan extension: the second design round, waves 7 to 11)

**Status:** Open — no package dispatched. Written on 2026-10-07 by the review of the four
second-round designs.

This file continues `PLAN.md`. It is the landing order of `DESIGN_BLAS_L3.md` (packages L3X),
`DESIGN_BLAS_BAND_PACKED.md` (BPX), `DESIGN_GEMM_CPU.md` (GMX) and `DESIGN_SPARSE.md` (SPX), on
the tree wave 6 of `PLAN.md` leaves. A package's text is in the design section its row names.
This file adds the order, the waves, one owner per shared file and the build-configuration lines
the orchestrator writes. Where a design's own wave table or CMake lines differ from what is
here, this file is the one to follow; each design says so in its head.

Nothing was built to write it. No `sb`, `se`, `cmake`, `ninja` or `ctest` process was started
and no source file was edited. The file's name was given by the dispatch; the documentation
layout checker reports it under `rule_work_folders`, as it reports every design file of this
folder.

### 0. Ground rules

The rules of `PLAN.md`, section 0, hold unchanged: every dispatch opens with the seven lines of
the delegation block, runs on `opus` at the effort its row gives, never builds and never edits a
build configuration file; the orchestrator makes the edits of section 3 and closes a wave.

Every package below is `low` by the rule of `PLAN.md`: its files are given in full in its
design and were parsed there, or are built from a table. Six of them hold numeric kernels or
branch-heavy scalar code that nobody has run. Section 5, point 6, asks whether any should be a
`high` dispatch, and names the six for a `medium` review either way.

### 1. Packages in landing order

Paths are relative to `D:\Projects\sushiblas`. "New:" marks files the package creates; "Edit:"
marks files that exist when it runs. "Needs" names the packages whose files have to be on disk
for the package's syntax check; every package also needs wave 6 of `PLAN.md`.

| Wave | Id | Title | Files | Needs | Effort | Read |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | L3X-1 | Symmetric access brick; GEMM through access bricks | New: `include/SushiBLAS/kernels/blas/symmetric_access.hpp`. Edit: `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`, `include/SushiBLAS/kernels/reference/blas/level3/gemm.hpp` | none | low | L3 4.1 to 4.3 |
| 7 | L3X-2 | SYR2K and HER2K kernels | New: `kernels/reference/blas/level3/syr2k.hpp`, `kernels/sycl/blas/level3/syr2k.hpp`. Edit: `kernels/blas.hpp` | none | low | L3 5.1 to 5.5 |
| 7 | L3X-3 | Level 3 public surface | Edit: `include/SushiBLAS/engine/blas/level3.hpp` | none | low | L3 3.5 |
| 7 | L3X-5 | Parallel TRMV | Edit: `kernels/sycl/blas/level2/row_reduction.hpp`, `kernels/sycl/blas/level2/trmv.hpp`, `src/engine/blas/level2/trmv.cpp` | none | low | L3 6 |
| 7 | L3X-6 | Blocked TRSV and TRSM | New: `kernels/sycl/blas/level3/triangular_system.hpp`, `kernels/sycl/blas/level3/triangular_plan.hpp`. Edit: `kernels/sycl/blas/level3/trsm.hpp` | none | low | L3 7 |
| 7 | GMX-1 | Blocking in the profile, and the plan | New: `include/SushiBLAS/kernels/blas/level3/packed_product_plan.hpp`, `tests/functional/unit/test_packed_product_plan.cpp`. Edit: `include/SushiBLAS/support/device_policy.hpp` | none | low | GMX 4, 8.1 |
| 7 | BPX-1 | Access and rotation bricks | New: `include/SushiBLAS/kernels/blas/band_access.hpp`, `include/SushiBLAS/kernels/blas/packed_access.hpp`, `include/SushiBLAS/kernels/blas/givens.hpp`, `include/SushiBLAS/kernels/blas/modified_givens.hpp` | none | low | BPX 5 |
| 7 | BPX-2 | Band, packed and scalar record-time bricks | New: `include/SushiBLAS/engine/blas/band_operand.hpp`, `include/SushiBLAS/engine/blas/packed_operand.hpp`, `include/SushiBLAS/engine/blas/scalar_operand.hpp`. Edit: `include/SushiBLAS/engine/blas/task_metadata.hpp` | none | low | BPX 6 |
| 7 | SPX-1 | Sparse value types and host assembly | New: `include/SushiBLAS/core/sparse_types.hpp`, `include/SushiBLAS/sparse_vector.hpp`, `include/SushiBLAS/sparse_matrix.hpp`, `include/SushiBLAS/support/sparse_assembly.hpp` | none | low | SPX 5 |
| 7 | SPX-2 | Sparse kernel bricks | New: `kernels/sparse/csr_pattern.hpp`, `kernels/sparse/index_list.hpp`, `kernels/reference/sparse/structure.hpp` | none | low | SPX 6 |
| 7 | SPX-12 | Layering checker | Edit: `tools/layering/check_layering.py`, `tools/tests/test_check_layering.py` | none | low | SPX 14 |
| 8 | L3X-4 | SYMM, HEMM, SYR2K and HER2K operations | New: `src/engine/blas/level3/symm.cpp`, `src/engine/blas/level3/syr2k.cpp` | L3X-1, L3X-2, L3X-3 | low | L3 4.5, 4.6, 5.6, 5.7 |
| 8 | L3X-7 | Tests of the four new Level 3 routines | New: `tests/functional/unit/test_symmetric_access.cpp`, `tests/functional/unit/test_symm.cpp`, `tests/functional/unit/test_syr2k.cpp` | L3X-1, L3X-2, L3X-3 | low | L3 9.1 to 9.3 |
| 8 | L3X-8 | Tests of the parallel triangular routines | New: `tests/functional/unit/test_triangular_plan.cpp`, `tests/functional/unit/test_triangular_parallel.cpp`. Edit: `tests/functional/unit/test_blas_half_accumulation.cpp` | L3X-5, L3X-6 | low | L3 9.4 to 9.6 |
| 8 | GMX-2 | The packed product bricks | New: `include/SushiBLAS/kernels/sycl/blas/level3/packed_panel.hpp`, `include/SushiBLAS/kernels/sycl/blas/level3/packed_micro_kernel.hpp`, `include/SushiBLAS/kernels/sycl/blas/level3/packed_product.hpp`, `tests/functional/unit/test_packed_panel.cpp`, `tests/functional/unit/test_packed_micro_kernel.cpp` | GMX-1 | low | GMX 3, 5, 8.2 |
| 8 | BPX-3 | Rotation kernels | New: `kernels/reference/blas/level1/rotg.hpp`, `kernels/reference/blas/level1/rotm.hpp`, `kernels/reference/blas/level1/rotmg.hpp`, `kernels/sycl/blas/level1/rotg.hpp`, `kernels/sycl/blas/level1/rotm.hpp`, `kernels/sycl/blas/level1/rotmg.hpp` | BPX-1 | low | BPX 8.1, 8.2 |
| 8 | BPX-4 | Band and packed products | New: `kernels/reference/blas/level2/gbmv.hpp`, `kernels/reference/blas/level2/symmetric_product.hpp`, `kernels/reference/blas/level2/sbmv.hpp`, `kernels/reference/blas/level2/spmv.hpp`, `kernels/sycl/blas/level2/gbmv.hpp`, `kernels/sycl/blas/level2/symmetric_product.hpp`, `kernels/sycl/blas/level2/sbmv.hpp`, `kernels/sycl/blas/level2/spmv.hpp` | BPX-1, L3X-5 | low | BPX 8.3 |
| 8 | BPX-5 | Band and packed triangular product and solve | New: `kernels/reference/blas/level2/triangular_product.hpp`, `kernels/reference/blas/level2/tbmv.hpp`, `kernels/reference/blas/level2/tpmv.hpp`, `kernels/reference/blas/level2/triangular_solve.hpp`, `kernels/reference/blas/level2/tbsv.hpp`, `kernels/reference/blas/level2/tpsv.hpp`, `kernels/sycl/blas/level2/triangular_product.hpp`, `kernels/sycl/blas/level2/tbmv.hpp`, `kernels/sycl/blas/level2/tpmv.hpp`, `kernels/sycl/blas/level2/triangular_solve.hpp`, `kernels/sycl/blas/level2/tbsv.hpp`, `kernels/sycl/blas/level2/tpsv.hpp` | BPX-1 | low | BPX 8.4 |
| 8 | BPX-6 | Packed rank updates | New: `kernels/reference/blas/level2/spr.hpp`, `kernels/reference/blas/level2/spr2.hpp`, `kernels/sycl/blas/level2/spr.hpp`, `kernels/sycl/blas/level2/spr2.hpp` | BPX-1 | low | BPX 8.5 |
| 8 | BPX-7 | Tests of the band, packed and rotation bricks | New: `tests/functional/unit/test_band_access.cpp`, `tests/functional/unit/test_packed_access.cpp`, `tests/functional/unit/test_structured_operands.cpp`, `tests/functional/unit/test_givens.cpp` | BPX-1, BPX-2 | low | BPX 10.2, 15.2 |
| 8 | SPX-3 | Sparse record-time bricks | New: `include/SushiBLAS/engine/sparse/index_check.hpp`, `include/SushiBLAS/engine/sparse/sparse_vector_operand.hpp`, `include/SushiBLAS/engine/sparse/sparse_matrix_operand.hpp`, `include/SushiBLAS/engine/sparse/accumulator_size.hpp` | SPX-1 | low | SPX 7 |
| 8 | SPX-4 | Sparse builder | New: `include/SushiBLAS/sparse_builder.hpp`, `src/sparse_builder.cpp` | SPX-1 | low | SPX 8 |
| 8 | SPX-5 | Sparse Level 1 kernels | New: `kernels/reference/sparse/level1.hpp`, `kernels/sycl/sparse/level1.hpp` | SPX-2 | low | SPX 9.1, 9.2 |
| 8 | SPX-6 | Sparse product and solve kernels | New: `kernels/sycl/sparse/row_blocks.hpp`, `kernels/reference/sparse/spmv.hpp`, `kernels/reference/sparse/spsymv.hpp`, `kernels/reference/sparse/spmm.hpp`, `kernels/reference/sparse/sptrsv.hpp`, `kernels/sycl/sparse/spmv.hpp`, `kernels/sycl/sparse/spsymv.hpp`, `kernels/sycl/sparse/spmm.hpp`, `kernels/sycl/sparse/sptrsv.hpp` | SPX-2 | low | SPX 9.3 to 9.6 |
| 9 | GMX-3 | GEMM takes the packed path | New: `include/SushiBLAS/kernels/sycl/blas/level3/gemm_packed.hpp`. Edit: `include/SushiBLAS/engine/blas/utils.hpp`, `include/SushiBLAS/engine/blas/dispatch.hpp` | GMX-2 | low | GMX 6 |
| 9 | BPX-8 | Level 1 and Level 2 public surface | Edit: `include/SushiBLAS/engine/blas/level1.hpp`, `include/SushiBLAS/engine/blas/level2.hpp`, `kernels/blas.hpp` | BPX-3, BPX-4, BPX-5, BPX-6 | low | BPX 7 |
| 9 | SPX-7 | Sparse view and include files | New: `include/SushiBLAS/engine/sparse.hpp`, `kernels/sparse.hpp`. Edit: `include/SushiBLAS/SushiBLAS.h` | SPX-1, SPX-2, SPX-4, SPX-5, SPX-6 | low | SPX 4.3, 9.7, 10 |
| 9 | SPX-9 | Tests of the sparse bricks | New: `tests/common/sparse_samples.hpp`, `tests/functional/unit/test_sparse_types.cpp`, `tests/functional/unit/test_sparse_assembly.cpp`, `tests/functional/unit/test_sparse_structure.cpp`, `tests/functional/unit/test_sparse_builder.cpp` | SPX-1, SPX-2, SPX-3, SPX-4 | low | SPX 12.1, 12.3 |
| 10 | GMX-4 | Tests of the packed GEMM | New: `tests/common/packed_product_harness.hpp`, `tests/functional/unit/test_packed_product.cpp` | GMX-3 | low | GMX 8.3, 8.4 |
| 10 | GMX-6 | The blocking benchmark | New: `benchmarks/bench_gemm_blocking.cpp`. Edit: `benchmarks/bench_common.hpp` | GMX-3 | low | GMX 9 |
| 10 | BPX-9 | Rotation operations | New: `src/engine/blas/level1/rotg.cpp`, `src/engine/blas/level1/rotm.cpp`, `src/engine/blas/level1/rotmg.cpp` | BPX-2, BPX-8 | low | BPX 9.1, 9.2 |
| 10 | BPX-10 | Band operations | New: `src/engine/blas/level2/gbmv.cpp`, `src/engine/blas/level2/sbmv.cpp`, `src/engine/blas/level2/tbmv.cpp`, `src/engine/blas/level2/tbsv.cpp` | BPX-2, BPX-8 | low | BPX 9.2, 9.3 |
| 10 | BPX-11 | Packed operations | New: `src/engine/blas/level2/spmv.cpp`, `src/engine/blas/level2/tpmv.cpp`, `src/engine/blas/level2/tpsv.cpp`, `src/engine/blas/level2/spr.cpp`, `src/engine/blas/level2/spr2.cpp` | BPX-2, BPX-8 | low | BPX 9.2, 9.3 |
| 10 | BPX-12 | Tests of the band, packed and rotation routines | New: `tests/common/blas_structured.hpp`, `tests/functional/unit/test_rotg.cpp`, `tests/functional/unit/test_rotm.cpp`, `tests/functional/unit/test_rotmg.cpp`, `tests/functional/unit/test_blas_band.cpp`, `tests/functional/unit/test_blas_packed.cpp`, `tests/functional/unit/test_blas_structured_validation.cpp` | BPX-8 | low | BPX 10.1, 10.3 |
| 10 | BPX-13 | Band and packed benchmark | New: `benchmarks/bench_band_packed.cpp` | BPX-8 | low | BPX 11 |
| 10 | SPX-8 | Sparse operation sources | New: `src/engine/sparse/axpyi.cpp`, `src/engine/sparse/doti.cpp`, `src/engine/sparse/gthr.cpp`, `src/engine/sparse/gthrz.cpp`, `src/engine/sparse/sctr.cpp`, `src/engine/sparse/roti.cpp`, `src/engine/sparse/spmv.cpp`, `src/engine/sparse/spsymv.cpp`, `src/engine/sparse/spmm.cpp`, `src/engine/sparse/sptrsv.cpp` | SPX-3, SPX-5, SPX-6, SPX-7 | low | SPX 11 |
| 10 | SPX-10 | Tests of the sparse routines | New: `tests/functional/unit/test_sparse_level1.cpp`, `tests/functional/unit/test_sparse_spmv.cpp`, `tests/functional/unit/test_sparse_spsymv.cpp`, `tests/functional/unit/test_sparse_spmm.cpp`, `tests/functional/unit/test_sparse_sptrsv.cpp`, `tests/functional/unit/test_sparse_validation.cpp`, `tests/functional/regression/test_sparse_recording.cpp` | SPX-5, SPX-6, SPX-7, SPX-9 | low | SPX 12.2, 12.4 |
| 10 | SPX-11 | Sparse benchmark | New: `benchmarks/bench_sparse.cpp` | SPX-4, SPX-7 | low | SPX 13 |
| 11 | GMX-5 | SYRK and HERK on the packed product | New: `tests/functional/unit/test_packed_syrk.cpp`. Edit: `kernels/sycl/blas/level3/syrk.hpp`, `src/engine/blas/level3/syrk.cpp` | GMX-3, GMX-4 | low | GMX 7.1, 8.5 |
| 11 | GMX-7 | SYMM and HEMM on the packed product | New: `include/SushiBLAS/kernels/sycl/blas/level3/gemm_accessed_packed.hpp`, `tests/functional/unit/test_packed_symm.cpp`. Edit: `src/engine/blas/level3/symm.cpp` | GMX-3, GMX-4, L3X-1, L3X-4 | low | GMX 7.3, 8.6 |
| 11 | L3X-9 | Benchmark of the triangular and symmetric routines | New: `benchmarks/bench_triangular.cpp` | L3X-3, L3X-6, GMX-6 | low | L3 10 |

Forty-one packages. A test package needs the headers it includes on disk, and the operation
sources it calls only at link time, which the build of its wave gives; that is the rule
`DESIGN_BLAS.md` set for BLAS-11. So L3X-7 shares wave 8 with L3X-4, BPX-12 shares wave 10 with
BPX-9 to BPX-11, and SPX-10 shares it with SPX-8.

BPX-4 names L3X-5 under "Needs" although it opens none of its files: `gbmv` and
`symmetric_product` include `kernels/sycl/blas/level2/row_reduction.hpp`, and their syntax check
should run against the text L3X-5 leaves.

### 2. Waves

Inside a wave the file sets are pairwise disjoint and every "Needs" is in an earlier wave;
section 6 has the script that checked both.

| Wave | Packages | Closes with | Tree compiles at the end |
| --- | --- | --- | --- |
| 7 | L3X-1, L3X-2, L3X-3, L3X-5, L3X-6, GMX-1, BPX-1, BPX-2, SPX-1, SPX-2, SPX-12 | no build; `python -m unittest discover -s tools/tests` for SPX-12 | yes, but one existing test would fail, see below |
| 8 | L3X-4, L3X-7, L3X-8, GMX-2, BPX-3, BPX-4, BPX-5, BPX-6, BPX-7, SPX-3, SPX-4, SPX-5, SPX-6 | the lines of section 3, `sb build`, `sb test`, the five checks of `AGENTS.md`; then the same on the CUDA build | yes |
| 9 | GMX-3, BPX-8, SPX-7, SPX-9 | no build | yes: sixteen `Level1` and `Level2` members are declared and not yet defined, and nothing calls them |
| 10 | GMX-4, GMX-6, BPX-9, BPX-10, BPX-11, BPX-12, BPX-13, SPX-8, SPX-10, SPX-11 | the lines of section 3, `sb build --benchmarks`, `sb test`, the five checks; then the same on the CUDA build | yes |
| 11 | GMX-5, GMX-7, L3X-9 | the lines of section 3, `sb build --benchmarks`, `sb test`, the five checks; then the same on the CUDA build | yes |

Five waves, three of which end with a build.

**Waves 7 and 8 are one build unit.** L3X-6 changes the order in which a backward solve
subtracts, and `HalfTrsmCarriesTheSolutionInFloat` compares the SYCL result with a host loop bit
for bit. The host loop moves with L3X-8, which needs L3X-6 on disk. A test run between the two
waves would show that one failure and nothing else.

**What the wave-8 build compiles for the first time.** All of the Level 3 design except its
benchmark; the plan and the three bricks of the packed product, through their tests; the four
band, packed and rotation bricks, through theirs. The band, packed and sparse kernels of wave 8
are on disk and not yet included by anything: `kernels/blas.hpp` gains them in wave 9 and the
sparse umbrella is written there, so their first compile is the wave-10 build.

**Wave 9 has no build** because every package in it only declares or includes. An orchestrator
who wants the earlier compile of the sparse public headers can run `sb build` after it.

**The CUDA build closes waves 8, 10 and 11.** It is the first ahead-of-time compile of every
kernel of this round, and the first run of the work-group shape of TRMV, of the blocked TRSM
with a GPU's block width, and of the work-group shape of `gbmv` and the band and packed
symmetric products.

What "done" means for each package is in its design: L3 sections 3.2 and 9, GMX 11.2, BPX 4.3,
SPX 15.2.

#### 2.1 One owner per shared file

The files below are the ones more than one design of this round touches, reads or was
expected to touch. Each has one owner per wave.

| File | Owners, in order | Why the order is safe |
| --- | --- | --- |
| `kernels/blas.hpp` | L3X-2 (wave 7), then BPX-8 (wave 9) | L3X-2 adds two lines after the SYRK lines of the Level 3 group; BPX-8 adds lines after the `level1.hpp` line and after the `trmv.hpp` line |
| `src/engine/blas/level3/symm.cpp` | L3X-4 creates it (wave 8), then GMX-7 (wave 11) | GMX-7 adds one include, one lease and two arguments to the text L3X-4 writes |
| `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp` | L3X-1 (wave 7) only | GMX adds `gemm_packed.hpp` and `gemm_accessed_packed.hpp`, which include it and do not open it |
| `include/SushiBLAS/engine/blas/level3.hpp` | L3X-3 (wave 7) only | no other design declares a Level 3 method |
| `include/SushiBLAS/engine/blas/level1.hpp`, `level2.hpp` | BPX-8 (wave 9) only | L3X-5 changes `trmv.cpp` and leaves the declaration of `trmv` as it is |
| `include/SushiBLAS/support/device_policy.hpp` | GMX-1 (wave 7) only | L3X-6 and SPX-6 read `Device::Profile` and add nothing to it; GMX-1 adds two fields at its end and renames none |
| `include/SushiBLAS/engine/blas/utils.hpp`, `dispatch.hpp` | GMX-3 (wave 9) only | the sources of waves 8 and 10 include `utils.hpp` and call functions GMX-3 does not change |
| `kernels/sycl/blas/level2/row_reduction.hpp` | L3X-5 (wave 7) only | BPX-4 (wave 8) calls `reduce_rows_by_group`, whose signature L3X-5 keeps |
| `benchmarks/bench_common.hpp` | GMX-6 (wave 10) only | L3X-9 (wave 11) calls `Bench::time_iterations`, which GMX-6 adds; BPX-13 and SPX-11 call `Bench::measure`, whose signature GMX-6 keeps |

`include/SushiBLAS/SushiBLAS.h`, `tools/layering/check_layering.py`,
`include/SushiBLAS/engine/blas/task_metadata.hpp`, `kernels/sycl/blas/level3/syrk.hpp`,
`kernels/sycl/blas/level3/trsm.hpp` and `tests/functional/unit/test_blas_half_accumulation.cpp`
each have one package of this round and no second.

#### 2.2 Leaving a design out

The four designs can be taken separately. What moves when one is not:

| Left out | What changes in the rows that stay |
| --- | --- |
| `DESIGN_GEMM_CPU.md` | GMX-1 to GMX-7 go. L3X-9 moves to wave 10, carries the third hunk of GMX 9.2 (`time_iterations` and the `measure` built on it) and adds `benchmarks/bench_common.hpp` to its files. Wave 11 is empty |
| `DESIGN_BLAS_L3.md` | L3X-1 to L3X-9 and GMX-7 go. Waves 7 and 8 can each close with a build. BPX-4 loses its need of L3X-5. The triangular sibling tests of BPX-12 keep their exact data; nothing in them depends on this design |
| `DESIGN_BLAS_BAND_PACKED.md` | BPX-1 to BPX-13 go. Nothing else moves |
| `DESIGN_SPARSE.md` | SPX-1 to SPX-12 go. Nothing else moves. `SPARSE_REQUIREMENTS.md` puts this design in v0.1.0, so this row is here for completeness |

The inline form of the plan is the row order of section 1.

### 3. Build-configuration edits, per wave

The orchestrator makes these. No implementer does. Lines are verbatim, with their four leading
spaces; none was run. Three designs anchored test lines on the same line of the same group;
here every insertion has one anchor, and a later wave anchors on the last line an earlier one
added.

#### Wave 7

None. `test_packed_product_plan.cpp` is on disk after this wave and is listed in wave 8, where
the first build is.

#### Wave 8

`src/CMakeLists.txt`, in `SUSHIBLAS_SOURCES`, after `    engine/blas/level3/trsm.cpp`:

```cmake
    engine/blas/level3/symm.cpp
    engine/blas/level3/syr2k.cpp
```

`tests/CMakeLists.txt`, in the group `# --- Level 3 ---`, after
`    functional/unit/test_blas_complex.cpp`, the last line wave 6 added there:

```cmake
    functional/unit/test_symmetric_access.cpp
    functional/unit/test_symm.cpp
    functional/unit/test_syr2k.cpp
    functional/unit/test_triangular_plan.cpp
    functional/unit/test_triangular_parallel.cpp
    functional/unit/test_packed_product_plan.cpp
    functional/unit/test_packed_panel.cpp
    functional/unit/test_packed_micro_kernel.cpp
    functional/unit/test_band_access.cpp
    functional/unit/test_packed_access.cpp
    functional/unit/test_structured_operands.cpp
    functional/unit/test_givens.cpp
```

#### Wave 9

None.

#### Wave 10

`src/CMakeLists.txt`, in `SUSHIBLAS_SOURCES`, after `    io.cpp`:

```cmake

    # Sparse: host-side construction
    sparse_builder.cpp
```

After `    engine/blas/level1/rot.cpp`:

```cmake
    engine/blas/level1/rotg.cpp
    engine/blas/level1/rotm.cpp
    engine/blas/level1/rotmg.cpp
```

After `    engine/blas/level2/syr2.cpp`:

```cmake
    engine/blas/level2/gbmv.cpp
    engine/blas/level2/sbmv.cpp
    engine/blas/level2/tbmv.cpp
    engine/blas/level2/tbsv.cpp
    engine/blas/level2/spmv.cpp
    engine/blas/level2/tpmv.cpp
    engine/blas/level2/tpsv.cpp
    engine/blas/level2/spr.cpp
    engine/blas/level2/spr2.cpp
```

After `    engine/blas/level3/syr2k.cpp`, which wave 8 added:

```cmake

    # Sparse BLAS
    engine/sparse/axpyi.cpp
    engine/sparse/doti.cpp
    engine/sparse/gthr.cpp
    engine/sparse/gthrz.cpp
    engine/sparse/roti.cpp
    engine/sparse/sctr.cpp
    engine/sparse/spmv.cpp
    engine/sparse/spsymv.cpp
    engine/sparse/spmm.cpp
    engine/sparse/sptrsv.cpp
```

`tests/CMakeLists.txt`, in the group `# --- Level 1 ---`, after
`    functional/unit/test_rot.cpp`:

```cmake
    functional/unit/test_rotg.cpp
    functional/unit/test_rotm.cpp
    functional/unit/test_rotmg.cpp
```

In the group `# --- Level 2 ---`, after `    functional/unit/test_trsv.cpp`:

```cmake
    functional/unit/test_blas_band.cpp
    functional/unit/test_blas_packed.cpp
    functional/unit/test_blas_structured_validation.cpp
```

In the group `# --- Level 3 ---`, after `    functional/unit/test_givens.cpp`, the last line
wave 8 added there:

```cmake
    functional/unit/test_packed_product.cpp
```

A new group directly above the line
`    # --- Spatial (windowed movement over an N-D grid) ---`:

```cmake
    # --- Sparse ---
    functional/unit/test_sparse_types.cpp
    functional/unit/test_sparse_assembly.cpp
    functional/unit/test_sparse_structure.cpp
    functional/unit/test_sparse_builder.cpp
    functional/unit/test_sparse_level1.cpp
    functional/unit/test_sparse_spmv.cpp
    functional/unit/test_sparse_spsymv.cpp
    functional/unit/test_sparse_spmm.cpp
    functional/unit/test_sparse_sptrsv.cpp
    functional/unit/test_sparse_validation.cpp

```

In the group `# --- Regression ---`, after
`    functional/regression/test_blas_symmetric_bits.cpp`, the last line wave 6 added there:

```cmake
    functional/regression/test_sparse_recording.cpp
```

`benchmarks/CMakeLists.txt`, the list of programs becomes:

```cmake
set(SB_BENCHMARKS bench_gemm bench_gemm_blocking bench_level1 bench_reductions bench_elementwise bench_band_packed bench_sparse)
```

#### Wave 11

`tests/CMakeLists.txt`, in the group `# --- Level 3 ---`, after
`    functional/unit/test_packed_product.cpp`, which wave 10 added:

```cmake
    functional/unit/test_packed_syrk.cpp
    functional/unit/test_packed_symm.cpp
```

`benchmarks/CMakeLists.txt`, the list of programs becomes:

```cmake
set(SB_BENCHMARKS bench_gemm bench_gemm_blocking bench_level1 bench_reductions bench_elementwise bench_band_packed bench_sparse bench_triangular)
```

and after `endforeach()`:

```cmake

# bench_triangular times the TRSM kernel at block widths the public API does not select.
target_include_directories(bench_triangular PRIVATE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
```

No label and no new `gtest_discover_tests` block in any wave: every new suite carries the
`Unit_` or the `Regression_` prefix the three existing blocks pick up. The install rule needs no
line; the new public headers are under `include/`.

### 4. What the review found and changed

The review read the four designs against the first round and against each other. Its findings
by the six items it was asked to check, and what it edited.

**Conflicts between the designs, and how each was settled.**

1. The band and packed sibling tests compared `tbmv` and `tpmv` with `trmv`, and `tbsv` and
   `tpsv` with `trsv`, with `EXPECT_EQ` on random data in both modes. `DESIGN_BLAS_L3.md` makes
   the dense two parallel: its TRMV sums a row in lanes where the device plans a work-group, and
   its TRSM subtracts a backward solve from the top down. Under `FORCE_SYCL` those comparisons
   would have failed on rounding, on the CUDA build for the product and on every device for a
   backward solve. Settled in `DESIGN_BLAS_BAND_PACKED.md`: the triangular sibling cases are now
   two cases of their own on integer data whose every sum is exact (its sections 3.5, 8.6, 10.3,
   11, 14.3 and 14.4); `DESIGN_BLAS_L3.md`, section 7.2, states the same fact from its side.
2. Both `DESIGN_BLAS_L3.md` and `DESIGN_GEMM_CPU.md` wanted a timing loop over an arbitrary
   body in the benchmarks: one as a helper inside its own program, one in `bench_common.hpp`.
   Settled: `bench_common.hpp` has one owner, GMX-6, and L3X-9 uses its `Bench::time_iterations`
   and lands a wave later (L3 section 10, GMX section 9.2, and the fallback of section 2.2
   here).
3. `kernels/blas.hpp` is edited by L3X-2 and by BPX-8. Settled by order: waves 7 and 9.
4. `symm.cpp` is created by L3X-4 and edited by GMX-7. `DESIGN_GEMM_CPU.md` said the two designs
   share no source file; its head now names the one they share. Settled by order: waves 8
   and 11.
5. Three designs anchored test lines after `test_blas_complex.cpp`, and three gave the whole
   `SB_BENCHMARKS` line as if alone. Settled in section 3 here.
6. `spmv` is the packed symmetric product in `DESIGN_BLAS_BAND_PACKED.md` and the CSR product in
   `DESIGN_SPARSE.md`, with kernels of that name in the same two namespaces. They overload
   without ambiguity; the argument counts were checked (ten and twelve against nine and
   thirteen). Not settled: section 5, point 3. One thing was changed: the sparse SYCL kernel
   now logs `SYCL SPARSE SPMV`, so a log line says which kernel ran.
7. "Packed" is BLAS packed storage in one design and panel packing in another. No identifier
   collides. Not settled: section 5, point 4.

**SOLID shape.** Every new unit has one job, and every new routine follows the five steps and
the kernel-pair shape of its dense sibling. No first-round brick is redefined. No kernel and no
support header of the four designs includes anything under `engine/` or names the recorder.
Three overlaps were found between the new bricks; none is a defect of one design, and each is
recorded as open work in the design that would host the merge (L3 section 13, points 19 to 21;
BPX 14.3; SPX 18): the mirrored read of a symmetric matrix exists three times (`symv`'s lambda,
`SymmetricAccess`, `symmetric_product`'s lambda); "the stored columns of row r" exists as
`TriangleRow` and as the accessors' `first_column` and `end_column`; "how much a device runs at
once" is derived from the profile in three places in three units. Two sibling differences are
deliberate and stated where they occur: `rotg` and `rotmg` share one brick between the two back
ends (BPX 3.3), and the sparse kernels hoist `beta == 0` out of the device lambda where the
dense ones compare inside it (SPX 9.1, rule 3).

**Register.** The four files were scanned for the vocabulary and structure the `humanizer`
English file lists, and read. No finding in the prose as written. Four sentences stated an
unmeasured cost as a fact and were reworded under item (f) below.

**C++ and first-round names.** Every name the four designs take from the first round was
compared with the first-round text or with the file on disk where the wave has landed:
`TaskRecorder`, `TaskHandle`, `OperandSet`, `Internal::execute_typed_task`,
`Internal::execute_blas_task`, `KernelPairDispatcher::run`, `ScratchBuffer`, `ScratchPool`,
`Internal::allocate_blas_scratch`, `allocate_level1_scratch`, `running_solution_bytes`,
`BlasScalar`, the four flag types and their helpers, both conjugation policies,
`ConjugationFor`, `RankUpdateScalar`, `OperandAccess`, `operand_access`, `ReferenceSum`,
`ReferenceAccumulator`, `Accumulator`, `RunningSolution`, `multiply`, `divide`, `conjugate_if`,
the `describe_` and `require_` functions, `BlasMetadata`, `Launch::indexed`,
`Reduce::planned_partials`, `Reduce::group_width`, `Detail::row_reduction_width`,
`reduce_rows_by_group`, `Device::Profile` and its fields, `Device::work_group_size`,
`K_WAVES_PER_COMPUTE_UNIT`, `hosts_local_tile`, `LocalTileDemand`, `register_block_dim`,
`K_LOCAL_ACC_BYTES`, `group_count`, `Test::GraphProbe`, `Engine::set_mode`,
`Engine::get_scratch_pool`, `TaskMetadata::set_param`. No mismatch was found: no name is
misspelled, no signature is called with another argument list, and no construct needs more than
C++17. The review compiled nothing; each design's own record says which blocks its author
parsed and which exist only as tables.

**Numerical soundness.** Worked by hand, with pencil arithmetic and no program:

| What | Example worked | Result |
| --- | --- | --- |
| Blocked triangular solve | order 5, block width 2, forward and backward, through `TriangularBlockSolve` and `TriangularBlockUpdate`: which launch first touches each unknown, and the order in which each solved unknown is subtracted | every unknown is scaled by alpha once, at its first touch; forward subtracts t = 0 up, backward t = 4 down, for any width. The decode of the right side (`op_a.at(A, t, s)`, `at_b.index(v, s)`, forward when op(A) is upper) was derived from `X * op(A) = alpha * B` and matches |
| The solve plan | the break-even rule on a 16-unit CPU and an 80-unit GPU; the optimum `2 * sqrt(t_launch / t_step)` re-derived | 546 unblocked and 547 blocked; 1 024 with 8 streams unblocked; widths 346 and 120 before rounding. As the design states |
| GEMM blocking | the plan for FLOAT64 2048 cubed, FLOAT32 2048 cubed and COMPLEX64 1024 cubed from the three budgets; a 7 x 17 x 3 FLOAT32 product under the widest blocking, and m = 130 against rows of 126, for the short last register block and macro-block; k = 300 against a depth cap of 256 | the three table rows reproduce. A short last micro-panel is zero-filled, the tile index `(h * row_groups + g) * K_TILE_REALS` is the same in the micro-kernel and in the store, each element of C is stored once, the panel and tile sizes of a slot cover what is written, and every vector load is aligned. The sum order is ascending l for every blocking |
| HERK on the packed product | the conjugation flag of the second factor for `NO_TRANS` and `CONJ_TRANS` | `A * A^H` and `A^H * A` |
| Hermitian band and packed products | the 3 x 3 Hermitian matrix of `HbmvConjugatesTheMirror` and a 2 x 2 packed one against `x = (1, i)` | `(2i, 10 - 2i, 8 + 4i)` and `(1 + i, 1 + 2i)`; the mirror is conjugated, the diagonal's imaginary part is dropped |
| Hermitian packed updates | `HprKeepsTheDiagonalReal` and `Hpr2ConjugatesAlpha` | both tables reproduce for both triangles |
| Band index mapping | ten of the fourteen elements of the worked 4 x 5 matrix in column-major and eight in row-major, the first and last of every line among them; rows 0 and 3 and columns 2 and 4 of `gbmv` on it through `first_column` and `end_column`; the four triangle band arrays of the tridiagonal test; CBLAS's row-major convention derived from its Fortran call | every offset lands on the element the literal arrays hold; the products are `(74, 230, 474, 530)` and `(53, 152, 330, 326, 285)` |
| Packed index mapping | the four (layout, uplo) formulas derived from the storage order, three elements read off the 3 x 3 arrays, `is_upper` for the four flag pairs, `transposed()` | as the tables state |
| Rotations | `givens_rotation` against `drotg.f90` and `zrotg.f90` as the reviewer recalls them; `modified_givens` for `(5, 5, 4, 2)`, `(1, 1, 1, 1)` and the rescaling case; the `rotm` table | the values reproduce. The Fortran was not at hand, see section 5, point 9 |
| SYMM, HEMM, SYR2K, HER2K | the hand-computed rows of `test_symm.cpp` and `test_syr2k.cpp`, the tile sizes of `Syr2kTileConfig` | all reproduce |
| Transposed sparse product | a 3 x 2 matrix with four entries in two row blocks: the block boundaries from `block_first_row`, the two partial slices, the fold; then the sample matrix against `(1, 2, 3, 4)` | block b writes slice b only, the fold writes `y[j]` only and runs behind the scatter, and each output receives its terms in ascending row order. No element has two writers in either stage |
| One-triangle sparse product, sparse solve | the 2 x 2 Hermitian case; a 2 x 2 transposed solve; the Level 1 table | `(4 + i, 1 + 5i)`, `(1.5, 2)`, and the Level 1 values reproduce |

No numerical defect was found in a kernel. The one numerical finding is conflict 1 above, which
is a test expectation.

**File sets.** Section 2.1.

**Performance claims.** Each design keeps its expectations in one section with the command that
tests them. Four sentences elsewhere read as facts and were reworded as expectations with the
row that would test each: "which is where TRSV gains" (L3 7.1); "a rank-k update costs a little
over half a GEMM" (GMX 7.1); "it will be the slowest step of that loop" and "`spmv` on a matrix
stored in full is the faster call" (SPX 1.1, 9.4); and "its serial form costs about what a
serial product costs" (SPX 9.3).

### 5. Open for the owner

1. **Which of the four land before the 0.1.0 tag.** `SPARSE_REQUIREMENTS.md` puts the sparse
   family in. `PLAN.md`, section 5, lists `symm`, `hemm`, `syr2k`, `her2k`, the parallel
   triangular routines and CPU GEMM packing as waiting until after 0.1.0, and
   `DESIGN_GEMM_CPU.md` recommends that for itself. The band, packed and rotation routines were
   not in `PLAN.md` at all. Section 2.2 says what moves when a design is left out.
2. **How the plan runs**: in the five waves of section 2, or one design at a time in the row
   order of section 1. The waves mix four designs, so the wave-8 build is the first compile of
   three of them at once; the suites are separate, but a compile error in a shared header stops
   all three.
3. **`spmv` names two routines**: the packed symmetric product on `BLASOps` and the CSR product
   on `SparseOps`. Recommended by both designs: keep both. The alternative renames the four
   sparse matrix routines (SPX 17, decision 11; BPX 14.1, point 11).
4. **"Packed" names two things**: BLAS packed storage and the packed panels of the CPU GEMM
   (GMX 13, point 16). Recommended: keep both terms and have the READMEs say which is meant.
5. **Scratch above 4 KiB.** Three designs lease more than the pool's largest class and so
   allocate and free device memory on every record: the TRMV snapshot (L3 13, point 16), the
   panels of a packed product, megabytes each (GMX 13, decision 3), and the partials of a
   transposed or one-triangle sparse product (SPX 13 and 18). Each says "measure first". The
   remedy all three point at is the same: size classes above 4 KiB in `ScratchShelf`, which is a
   change to a first-round brick.
6. **Effort and review of the numeric packages.** All are `low` because their text is given.
   Six hold code nobody has run: L3X-2 (SYR2K), L3X-5 (parallel TRMV), L3X-6 (blocked TRSM),
   GMX-2 (the packed product), SPX-6 (eight sparse kernels) and BPX-1 (`givens.hpp`, ninety
   lines of branches from Fortran). Say so if any should be a `high` dispatch. Each should get a
   `medium` review with its design section open.
7. **The order of a backward substitution** (L3 13, decision 3). Top-down makes the SYCL result
   independent of the block width and moves `AUTO` and `FORCE_SYCL` results of backward solves
   in the last bits against wave 6. It also decides whether a backward `tbsv` and a backward
   `trsv` agree to the bit on inexact data; with the recommendation they do not.
8. **Four new benchmark programs**: `bench_triangular`, `bench_gemm_blocking`,
   `bench_band_packed`, `bench_sparse`. Each design asks for its own; together they double the
   folder.
9. **The rotation bricks were written from Fortran fetched through a summarising tool**
   (BPX 14.4, point 2), and this review compared them with the reviewer's memory of
   `drotg.f90`, `zrotg.f90` and `drotmg.f`, not with the files. The arithmetic of the design's
   form is right for the cases worked. One point could not be settled: whether the reference
   `drotmg.f` writes the implied entries of H only when the flag is 0 or 1, as the design does,
   or on every pass of its rescaling loop. The design's form is the one that keeps H correct
   over two passes. A reviewer with a LAPACK checkout settles it before wave 8 closes.
10. **Each design's own decisions**, unchanged by this review:
    - `DESIGN_BLAS_L3.md`, section 13, points 1 to 9: the shape of the SYR2K kernel; the
      off-diagonal update of the blocked solve; no serial arm for a small TRMV; the two
      block-width constants; `gemm_accessed` in the public kernel headers; the signatures.
    - `DESIGN_GEMM_CPU.md`, section 13, points 1 to 9: on by default; the bit-equality test of
      the two kernels as a hard assertion; SYRK with GEMM or after it; the names.
    - `DESIGN_BLAS_BAND_PACKED.md`, section 14.1, points 1 to 10: the shape of a row-major band
      tensor; where the band widths sit in a signature; a `rot` that takes c and s as tensors;
      one brick for both back ends in `rotg` and `rotmg`; `rotmg` refuses HALF; a strided b in
      `tbsv` and `tpsv`.
    - `DESIGN_SPARSE.md`, section 17, points 1 to 10: the matrix and vector types; the module
      boundaries; the scatter scheme; symmetry as a routine; deleting
      `engine/blas/sparse.hpp`; the builder and its exception type; the names.
11. **`Core::IndexBase` carries 221 and 222**, written from memory in the design. The reviewer
    recalls the same two numbers from `blas_enum.h` and also had no copy. SPX-1's implementer
    checks them.

### 6. Verification of this plan

Disjointness and order were checked mechanically. A script read the table of section 1 out of
this file and reported any file claimed by two packages of one wave, any "Needs" that is not in
an earlier wave, any "New:" path that is already on disk, and any "Edit:" path that is neither
on disk nor created by an earlier package. Its output:

```
python -B check_plan_ext.py
wave 7: L3X-1 (3 files), L3X-2 (3 files), L3X-3 (1 files), L3X-5 (3 files), L3X-6 (3 files), GMX-1 (3 files), BPX-1 (4 files), BPX-2 (4 files), SPX-1 (4 files), SPX-2 (3 files), SPX-12 (2 files); 33 distinct files
wave 8: L3X-4 (2 files), L3X-7 (3 files), L3X-8 (3 files), GMX-2 (5 files), BPX-3 (6 files), BPX-4 (8 files), BPX-5 (12 files), BPX-6 (4 files), BPX-7 (4 files), SPX-3 (4 files), SPX-4 (2 files), SPX-5 (2 files), SPX-6 (9 files); 64 distinct files
wave 9: GMX-3 (3 files), BPX-8 (3 files), SPX-7 (3 files), SPX-9 (5 files); 14 distinct files
wave 10: GMX-4 (2 files), GMX-6 (2 files), BPX-9 (3 files), BPX-10 (4 files), BPX-11 (5 files), BPX-12 (7 files), BPX-13 (1 files), SPX-8 (10 files), SPX-10 (7 files), SPX-11 (1 files); 42 distinct files
wave 11: GMX-5 (3 files), GMX-7 (3 files), L3X-9 (1 files); 7 distinct files
41 packages, 160 file claims, 0 problems
```

The script was then given a fault to see that it reports one: a package in wave 11 that
claims `symm.cpp` beside GMX-7, creates `kernels/blas.hpp`, edits a file that does not exist,
and needs L3X-9 and a package that is not in the table. It printed five problems, one for each,
and was deleted after the run.

What the script cannot see: a dependency this plan failed to write down. The "Needs" column was
derived by reading each package's includes, not by a tool. It also reads "on disk" from today's
tree, in which waves 0 to 3 of `PLAN.md` have landed or are landing and waves 4 to 6 have not;
every "Edit:" path of this plan exists today, and none is created or deleted by waves 4 to 6.

What the review did not do: compile anything. The hand examples of section 4 check arithmetic
and indexing on paper. Whether a kernel computes them is for the builds that close waves 8, 10
and 11.

## DESIGN_YOLO26_PACKAGES (Section 2 of DESIGN_YOLO26: packages, order and ownership)

#### 2.1 The table

Paths are relative to `D:\Projects\sushiblas`. "New" marks a file the package creates. "After"
names what has to be on disk before the package is written, so its syntax check can run, and the
`PLAN.md` package that owns a shared file before it.

| Wave | Id | Title | Files | After | Effort | Read |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | YL-1 | Spatial bricks and the fp16 check | `include/SushiBLAS/support/spatial_geometry.hpp`, `include/SushiBLAS/engine/spatial/window.hpp`, `include/SushiBLAS/support/device_policy.hpp`, `include/SushiBLAS/engine/operand_check.hpp`, `tests/functional/unit/test_operand_check.cpp` | SM-8 (`window.hpp`), SM-5 (`operand_check.hpp` and its test), SM-4 (`device_policy.hpp`) | low: every addition given verbatim | 3.2, 4.3 |
| 7 | YL-2 | HALF sums of the window kernels | `kernels/sycl/spatial/col2im.hpp`, `kernels/sycl/spatial/pool_max.hpp`, `kernels/sycl/spatial/pool_avg.hpp`, `kernels/sycl/spatial/upsample.hpp`, `kernels/reference/spatial/col2im.hpp`, `kernels/reference/spatial/pool_max.hpp`, `kernels/reference/spatial/pool_avg.hpp`, `kernels/reference/spatial/upsample.hpp`. New: `tests/functional/unit/test_spatial_half.cpp` | R8 built (wave 2); no `PLAN.md` package owns a spatial kernel | low: seven textual rules with their hit counts | 4.1, 4.4 |
| 7 | YL-3 | Host loops and the recorder's public guard | New: `include/SushiBLAS/support/serial_1d.hpp`, `include/SushiBLAS/support/serial_reduce.hpp`, `tests/functional/unit/test_serial_loops.cpp`. `include/SushiBLAS/engine/task_recorder.hpp` | R1 (`task_recorder.hpp`), BLAS-1 (`reference_accumulator.hpp`), wave 3 built | low: both headers given in full and parsed; one hunk | 5, 7.2 |
| 7 | YL-4 | Declared reads | New: `include/SushiBLAS/engine/math/elementwise/read_binding.hpp`, `tests/functional/unit/test_read_binding.cpp` | SM-5 (`scalar_binding.hpp`), R1 (`operand_set.hpp`) | low: the header is given in full and parsed | 6 |
| 8 | YL-5 | Window sum and input gradient, the kernels | New: `kernels/reference/spatial/window_dot.hpp`, `kernels/sycl/spatial/window_dot.hpp` | YL-1 | low: both files given in full and parsed | 3.4 |
| 8 | YL-6 | Segmented fold and the weight gradient, the kernels | `include/SushiBLAS/support/reduce_axis.hpp`. New: `kernels/reference/spatial/window_dot_weights.hpp`, `kernels/sycl/spatial/window_dot_weights.hpp`, `tests/functional/unit/test_reduce_segments.cpp` | YL-1; SM-7 built (its tests call `Reduce::reduce_axis`) | high: a numeric kernel and the launch brick four existing reductions run on | 3.5, 3.6, 4.2 |
| 9 | YL-7 | Window sum, the three operations | `include/SushiBLAS/engine/spatial.hpp`, `kernels/spatial.hpp`, `tests/functional/regression/test_spatial_determinism.cpp`. New: `src/engine/spatial/window_dot.cpp`, `src/engine/spatial/window_dot_weights_backward.cpp`, `tests/functional/unit/test_spatial_window_dot.cpp` | YL-5, YL-6; R5 (`spatial.hpp`), R8 (the determinism test), BLAS-10 (`tests/common/blas_truth.hpp`) | low: declarations and both sources given in full and parsed | 3.1, 3.3, 3.7 |
| 9 | YL-8 | Functor seams: declared reads and the host path | `include/SushiBLAS/engine/task_recorder.hpp`, `include/SushiBLAS/engine/execution_mode.hpp`, `include/SushiBLAS/engine/math/elementwise/dispatch.hpp`, `include/SushiBLAS/engine/math/nonlinear/dispatch.hpp`, `include/SushiBLAS/engine/logic/dispatch.hpp`, `src/engine/math/elementwise/clamp.cpp`, `src/engine/math/elementwise/convert.cpp`, `src/engine/logic/all.cpp`, `src/engine/logic/any.cpp`, `tests/functional/unit/test_dispatch_seam.cpp`, `tests/functional/unit/test_math_validation.cpp`, `tests/functional/unit/test_task_recorder.cpp`. New: `include/SushiBLAS/engine/math/elementwise/map_task.hpp`, `tests/functional/unit/test_functor_reference.cpp` | YL-3, YL-4; SM-6 (the three seam headers, the four sources, two of the tests), R9 (`test_task_recorder.cpp`) | high: three public seam headers | 7 |
| 9 | YL-9 | Reduction seams: the host path | `include/SushiBLAS/engine/math/reductions.hpp`, `include/SushiBLAS/engine/math/reductions/dispatch.hpp`, `include/SushiBLAS/engine/math/reductions/centred_reduction.hpp`, `include/SushiBLAS/engine/math/reductions/arg_reduction.hpp`, `src/engine/math/reductions/argmax.cpp`, `argmin.cpp`, `count_nonfinite.cpp`, `logsumexp.cpp`, `max.cpp`, `max_along.cpp`, `mean.cpp`, `mean_spatial.cpp`, `min.cpp`, `min_along.cpp`, `std.cpp`, `var.cpp` in the same folder, `tests/functional/unit/test_reductions.cpp` | YL-3; SM-7 (every file) | high: a public seam header | 8 |
| 9 | YL-10 | Random seam: the host path | `include/SushiBLAS/engine/math/random/dispatch.hpp`, `src/engine/math/random/bernoulli.cpp`, `constant.cpp`, `discrete_uniform.cpp`, `exponential.cpp`, `he_normal.cpp`, `he_uniform.cpp`, `log_normal.cpp`, `normal.cpp`, `orthogonal.cpp`, `poisson.cpp`, `shuffle.cpp`, `truncated_normal.cpp`, `uniform.cpp`, `xavier_normal.cpp`, `xavier_uniform.cpp` in the same folder, `kernels/sycl/random/orthonormalize.hpp`, `kernels/random.hpp`. New: `include/SushiBLAS/support/orthonormal_sweep.hpp`, `kernels/reference/random/orthonormalize.hpp`, `tests/functional/unit/test_random_reference.cpp` | YL-3; SM-6 (the seam header and four of the sources), SM-4 (`orthonormalize.hpp`) | high: a public seam header | 9 |
| after | YL-11 | SushiAI: declared reads and the mode, by the sushiai session, inside R11 | `D:\Projects\sushiai` | wave 9 built | not this session's | 11 |
| after | YL-12 | Records, by the orchestrator, inside the closing commit | the files of section 12.5 | every package above | the orchestrator's | 12.5 |

Effort follows the owner's rule and `PLAN.md`: `high` is kept for a numeric kernel or a public
header an implementer has to derive. Four packages are `high` and each needs the owner's word
first: YL-6, YL-8, YL-9 and YL-10.

#### 2.2 Waves

The waves follow wave 6 of `PLAN.md`. Inside a wave the file sets are pairwise disjoint.

| Wave | Packages | Closes with | Tree compiles at the end |
| --- | --- | --- | --- |
| 7 | YL-1 (low), YL-2 (low), YL-3 (low), YL-4 (low) | `sb build`, `sb test`, the five checks of `AGENTS.md` | yes |
| 8 | YL-5 (low), YL-6 (high) | `sb build`, `sb test`, the five checks | yes |
| 9 | YL-7 (low), YL-8 (high), YL-9 (high), YL-10 (high) | `sb build`, `sb test`, the five checks; then the same on the CUDA build | yes |

Wave 7 only adds, except that HALF results of five spatial operations change (YL-2) and HALF on
a device without `fp16` is refused (YL-1). Wave 8 adds two kernel pairs nobody calls yet and
replaces the body of `Reduce::reduce_axis` under four reductions; its build is the one that
proves the FLOAT32 and FLOAT64 bits of `sum_along`, `max_along`, `min_along` and `mean_spatial`
did not move. Wave 9 is one build unit, as waves 1 and 2 of `PLAN.md` are: YL-8 deletes
`TaskRecorder::record` and `TaskRecorder::require_sycl_path`, and the tree compiles again only
when YL-9 and YL-10 have stopped calling them.

`engine/task_recorder.hpp` is written twice, a wave apart. YL-3 makes `require_host_reachable`
public and changes nothing else, so the seam packages of wave 9 can each be parsed alone against
the recorder on disk. YL-8 then deletes the two SYCL-only members. `record` is never kept alive
for a caller that has moved: every caller moves in wave 9, in the wave that deletes it.

Earliest starts, for an owner who wants fewer waves. By file ownership YL-2 could start after
wave 2, YL-3 and YL-4 after wave 3, YL-1 after wave 3. None of the four needs anything waves 4 to
6 change, so wave 7 can be written beside the BLAS waves and built with wave 6. This file keeps
them apart because wave 6 is already the widest BLAS build. Waves 7 and 8 can be merged too if
YL-5 and YL-6 check their kernels against a scratch copy of the `TapSpan` struct of 3.2.

What "done" means for each package. An implementer reports the syntax check; the orchestrator
judges the row at the build that closes the wave.

| Package | Done when |
| --- | --- |
| YL-1 | The two new cases of `test_operand_check.cpp` pass and every existing test still does |
| YL-2 | `Unit_SpatialHalf` passes; the seven rules, run again, report no hit; every existing spatial test passes unchanged |
| YL-3 | `Unit_SerialLoops` passes; `grep -n "require_host_reachable" include/SushiBLAS/engine/task_recorder.hpp` shows the function in the public part |
| YL-4 | `Unit_ReadBinding` passes |
| YL-5 | Each of the two headers parses alone |
| YL-6 | `Unit_ReduceSegments` passes; `Unit_AxisReductions*`, `Unit_MeanSpatial*` and `Unit_ReductionDeterminism*` pass unchanged |
| YL-7 | `Unit_SpatialWindowDot` and the two new determinism rows pass, on the CPU build and on the CUDA build |
| YL-8 | `Unit_FunctorReference` and the rewritten cases of 10.2 pass; `grep -rn "require_sycl_path\|recorder_\?\.record(" include src tests` prints nothing |
| YL-9 | The rewritten and the new cases of `test_reductions.cpp` pass |
| YL-10 | `Unit_RandomReference` passes; the two textual rules of 9.3, run again, report no hit |

#### 2.3 Files another package changes first

| File | Before | Then |
| --- | --- | --- |
| The three functor seam headers | R3 (wave 2), SM-6 (wave 3) | YL-8 |
| `engine/math/reductions/dispatch.hpp`, `reductions.hpp`, the reduction sources | R4, SM-7 | YL-9 |
| `reductions/centred_reduction.hpp`, `reductions/arg_reduction.hpp` | SM-7 creates them | YL-9 |
| `engine/math/random/dispatch.hpp`, the random sources | R4, SM-6 | YL-10 |
| `engine/task_recorder.hpp` | R1; `DESIGN_RECORDER.md` says "nobody" afterwards | YL-3 (wave 7), YL-8 (wave 9) |
| `engine/execution_mode.hpp` | R1 | YL-8, one comment |
| `engine/spatial.hpp` | R5 | YL-7 |
| `engine/spatial/window.hpp` | SM-8 | YL-1 |
| `engine/operand_check.hpp`, `test_operand_check.cpp` | SM-5 | YL-1 |
| `support/device_policy.hpp` | SM-4 | YL-1 |
| `kernels/sycl/random/orthonormalize.hpp` | SM-4 | YL-10 |
| `support/reduce_axis.hpp`, `support/spatial_geometry.hpp`, the spatial kernels, `kernels/spatial.hpp`, `kernels/random.hpp` | no `PLAN.md` package | YL-6, YL-1, YL-2, YL-7, YL-10 |
| `test_dispatch_seam.cpp`, `test_math_validation.cpp` | R8, SM-6 | YL-8 |
| `test_task_recorder.cpp` | R9 creates it | YL-8 |
| `test_reductions.cpp` | R8, SM-7 | YL-9 |
| `test_spatial_determinism.cpp` | R8 | YL-7 |
| `tests/CMakeLists.txt`, `src/CMakeLists.txt` | the orchestrator only | the orchestrator only |

R9 has written tests that assert the refusal Y4 removes, and the designs of SM-6 and SM-7 ask
for more; section 10.2 lists them. Where one of those cases is not written yet when the owner
reads this, it can be left out, and YL-8 and YL-9 write the replacement once. That is the
orchestrator's call.

#### 2.4 Build-configuration lines, for the orchestrator

No implementer edits a build configuration file. Lines are verbatim, with their four leading
spaces; none was run.

Wave 7, `tests/CMakeLists.txt`. In the group `# --- Spatial (windowed movement over an N-D grid) ---`,
after `    functional/unit/test_mean_spatial.cpp`:

```cmake
    functional/unit/test_spatial_half.cpp
```

In the group `# --- Support layer and scratch ---` that wave 0 added, after
`    functional/unit/test_scratch_buffer.cpp`:

```cmake
    functional/unit/test_serial_loops.cpp
```

In the group `# --- Dispatch seam (the public SushiBLAS::Dispatch generic executors) ---`, after
`    functional/unit/test_dispatch_seam.cpp`:

```cmake
    functional/unit/test_read_binding.cpp
```

Wave 8, `tests/CMakeLists.txt`. In the group `# --- Reductions ---`, after
`    functional/unit/test_reduce_plans.cpp`:

```cmake
    functional/unit/test_reduce_segments.cpp
```

Wave 9, `src/CMakeLists.txt`, in `SUSHIBLAS_SOURCES`, after `    engine/spatial/broadcast.cpp`:

```cmake
    engine/spatial/window_dot.cpp
    engine/spatial/window_dot_weights_backward.cpp
```

Wave 9, `tests/CMakeLists.txt`. In the spatial group, after the line wave 7 added:

```cmake
    functional/unit/test_spatial_window_dot.cpp
```

In the group `# --- Math: Random ---`, after `    functional/unit/test_uniform.cpp`:

```cmake
    functional/unit/test_random_reference.cpp
```

In the dispatch-seam group, after the line wave 7 added:

```cmake
    functional/unit/test_functor_reference.cpp
```

The three `gtest_discover_tests` blocks pick the new suites up by their `Unit_` prefix. The two
new determinism rows live in a file that is already listed.

#### 2.5 Syntax check for each package

Item 6 of the dispatch block, for an implementer: take the flags of any `src/` entry of
`build/compile_commands.json`, drop `-o`, `-c`, `-O3`, `-fsycl-targets=...` and `--cuda-path=...`,
add `-fsyntax-only`. A package that only adds headers checks each through a one-line translation
unit in the scratchpad. YL-7 checks its two sources as they are. YL-8, YL-9 and YL-10 check each
source they change; in wave 9 a source of one of the three parses whether or not the other two
have landed, because none of them calls a member another one removes.



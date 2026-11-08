# Test review: plan

**Status:** Open — T1 to T8 done; the M jobs next

Applies the verdicts of `REPORT.md`, in hierarchy order: the shared helpers, adapters and seams the
rewrites need come first, then the defects the review found, then each module's tests. A module job
applies every record of its module: rewrite, resize, split, merge and delete as the record says, and
writes each add. Nothing in a module job changes production code except a seam a job above built.

## Jobs

| Job | Files | Waits on | Status |
| --- | --- | --- | --- |
| T1 | `tests/support/tolerance.hpp` in use: `expect_tensor` and every tolerance constant moved onto it |  | done |
| T2 | A host-callable functor or host-buffer seam for the host logic now labelled medium | SushiRuntime T1 | done |
| T3 | The operations, facade and log-category copies deleted, the smaller copy kept |  | done |
| T4 | Benchmarks moved to the benchmark runner |  | done |
| T5 | The defects of this report, each with a test that fails first, and the pages corrected |  | done |
| T6 | The CLI tests on sushicore's in-memory runner and console | sushicore T1 | done |
| T7 | The device-math float and double rows taken from SushiEngine `Small_KernelMath` |  | done |
| T8 | `tools/testing/check_test_additions.py`, `tools/tests/test_check_test_additions.py` deleted |  | done |
| M1 | the tests of `applications` (6 files): 3 rewrite, 5 split, 4 merge, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M2 | the tests of `cli` (8 files): 56 rewrite, 18 resize, 1 split, 13 merge, 6 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M3 | the tests of `data` (12 files): 12 rewrite, 17 resize, 22 split, 3 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M4 | the tests of `engine` (56 files): 67 rewrite, 70 resize, 31 split, 43 merge, 35 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M5 | the tests of `foundation/arithmetic` (6 files): 19 rewrite, 1 resize, 3 split, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M6 | the tests of `foundation/blas_test_oracle` (4 files): 6 rewrite, 1 resize, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M7 | the tests of `foundation/core` (9 files): 18 rewrite, 3 resize, 1 split, 2 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M8 | the tests of `foundation/cost` (4 files): 5 rewrite, 7 split, 6 merge, 4 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M9 | the tests of `foundation/geometry` (6 files): 15 rewrite, 1 resize, 3 split, 2 merge, 5 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M10 | the tests of `foundation/graph_primitives` (14 files): 82 rewrite, 10 resize, 7 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M11 | the tests of `foundation/host_device` (3 files): 1 rewrite, 3 resize, 1 split, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M12 | the tests of `foundation/launch` (6 files): 14 rewrite, 14 resize, 1 split, 1 merge, 8 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M13 | the tests of `foundation/orthonormal` (1 file): 3 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M14 | the tests of `foundation/profile` (3 files): 2 rewrite, 3 resize, 3 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M15 | the tests of `foundation/reduction` (2 files): 4 rewrite, 8 resize, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M16 | the tests of `foundation/reference` (3 files): 4 rewrite, 10 resize, 1 split, 4 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M17 | the tests of `foundation/serial_reductions` (1 file): 3 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M18 | the tests of `foundation/small_integers` (1 file): 2 rewrite, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M19 | the tests of `foundation/sparse_assembly` (1 file): 2 rewrite, 14 resize | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M20 | the tests of `foundation/sparse_fixtures` (2 files): 7 rewrite, 2 split, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M21 | the tests of `foundation/substitution` (1 file): 4 rewrite, 3 merge, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M22 | the tests of `foundation/test_runner` (4 files): 6 rewrite, 1 resize, 1 split, 1 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M23 | the tests of `foundation/typed_suite_naming` (1 file): 1 resize, 1 merge | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M24 | the tests of `kernels/blas_kernels` (56 files): 55 rewrite, 2 resize, 12 split, 1 merge, 10 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M25 | the tests of `kernels/blas_seam_kernels` (25 files): 62 rewrite, 14 resize, 4 split, 5 merge, 5 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M26 | the tests of `kernels/blend_kernels` (3 files): 4 rewrite, 1 merge | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M27 | the tests of `kernels/linalg_kernels` (7 files): 14 rewrite, 3 split, 1 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M28 | the tests of `kernels/packed_product_harness` (1 file): 1 rewrite, 1 resize, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M29 | the tests of `kernels/random_kernels` (6 files): 12 rewrite, 4 resize, 2 split, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M30 | the tests of `kernels/signal_kernels` (6 files): 10 rewrite, 2 merge, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M31 | the tests of `kernels/sparse_kernels` (16 files): 13 rewrite, 1 resize, 4 split, 6 merge, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M32 | the tests of `kernels/spatial_kernels` (19 files): 6 rewrite, 6 split, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M33 | the tests of `library` (13 files): 63 rewrite, 26 resize, 8 split, 4 merge, 16 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M34 | the tests of `operations/benchmark_harness` (3 files): 4 rewrite, 8 resize, 2 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M35 | the tests of `operations/blas` (53 files): 327 rewrite, 41 resize, 4 split, 16 merge, 126 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M36 | the tests of `operations/blend` (1 file): 11 rewrite, 1 resize, 2 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M37 | the tests of `operations/elementwise` (38 files): 38 rewrite, 52 resize, 6 split, 7 merge, 7 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M38 | the tests of `operations/experimental` (3 files): 6 rewrite, 11 resize | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M39 | the tests of `operations/index` (1 file): 3 rewrite, 2 resize, 3 split, 11 merge, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M40 | the tests of `operations/io` (5 files): 15 rewrite, 8 resize, 7 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M41 | the tests of `operations/logic` (15 files): 13 rewrite, 2 split, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M42 | the tests of `operations/nonlinear` (10 files): 22 rewrite, 1 resize | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M43 | the tests of `operations/random` (11 files): 32 rewrite, 1 resize, 8 split, 11 merge, 4 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M44 | the tests of `operations/reductions` (13 files): 66 rewrite, 3 resize, 12 split, 22 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M45 | the tests of `operations/signal` (1 file): 5 rewrite, 1 resize, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M46 | the tests of `operations/sparse` (12 files): 50 rewrite, 6 resize, 9 split, 8 merge, 8 delete | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M47 | the tests of `operations/spatial` (12 files): 44 rewrite, 17 resize, 12 split, 1 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M48 | the tests of `package` (1 file): 1 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | done |
| M49 | the tests of `tools` (7 files): 11 rewrite, 37 resize, 11 split, 14 merge, 5 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| H1 | Seam: `data/tensor` host storage and a host `TensorFactory` (no runtime), and the shape-only tensor for validation; `TensorFactoryContract` and `StorageContract` suites, small on host, medium on the runtime | M46 | done |
| H2 | Seam: `engine/engine` `KernelPairDispatcher` holds the queue (`pair.queue()`, `pair.run(deps, reference, sycl)`), the body drops its queue parameter; `TaskSink` port in `TaskRecorder` with the graph adapter and a host adapter that runs bodies in record order under `ForceReference`; `TaskSinkContract` | H1 | done |
| H3 | Seam: `ScratchPool` block source port with a host adapter (RW-153), `Engine` host constructor, a host `RNGStream` path, `engine_fixture` `HostEngineTest` and a host `RecordingFixture`; `EngineContract` | H2 | done |
| H4 | Bodies onto H2 in `engine/*` and `operations/elementwise`, `logic`, `nonlinear`, `random`, `reductions` | H2 | done |
| H5 | Bodies onto H2 in `operations/blas`, `sparse`, `spatial`, `signal`, `blend`, `index`, `io`, `experimental` and `library` | H4 | done |
| H6 | Index and tail math out of the SYCL launches of `kernels/*` into host-callable functions, rows `Small_`, one `Medium_` launch per family | H5 | done (H6a to H6e: spatial, blend, sparse, signal, blas_seam, blas, linalg, random orthonormal rules, packed product harness strides) |
| H7 | RW-155, RW-156: `Engine` and `ScratchPool` hold no nullable runtime; `is_host()` and the throwing `runtime()` go; `RecordingGraph` and its 26 `RecordingGraph(engine)` callers take what they need, not an `Engine` | H6 | done |
| S1 | RW-150, RW-151: the LogicOps element functors and the RandomOps refusals as host-callable functions, their tests `Small_`, one `Medium_` launch per family | M41, M43, H7 | done |
| S2 | RW-148: the npy and sushi header parsers over bytes, their refusals `Small_` | M40, H7 | done |
| S3 | RW-147: the graph-primitive decisions as host functions, their tests `Small_` | M10, H7 | done |
| S4 | RW-149: the index-ops recording and refusal tests `Small_` on a host seam | M39, H7 | done |
| S5 | RW-152: the reduction refusals, dtype, empty-input and partial-plan rules as host functions, their tests `Small_` | M44, H7 | todo |
| S6 | RW-153: a `ScratchPool` block source port with an in-memory adapter, the lease tests `Small_` | M44 | done |

M10, M39, M40, M43 and M44 were `todo` again: each left resize records at `Medium_` for a missing seam, which the owner's order of 2026-11-07 does not accept. The S job that builds the seam applies those records, and the M job turns `done` with it: S1 closes M43, S2 closes M40, S3 closes M10, S4 closes M39, S5 and S6 close M44.

Owner order of 2026-11-07, from SushiSkills `docs/agent/2026_11_03_INDUSTRIAL_TESTING/SPEC.md` ("Rework" steps 3 to 5): the seams H1 to H6 come before the remaining triage. After them, the S jobs and the reopened M jobs move their logic tests onto the seams, and only then the remaining M jobs run. The counts by size per module are recorded in REPORT.md "Suite shape" before H1 and after the last job; the aim is 80% small, and a module below it is a refactoring finding.

Resume note, 2026-11-07, after commit `9eb150e7`: H6e, H7, S1 and S2 are done (RW-148, RW-150, RW-151, RW-155 and RW-156 closed). Left in order: S3 (RW-147: the graph-primitive argument decisions of `counting_sort`, `prefix_scan`, `reduce` and `segmented_reduce` as host functions; the Medium_ rows that record only to refuse a mis-sized argument, a window past its buffer or empty options repeat the Small_ rows of `element_window`, `primitive_options` and `invalid_graph`), S4 (RW-149), S5 (RW-152), the reopened M10, M39, M40, M43 and M44 on the host engine, then the remaining M jobs, and the one build and test. M40 keeps its other records (the host-transfer and device rows), M43 its 32 rewrites, 8 splits and 11 merges apart from the refusals S1 moved. For M25: the 38 `Medium_PackedMicroKernel` and `Medium_PackedPanel` tests are host-only and look `Small_`. For M24: `Medium_RowReduction.TheWindowsCoverAWholeRowOrOneTriangle` now repeats `Small_MatrixVectorIndex` rows. In blas_seam_kernels `sycl/level3/gemm.hpp`, the `Detail::` aliases of `GroupShape`, `group_shape` and `group_count` have no outside user left. The suite shape is in REPORT.md "After H6e, H7, S1 and S2".

Resume note, 2026-11-07, after the IoLog, decision 37 and `sushiblas_add_module` commits: the IoLog rows are `Small_` on a `FileRemover` port (decision 14 amended), decision 37 is confirmed (one shared check for both modes), and `modules/library/sushiblas` is declared through `sushiblas_add_module(... ARCHIVE)`. M10 read, not changed: the test bodies of `test_counting_sort.cpp`, `test_prefix_scan.cpp`, `test_segmented_reduce.cpp` and `test_reduce/test_reduce.cpp` hold no loop or branch (the loops left are in named helpers and row builders), so the loop-to-rows and builder rewrites are already applied. What keeps M10 `todo`: the `Medium_` rows that only record a graph and read its nodes (names, classes, widths, window dependencies, `ZeroCountStampsWidthOne`, `LevelWidthsMatchTheLevelPlan`) need the four primitives to record through a `TaskSink` (a host sink that keeps the node facts) instead of a `RecordingGraph` over a runtime; that is a production seam in `graph_primitives`, not built yet. The above-capacity refusals stay `Medium_` by decision 18. M43, M1 to M4, M24, M25, M33 to M35, M37, M47, M49 are untouched.

Resume note, 2026-11-07, after `0b203597`: the `PrimitiveRecorder` port carries its operand type (decision 39, `HostBuffer`) and the REPORT.md Suite shape holds the count after it. The jobs left are the ones the note after `feee1d3d` lists, unchanged, and none of them is started.

Resume note, 2026-11-07, after `feee1d3d`: M10 is done (decision 39, `PrimitiveRecorder` port with `RuntimePrimitiveRecorder` and `HostPrimitiveRecorder`) and `test_random_log_category.cpp` is deleted with its registration (`compile_random_log_category.cpp` is a compile check in its place and the mirror finding is gone from `tools/known_findings.toml`). Left, in order: M2 to M4, M24, M25, M33 to M35, M37, M47, M49, S5 and the one build and test run. The suite shape is in REPORT.md "After M10 and the random log category".

Resume note, 2026-11-07, after the M2 start: M2 is partly applied and stays `todo`. Done: `PackageSource` port with `StorePackageSource` and `MemoryPackageSource` (`services/package_source.py`), `library_packages.package_type`, the pure `caller_flags` and `refuse_lean_benchmarks`, `command_harness` on a real `Config` (`make_config`), `test_console_lazy.py`, `test_aliases.py`, `test_config.py` and `test_cli.py` per their records, and the new `test_cli_settings.py`. Left in M2: `test_project.py` and `test_library_packages.py` still carry the `_Driver`, `_Config` and parts stand-ins and the service-level flag tests; the small rows (`test_build_flags.py`: defines, benchmark, sanitizer, distributed and lean flags, `caller_flags` order, the lean refusal, `TreeChoice.directory` with the override row, `package_type` and `package_options` rows) are not written; `test_help_screen.py` records (three merges, the rename of the examples test, the raw-colour scan moved to a checker) are not applied; `test_provision_commands.py` (two help rows small, the `ProvisionSettings` field check) is not applied. Not doable in this repository: the malformed-config row of `test_main_reports_a_malformed_config_file_as_one_line` stays `Medium_`, since sushicore's `load_appearance` reads TOML from disk with no port. M3, M4, M24, M25, M33 to M35, M37, M47 and M49 are untouched.

Resume note, 2026-11-09, after M2: M2 is done (decision 40). Every `cli` record is applied but one: the malformed-config row of `test_main_reports_a_malformed_config_file_as_one_line` stays medium until SushiCore gives `load_appearance` a port (RW-157). Left, in order: M3, M4, M24, M25, M33 to M35, M37, M47, M49, S5 and the one build and test run.

Resume note, 2026-11-09, after M3: M3 is done (decision 41, commit `83d12f17`). Left, in order: M4, M24, M25, M33 (partly started, see the umbrella note), M34, M35, M37, M47, M49, S5 and the one build and test run. M4 is untouched. Seams ready for it: `HostEngineTest`, `HostTaskSink`, `make_host_storage`, `TensorFactory::Host{}` and `Tensor::make_shape_only`; the pattern M3 used is a typed or row suite for each value axis, host storage for everything the logic reads, and a few `Medium_` tests on the runtime factory for the device boundary. Syntax check: build the command of a sibling entry of `build/default-debug/compile_commands.json` with the file swapped and `-fsyntax-only`; one TU takes 2 to 5 s.

## Acceptance

- **T1.** No comparison spells a tolerance literal, and an exact comparison is spelled as equality. Build: the repository's test command over the touched tests.
- **T2.** The host decisions run small with no runtime. Build: the repository's test command over each touched module.
- **T3.** No promise is tested twice. Build: the repository's test command.
- **T4.** No benchmark is built into a test program (RW-135 ruling, REPORT.md decision 8). Build: the repository's test command and its benchmark command.
- **T5.** Every defect above is fixed or has a backlog entry. Build: the repository's test command and `python tools/check_all.py .`
- **T6.** No CLI test patches the repository's own functions. Build: the CLI tests.
- **T7.** The rows run here. Build: the repository's test command over device math.
- **T8.** Neither file exists and the checkers pass. Build: `python tools/check_all.py .`
- **M jobs.** Every record of the module is applied: no test the report deletes remains, every kept and
  new test passes at the size its label states, and the module's tests pass through the project's own
  test command. Build: that command over the module.
- **H jobs.** Each seam has its contract suite, `Small_` on the host adapter and `Medium_` on the real one, and the tests it was built for run `Small_` on it. Build: the one build at the end.
- **S jobs.** Owner order of 2026-11-07: a test a record resizes to small runs `Small_` on a seam the job builds; `Medium_` stays only for a kernel against its host reference, an adapter's bytes or a thread handover.

The owner chooses inline or parallel mode.

Resume note, 2026-11-09, M33 started: M33 stays `todo`. Done: `test_dependency_stress.cpp` (the 5000-reader test deleted, loops in named helpers), `test_dispatch_complex.cpp` (one test per operation, the refusals as `TEST_P` rows), `test_dispatch_seam.cpp` (one-task assertions, rows for the operand refusals and the six read seams, helpers for AdamW, per-dtype scalar tests, the device-USM skip in a fixture), `test_empty_tensors.cpp` (now `Small_` on `HostEngineTest`, registered under `TESTS`) and the new `Small_` `test_dispatch_checks.cpp` (the real-only domain check and the parameter capacity, moved off the runtime). The tolerance records of these files were already applied by T1. Left in M33: `test_integer_dtypes.cpp`, `test_math_validation.cpp`, `test_operand_lifetime.cpp`, `test_pack_ordered_sum.cpp`, `test_regression.cpp`, `test_spatial_random_io_validation.cpp`, `test_storage_offset.cpp`, `test_task_dependencies.cpp`, `test_framework_sim.cpp`; then M34, M35, M37, M47, M49 are untouched. Syntax checked with clang -fsyntax-only through compile_commands.json; nothing built.

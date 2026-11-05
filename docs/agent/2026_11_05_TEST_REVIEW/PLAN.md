# Test review: plan

**Status:** Open — not started; the owner chooses inline or parallel

Applies the verdicts of `REPORT.md`, in hierarchy order: the shared helpers, adapters and seams the
rewrites need come first, then the defects the review found, then each module's tests. A module job
applies every record of its module: rewrite, resize, split, merge and delete as the record says, and
writes each add. Nothing in a module job changes production code except a seam a job above built.

## Jobs

| Job | Files | Waits on | Status |
| --- | --- | --- | --- |
| T1 | `tests/support/tolerance.hpp` in use: `expect_tensor` and every tolerance constant moved onto it |  | todo |
| T2 | A host-callable functor or host-buffer seam for the host logic now labelled medium | SushiRuntime T1 | todo |
| T3 | The operations, facade and log-category copies deleted, the smaller copy kept |  | todo |
| T4 | Benchmarks moved to the benchmark runner |  | todo |
| T5 | The defects of this report, each with a test that fails first, and the pages corrected | owner decisions 1, 2 | todo |
| T6 | The CLI tests on sushicore's in-memory runner and console | sushicore T1 | todo |
| T7 | The device-math float and double rows taken from SushiEngine `Small_KernelMath` |  | todo |
| T8 | `tools/testing/check_test_additions.py`, `tools/tests/test_check_test_additions.py` deleted |  | todo |
| M1 | the tests of `applications` (6 files): 3 rewrite, 5 split, 4 merge, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M2 | the tests of `cli` (8 files): 56 rewrite, 18 resize, 1 split, 13 merge, 6 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M3 | the tests of `data` (12 files): 12 rewrite, 17 resize, 22 split, 3 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M4 | the tests of `engine` (56 files): 67 rewrite, 70 resize, 31 split, 43 merge, 35 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M5 | the tests of `foundation/arithmetic` (6 files): 19 rewrite, 1 resize, 3 split, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M6 | the tests of `foundation/blas_test_oracle` (4 files): 6 rewrite, 1 resize, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M7 | the tests of `foundation/core` (9 files): 18 rewrite, 3 resize, 1 split, 2 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M8 | the tests of `foundation/cost` (4 files): 5 rewrite, 7 split, 6 merge, 4 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M9 | the tests of `foundation/geometry` (6 files): 15 rewrite, 1 resize, 3 split, 2 merge, 5 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M10 | the tests of `foundation/graph_primitives` (14 files): 82 rewrite, 10 resize, 7 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M11 | the tests of `foundation/host_device` (3 files): 1 rewrite, 3 resize, 1 split, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M12 | the tests of `foundation/launch` (6 files): 14 rewrite, 14 resize, 1 split, 1 merge, 8 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M13 | the tests of `foundation/orthonormal` (1 file): 3 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M14 | the tests of `foundation/profile` (3 files): 2 rewrite, 3 resize, 3 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M15 | the tests of `foundation/reduction` (2 files): 4 rewrite, 8 resize, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M16 | the tests of `foundation/reference` (3 files): 4 rewrite, 10 resize, 1 split, 4 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M17 | the tests of `foundation/serial_reductions` (1 file): 3 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M18 | the tests of `foundation/small_integers` (1 file): 2 rewrite, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M19 | the tests of `foundation/sparse_assembly` (1 file): 2 rewrite, 14 resize | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M20 | the tests of `foundation/sparse_fixtures` (2 files): 7 rewrite, 2 split, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M21 | the tests of `foundation/substitution` (1 file): 4 rewrite, 3 merge, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M22 | the tests of `foundation/test_runner` (4 files): 6 rewrite, 1 resize, 1 split, 1 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M23 | the tests of `foundation/typed_suite_naming` (1 file): 1 resize, 1 merge | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M24 | the tests of `kernels/blas_kernels` (56 files): 55 rewrite, 2 resize, 12 split, 1 merge, 10 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M25 | the tests of `kernels/blas_seam_kernels` (25 files): 62 rewrite, 14 resize, 4 split, 5 merge, 5 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M26 | the tests of `kernels/blend_kernels` (3 files): 4 rewrite, 1 merge | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M27 | the tests of `kernels/linalg_kernels` (7 files): 14 rewrite, 3 split, 1 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M28 | the tests of `kernels/packed_product_harness` (1 file): 1 rewrite, 1 resize, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M29 | the tests of `kernels/random_kernels` (6 files): 12 rewrite, 4 resize, 2 split, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M30 | the tests of `kernels/signal_kernels` (6 files): 10 rewrite, 2 merge, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M31 | the tests of `kernels/sparse_kernels` (16 files): 13 rewrite, 1 resize, 4 split, 6 merge, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M32 | the tests of `kernels/spatial_kernels` (19 files): 6 rewrite, 6 split, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M33 | the tests of `library` (13 files): 63 rewrite, 26 resize, 8 split, 4 merge, 16 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M34 | the tests of `operations/benchmark_harness` (3 files): 4 rewrite, 8 resize, 2 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M35 | the tests of `operations/blas` (53 files): 327 rewrite, 41 resize, 4 split, 16 merge, 126 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M36 | the tests of `operations/blend` (1 file): 11 rewrite, 1 resize, 2 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M37 | the tests of `operations/elementwise` (38 files): 38 rewrite, 52 resize, 6 split, 7 merge, 7 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M38 | the tests of `operations/experimental` (3 files): 6 rewrite, 11 resize | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M39 | the tests of `operations/index` (1 file): 3 rewrite, 2 resize, 3 split, 11 merge, 2 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M40 | the tests of `operations/io` (5 files): 15 rewrite, 8 resize, 7 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M41 | the tests of `operations/logic` (15 files): 13 rewrite, 2 split, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M42 | the tests of `operations/nonlinear` (10 files): 22 rewrite, 1 resize | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M43 | the tests of `operations/random` (11 files): 32 rewrite, 1 resize, 8 split, 11 merge, 4 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M44 | the tests of `operations/reductions` (13 files): 66 rewrite, 3 resize, 12 split, 22 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M45 | the tests of `operations/signal` (1 file): 5 rewrite, 1 resize, 1 split | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M46 | the tests of `operations/sparse` (12 files): 50 rewrite, 6 resize, 9 split, 8 merge, 8 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M47 | the tests of `operations/spatial` (12 files): 44 rewrite, 17 resize, 12 split, 1 merge, 1 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M48 | the tests of `package` (1 file): 1 rewrite | T1, T2, T3, T4, T5, T6, T7, T8 | todo |
| M49 | the tests of `tools` (7 files): 11 rewrite, 37 resize, 11 split, 14 merge, 5 delete | T1, T2, T3, T4, T5, T6, T7, T8 | todo |

## Acceptance

- **T1.** No comparison spells a tolerance literal, and an exact comparison is spelled as equality. Build: the repository's test command over the touched tests.
- **T2.** The host decisions run small with no runtime. Build: the repository's test command over each touched module.
- **T3.** No promise is tested twice. Build: the repository's test command.
- **T4.** No benchmark sits in the test tree. Build: the repository's test command and its benchmark command.
- **T5.** Every defect above is fixed or has a backlog entry. Build: the repository's test command and `python tools/check_all.py .`
- **T6.** No CLI test patches the repository's own functions. Build: the CLI tests.
- **T7.** The rows run here. Build: the repository's test command over device math.
- **T8.** Neither file exists and the checkers pass. Build: `python tools/check_all.py .`
- **M jobs.** Every record of the module is applied: no test the report deletes remains, every kept and
  new test passes at the size its label states, and the module's tests pass through the project's own
  test command. Build: that command over the module.

The owner chooses inline or parallel mode.

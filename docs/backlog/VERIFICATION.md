# Verification

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-10 | P2 | unmeasured | Run the benchmark programs other than `benchmark_band_packed` | each benchmark program has a recorded run on the module tree |  |  | 2026-10-19 |
| RW-11 | P2 | unmeasured | Record why CTest skips tests on the CUDA build | each skipped test has its reason written down |  |  | 2026-10-19 |
| RW-12 | P2 | unmeasured | Measure the parity allowances of the random reference tests | each allowance of `test_random_reference.cpp` that proves too tight is raised with the measured value beside it, and `LargeLambdaPoissonGivesTheSameCounts`… |  |  | 2026-10-19 |
| RW-13 | P2 | unmeasured | Check the rotation bricks and `Core::IndexBase` against their sources | `givens.hpp` and `modified_givens.hpp` match `drotg.f90`, `zrotg.f90` and `drotmg.f` of a LAPACK checkout, and the two values of `Core::IndexBase` match the… |  |  | 2026-10-19 |
| RW-14 | P2 | unmeasured | See each test added since 2026-10-05 fail on the old code | every validation, parity and edge-case test added since then fails on the old code and passes on the new |  |  | 2026-10-19 |
| RW-15 | P2 | unmeasured | See the record-time refusal of a device that cannot host the SYRK tile | `require_local_tile` has thrown on a device that cannot host the `syrk` or `syr2k` tile |  |  | 2026-10-19 |
| RW-16 | P2 | unmeasured | Configure and build as a consumer does | the tree configures with `SB_BUILD_TESTS` and `SB_BUILD_EXAMPLES` off, and `tests/package/` builds against an installed tree |  |  | 2026-10-19 |
| RW-17 | P2 | unmeasured | Compile the kernels for a second target | a second ahead-of-time target and a SPIR-V device have compiled the kernels, `Kernels::multiply`, `divide` and `conjugate` among them |  |  | 2026-10-19 |
| RW-18 | P2 | open | Settle the open items each design lists | each open or unverified item of the designs is settled or moved here |  |  | 2026-10-19 |
| RW-101 | P2 | open | State and test that tensors may be created from several threads | `TensorFactory::create` and `Engine::create_tensor` state it, and a test creates and releases tensors from eight threads |  |  | 2026-10-19 |
| RW-102 | P2 | open | State and test host access to a Shared tensor between executions | a plan compiled once copies a source the host rewrites between three executions, and the destination matches each time |  |  | 2026-10-19 |
| RW-129 | P2 | open | Bring the tests to the TEST_PIPELINE doctrine, tier by tier | check_test_value, check_test_sizes and the shuffled run report no finding in small, medium and large |  |  | 2026-11-02 |
| RW-130 | P2 | open | Give test_budget.py a program map so the small and medium budgets gate | sb test prints per-program small totals and a whole-run total, and a run over 10 s per program or 375 s default fails the gate |  | `tools/testing/test_budget.py` | 2026-11-02 |
| RW-131 | P3 | open | Name why each float EXPECT_EQ may stay, or give it a tolerance | no float EXPECT_EQ is left whose test name lacks the reason, incl. test_modified_givens.cpp:195-196 and the SYCL-vs-reference == checks |  | `tests/kernels/blas_seam_kernels/test_modified_givens.cpp` | 2026-11-02 |
| RW-132 | P3 | open | Move static_assert-only Medium_ cases to Small_ | no Medium_ case holds only static_assert, such as Medium_ReferenceAccumulator and Medium_PackedAccess ones |  | `tests/foundation/reference/test_reference_accumulator.cpp` | 2026-11-02 |
| RW-136 | P3 | open | Move the graph_primitives tests onto the program-wide runtime | Only the tests whose runtime settings differ build their own runtime; the README says so |  | `docs/agent/2026_11_05_TEST_REVIEW/REPORT.md` | 2026-11-06 |
| RW-138 | P3 | open | Make PoissonAcceptsTheCap fail on a constant fill | The test in test_random_large_lambda.cpp checks the draws vary, so a constant fill fails it |  | `docs/agent/2026_11_05_TEST_REVIEW/REPORT.md` | 2026-11-06 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |

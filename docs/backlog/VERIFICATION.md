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

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |

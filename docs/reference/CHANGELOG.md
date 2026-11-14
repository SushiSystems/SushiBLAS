# Changelog

The entries dated before 2026-09-07 were reshaped to one line each on 2026-10-05, and their
original text is frozen in the changelog document in the version history.

## Unreleased

## v0.1.2 — 2026-11-13

- 2026-11-13 — reference: Added no offset to the null input of an empty axis in `reduce_axis`, which UBSan reported in the sanitizer job (`serial_reduce.hpp`).
- 2026-11-13 — ci: Installed each Sushi library CLI in its own pipx environment, checked SushiRuntime out at its pinned tag, fetched the full history for the checkers and printed the tests over their time limit on a failure (`ci.yml`).
- 2026-11-13 — deps: Pinned SushiRuntime v0.5.2, which pins the same SushiLog as this repository, so `sb setup` no longer refuses two SushiLog pins (`sushistack.deps.toml`).
- 2026-11-13 — tests: Put `axis_name` of the gemm rows beside `GemmCase`, so the row name template finds it by argument-dependent lookup (`test_gemm.cpp`).
- 2026-11-11 — tests: Moved the `ArgmaxIndex` mantissa case to medium and exempted the six deep complex and half GEMM agreement rows over the medium limit (RW-160).
- 2026-11-11 — cli: Required sushicore 0.13, whose test command fails a run of a current tree that breaks a time limit, and told it whether the tree is current; the large shards each hold the one accelerator slot.
- 2026-11-10 — cli: Required sushicore 0.12.3, and ran the unknown-theme row and the malformed-config row small over its in-memory environment and files (RW-157, `73b44fcd`).
- 2026-11-10 — cli: Required sushicore 0.12.2, which narrows a sharded test run by `--filter` and reruns a failed shard's cases.
- 2026-11-10 — tests: Registered each test program's cases as CTest shards per size (RW-158, `7ed5b15d`).
- 2026-11-07 — verification: Let the host sink report what each recorded task stated, so the index-ops recording and refusal tests run small (RW-149, `62dba19b`).
- 2026-11-07 — general: Give the graph primitives a value-level seam so their capacity and refusal tests run small (RW-147, `e70bbcfc`).
- 2026-11-07 — verification: Give the engine fixture a host tensor and host run, so the LogicOps tests run small (RW-150, `d735f640`).
- 2026-11-07 — verification: Let RandomOps record on an in-memory task recorder, so its refusal tests run small (RW-151, `d735f640`).
- 2026-11-07 — io: Give io a header parser over bytes so the npy and sushi header refusals run small (RW-148, `9eb150e7`).
- 2026-11-07 — verification: Run the reduction refusals, dtype, empty-input and lease rules small on a host engine (RW-152, `b79973ee`).
- 2026-11-07 — verification: Make Engine hold no nullable runtime: is_host() and runtime() stop differing by kind (RW-155, `e64552f6`).
- 2026-11-07 — verification: Give ScratchPool no nullable runtime: it lends from its ScratchBlockSource alone (RW-156, `e64552f6`).
- 2026-11-07 — verification: Gave ScratchPool a block source port with a host adapter, so the lease tests run small (RW-153, `c986a77f`).
- 2026-11-07 — general: Gave the row-blocks fold a host function and a device GMRES test a poisoned start (RW-146, `6c2e8f68`).
- 2026-11-07 — profiling_logging: Gave the profile gate a reading port, so the reading side runs small and SB_ZONE's writing side stays medium (RW-140, `009e0dbc`).
- 2026-11-07 — profiling_logging: Gave the logger a port with an in-memory sink so its tests run small (RW-142, `5a1ffa4a`).
- 2026-11-07 — general: Replaced the test_runner LogCapture stand-in with SushiLog::LogCapture (RW-143, `5a1ffa4a`).
- 2026-11-07 — gpu_validation: Gave make_host_queue a device-list seam and SharedArray an allocator port (RW-141, `4529f987`).
- 2026-11-07 — general: Gave the operations refusal checks host-callable seams so their tests run small (RW-144, `3476a105`).
- 2026-11-07 — general: Moved the forced-mode and refusal test helpers into tests/support (RW-145, `3476a105`).
- 2026-11-07 — benchmarks_tuning: Ruled that micro benchmarks stay beside their tests and build into benchmark programs, never into a test program (RW-135, `20928ee7`).
- 2026-11-07 — verification: Ruled that HostDeviceTest takes the program-wide CPU queue from the test program's Environment (RW-137, `20928ee7`).

- 2026-11-06 — licence: Relicensed SushiBLAS under the Apache License 2.0 and removed `COMMERCIAL.md`; every source file's licence block follows (`LICENSE`, `tools/settings.py`).
- 2026-11-05 — tools: Took the shared checkers of 2026-11-05 with a baseline of the findings they meet, which RW-133 clears.
- 2026-11-02 — tests: Renamed the asserting helpers to `expect_*` (`expect_tensor`, `expect_accuracy`), so the checker counts 177 medium cases; removed the timers of four stress tests and `SUCCEED()` from six cases (`tests/`).
- 2026-11-02 — tests: Gave the small tests that asserted a literal or nothing a real assertion: `static_assert` alone for compile-time facts, `EXPECT_NO_THROW` for calls that must return, a tolerance for four adjoint identities (`tests/`).
- 2026-11-02 — cli: Moved `sb test` onto the shared test command of sushicore 0.10.11, which shuffles and prints the seed and the budget line, added `--all`, `--shuffle` and `--seed` and dropped the local label module (RW-128, `a1edb237`).
- 2026-11-02 — cli: Left `benchmark_gemm_blocking` out of a bare `sb bench`, since it needs arguments and failed the nightly with a usage error; `sb bench gemm_blocking -- <args>` still runs it (`services/project.py`).
- 2026-11-01 — bench: Gave the triangular benchmark sizes by device when none is given: 32, 128 and 512 on a CPU, 64, 512 and 4096 on a GPU, so the CPU nightly runner finishes (`tests/operations/blas/benchmark_triangular.cpp`).

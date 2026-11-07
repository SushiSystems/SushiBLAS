# Changelog

The entries dated before 2026-09-07 were reshaped to one line each on 2026-10-05, and their
original text is frozen in the changelog document in the version history.

## Unreleased

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

## v0.1.1 — 2026-10-31

- 2026-10-31 — ci: Brought the Windows pre-merge job inside its 150-minute timeout, with `sb build` taking about 78 minutes on main (`.github/workflows/ci.yml`).
- 2026-10-30 — cli: An unknown theme in `cli/config.toml` now runs at the default with one warning naming it, as sushicore 0.10.6 makes it, rather than exiting 1; the CLI tests check that (`cli/tests/test_cli.py`).
- 2026-10-30 — cli: Made `sb build --sanitize thread` link SushiLog, SushiProfiler and SushiRuntime packages built with ThreadSanitizer, since TSan reported races through the uninstrumented Release runtime (`library_packages.py`).
- 2026-10-30 — deps: Moved the SushiRuntime pin to v0.5.1, which pins SushiLog v0.3.0 and SushiProfiler v0.4.1 as this repository does (`cli/sushistack.deps.toml`).
- 2026-10-30 — tests: Fixed every Windows Debug test program crashing on load; each copies the Release gtest.dll it links over the Debug one vcpkg put beside it (`cmake/SushiBLASTests.cmake`).
- 2026-10-29 — ci: Gave the build jobs 120 to 150 minutes; each now builds the SushiRuntime package too, and its device compiles miss the cache (`.github/workflows/ci.yml`).
- 2026-10-29 — profiling: Fixed the profiling end-to-end tests deleting each other's folder when they run in parallel; each case writes under its own name (`test_profiler_parts.cpp`).
- 2026-10-29 — ci: Ran `sr setup` in the SushiRuntime checkout of the Windows job, whose package build needs the vcpkg and Ninja paths it records (`.github/workflows/ci.yml`).
- 2026-10-29 — blas_seam_kernels: Fixed the tiled GEMM launcher failing a `-Werror` build on a role parameter it never read, and the packed GEMM on a batch size its plan already holds (`gemm.hpp`, `gemm_packed.hpp`).
- 2026-10-29 — build: Gave the Debug line tables to the host pass alone, since a clang SYCL device pass rejects the flag under `-Werror` (`cmake/`).
- 2026-10-29 — build: Let a Debug tree link an imported package built as `None`, such as Ubuntu's GTest (`cmake/`).
- 2026-10-29 — cli: Fixed a Debug or sanitized build asking for Debug packages beneath it; it asks for the Release closure, as SushiRuntime does (`services/library_packages.py`).
- 2026-10-29 — deps: Asked for sushicore 0.10.4, which gives a package build on icpx the icx beside it for the C slot (`cli/pyproject.toml`).
- 2026-10-29 — cli: Checked that each library of the closure has a tag and a full commit pinned, not one fixed SushiProfiler tag, so a pin move no longer fails the CLI tests (`tests/test_library_packages.py`).
- 2026-10-29 — deps: Moved the SushiProfiler pin to v0.4.1, whose CLI installs against sushicore 0.10 (`cli/sushistack.deps.toml`).
- 2026-10-29 — ci: Installed the CLI of every Sushi library checked out beside sushiblas, since a library package is built by that library's own CLI (`.github/actions/`).
- 2026-10-29 — deps: Moved the SushiLog pin to v0.3.0, the first tag a consumer can build as a package (`cli/sushistack.deps.toml`).
- 2026-10-29 — ci: Checked out SushiProfiler at its pinned tag beside SushiLog in every build job, and installed the CLI before the checkers so they find sushicore (`.github/workflows/ci.yml`).
- 2026-10-29 — cli: Imported `configure_arguments` from sushicore where the build calls it, and listed sushicore's `backlog` and `design` groups in the command-list tests (`services/project.py`).

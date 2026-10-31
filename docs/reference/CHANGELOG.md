# Changelog

The entries dated before 2026-09-07 were reshaped to one line each on 2026-10-05, and their
original text is frozen in the changelog document in the version history.

## Unreleased

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

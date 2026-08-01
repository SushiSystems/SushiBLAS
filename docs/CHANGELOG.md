# Changelog

All notable changes to SushiBLAS are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — versions follow [Semantic Versioning](https://semver.org/).

The project has not yet been assigned a semantic version — it is in
prototype stage. Formal per-release entries begin at the v1.0.0 cut; until
then, `[Unreleased]` records the prototype-era milestones informally so the
history of *why* the codebase looks the way it does isn't lost. See
[CONTRIBUTING.md §6](CONTRIBUTING.md#6-documentation-is-part-of-the-change).

---

## [Unreleased]

### Known issues
- **Unresolved Release-mode test compile failure (found 2026-08-02, not yet
  diagnosed).** A `sa build --type release` run from the sushiai repo (which
  builds this repo as a sibling) reached `[153/219]` and failed to compile
  three existing test files — `tests/functional/unit/test_constant.cpp`,
  `test_leaky_relu.cpp`, and `test_gelu.cpp` — under `-O3 -DNDEBUG`. The
  captured build log only retained the `FAILED:` marker lines and the full
  `clang++` invocation for each, not the actual `error:` diagnostic text, and
  no build has been re-run since to capture it (builds were paused this
  session). These three files build fine in the Debug-ish configurations this
  repo's own `sb test` has been run under elsewhere in this changelog — this
  looks Release/`-DNDEBUG`-specific. **Next step:** re-run `sb build --type
  release` (or `sa build --type release` from sushiai) with full output
  captured and read the actual compiler diagnostic.

### Added
- **Zero oneMKL dependency, anywhere.** Every BLAS level (1, 2, 3, including
  TRSM/TRSV) now runs on SushiBLAS's own from-scratch SYCL kernels. Level 3
  (GEMM, SYRK) uses a local-memory-tiled kernel derived from portBLAS's
  tiling algorithm (Apache-2.0, see [NOTICE.md](../NOTICE.md)); levels 1, 2,
  and TRSM/TRSV are original SushiBLAS work. All five `Core::DataType`s
  (including `HALF`, which oneMKL never supported for several of these
  routines) are implemented uniformly across all 17 ops.
- **SushiBLAS is consumable out of tree.** `install(EXPORT SushiBLASTargets
  NAMESPACE SushiBLAS::)` plus a generated `SushiBLASConfig.cmake` and version
  file, so `find_package(SushiBLAS REQUIRED)` and
  `target_link_libraries(... SushiBLAS::sushiblas)` work from a project with
  no access to this source tree. `tests/package/` is a complete working
  consumer and asserts, from the only side it is observable, that the
  exported interface carries nothing but the include paths, the library, and
  SushiRuntime's transitively-inherited floating-point contract — no
  `-Werror`, no `-march`, no sanitizer selection, and not even the in-tree
  policy targets' names.
- `include/SushiBLAS/version.hpp` — `SUSHIBLAS_VERSION_{MAJOR,MINOR,PATCH,STRING}`
  and `SUSHIBLAS_VERSION_ENCODE` for compile-time branching. Checked against
  the CMake project version at configure time so the two statements of the
  version cannot drift. Unlike SushiRuntime (a `SHARED` library), SushiBLAS is
  `STATIC`, so there is no separately loaded artifact whose version could
  disagree with the compiled-in macro at run time — no `version_matches()`
  runtime check exists or is needed.
- A hand-written Philox4x32-10 counter-based SYCL PRNG
  (`kernels/sycl/random/philox.hpp`, constants verified against the Random123
  reference implementation), replacing oneMKL's VSL backend across all 11
  distribution ops (`uniform`, `normal`, `log_normal`, `exponential`,
  `poisson`, `bernoulli`, `discrete_uniform`, and the Xavier/He initializers).
- `Internal::execute_blas_task` / `Internal::dispatch_with_mode`
  (`engine/blas/utils.hpp`) — the shared dispatch helpers every BLAS op now
  uses instead of hand-writing the runtime-`DataType` switch and the
  `FORCE_REFERENCE`/normal/`VERIFY` triad at each call site (previously
  duplicated 5x per op in `gemm.cpp`/`syrk.cpp`).
- `sushiruntime unit/integration/regression` test structure: `tests/` is now
  `tests/common/` + `tests/functional/{unit,integration,regression}/`, with
  every GTest fixture renamed to the `Unit_`/`Integration_`/`Regression_`
  suite-name-prefix convention so `gtest_discover_tests(TEST_FILTER ...)` and
  CTest labels work identically to sushiruntime/sushiengine. The `sb` CLI's
  `--suite` option (`unit`/`integration`/`regression`/`functional`/`all`)
  selects a CTest label group.
- `NOTICE.md` — Apache-2.0 attribution for the portBLAS-derived GEMM tiling
  algorithm, and documentation of the (original, not ported) SYRK-via-GEMM
  composition and every deliberate simplification versus the upstream
  kernel.
- Full documentation set (this pass): `ARCHITECTURE.md`, `CLI_GUIDE.md`,
  `INTEGRATION.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, this changelog, and
  a rewritten `README.md`/`CONTRIBUTING.md`, mirroring SushiRuntime's
  documentation structure. `LAYOUT.md`, `STYLE_GUIDE.md`, and `TODO.md` are
  folded into `ARCHITECTURE.md` §5-7 and `CONTRIBUTING.md` §4-5 rather than
  kept as separate files.

### Changed
- **No more vendored SushiRuntime.** SushiBLAS used to check in a snapshot of
  SushiRuntime's headers and prebuilt library (`include/SushiRuntime/`,
  `lib/{sushiruntime,hwloc-15}.*`). It now resolves a live dependency through
  the same [D3 layered CMake resolution](ARCHITECTURE.md#8-the-sushiruntime-dependency-d3)
  pattern used stack-wide: a superproject TARGET-guard, then
  `find_package(SushiRuntime QUIET)`, then a sibling `add_subdirectory`.
  Every consumer links the identical `SushiRuntime::SushiRuntime` alias
  regardless of which mode resolved it.
- **No more real Intel oneAPI DPC++ requirement.** The build no longer calls
  `find_package(IntelSYCL REQUIRED)` / `find_package(OpenCL REQUIRED)` /
  `/Qmkl`. It defaults to the shared bundled intel-llvm `clang++`
  (`cmake/{Vcpkg,Toolchain,SyclTarget}.cmake`, mirroring sushiengine's own
  toolchain files), which is also what unblocked real compiler verification
  for the first time in the project's SB-0→SB-2 arc and surfaced (and let us
  fix) several months of dormant SushiRuntime API drift
  (`Memory::AllocStrategy` → `IAllocatorStrategy` singletons,
  `TaskGraph::execute()` → `compile()` + `IExecutionEngine::execute()`).
- **The `sb` CLI now matches the bundled-toolchain build.**
  `cli/sushiblas/{config,env}.py` and `services/project.py` used to model the
  pre-SB-0/SB-2 build (a vendored-runtime existence check, a real Intel
  oneAPI `icx-cl`/`icpx` compiler resolution) and had drifted out of sync
  with the CMake build they were supposed to drive. They now resolve the
  SushiRuntime sibling and the shared bundled `clang++` the same way
  `cmake/Runtime.cmake`/`Toolchain.cmake` do, mirroring sushiengine's `se`
  CLI. `_configure_args()` also gained the C-compiler-slot fix every
  `LANGUAGES CXX C` project in the stack needs: a `clang++`-personality
  binary hardcodes C++ mode regardless of file extension, so
  `CMAKE_C_COMPILER` must point at the toolchain's sibling `clang` binary,
  not the same `clang++` path used for `CMAKE_CXX_COMPILER`.
- Build flags no longer leak to consumers. The old directory-scope
  `add_compile_options(-march=native / -Wall... / /fp:fast)` block is now
  three policy targets (`sushiblas_warnings`, `sushiblas_perf`,
  `sushiblas_sanitizers`) composed by `sushiblas_apply_build_policy()`,
  applied `PRIVATE` per target. Because `sushiblas` is a `STATIC` library,
  the policy targets' flag *values* are pulled onto the real target via
  `target_compile_options`/`target_compile_definitions`/`target_link_options`
  + `$<TARGET_PROPERTY:...>` rather than `target_link_libraries` — the latter
  records a policy target's name in the exported `INTERFACE_LINK_LIBRARIES`
  even `BUILD_INTERFACE`-wrapped, which breaks `install(EXPORT)`/
  `export(EXPORT)` for a static archive.
- `LinalgOps`/`TransformsOps` (SVD/QR/eigendecomposition, FFT/DFT) relocated
  to `include/SushiBLAS/experimental/` and dropped from the `Engine` facade
  (no `engine.linalg()`/`engine.signal()`). Both were declaration-only with
  zero implementations and no callers, so this was a zero-risk move, not a
  behavior change (D1).

### Fixed
- **GEMM/SYRK NaN propagation when `beta == 0`.** Both kernels computed
  `C[idx] = alpha*sum + beta*C[idx]` unconditionally, including when
  `beta == 0`; a freshly-allocated, never-initialized `C` tensor whose
  garbage bit pattern happened to decode as a `HALF`-precision NaN poisoned
  the whole output, since `0 * NaN == NaN` in IEEE-754 — standard BLAS
  semantics never read `C` at all when `beta == 0`, for exactly this reason.
  Fixed with an explicit `beta == T(0)` branch in all four kernels
  (`kernels/{reference,sycl}/blas/level3/{gemm,syrk}.hpp`).
- **`fill_random`/`assert_tensor_near` silently no-op for rank < `MAX_TENSOR_RANK`.**
  Both test helpers computed a tensor's element count by multiplying *all six*
  slots of `Tensor::shape` (zero-padded past the tensor's actual rank), so any
  operand with rank below 6 — every existing test — always multiplied in the
  zero padding and got `size == 0`, silently copying zero bytes. GEMM/SYRK's
  earlier "44/44 passed" results were real (both the SYCL and reference paths
  ran on the same, just-not-actually-randomized input) but exercised no real
  random coverage. Fixed by using `Tensor::num_elements` directly.
- `dot.cpp`'s use of `Internal::is_complex_v` after deleting the now-dead
  `level1_internal.hpp` (the trait's only prior definition) — moved into the
  shared `engine/blas/utils.hpp` instead.
- `gemv`'s `HALF` case and `IO::to_string()`'s `HALF` printer branch: both
  used to be a silent gap (a `switch` fallthrough, and an all-zero print
  respectively) rather than a loud, explicit unsupported-case error.
- Two identically-shaped CMake module-filename collisions, both from D3's
  sibling `add_subdirectory` putting multiple repos' `cmake/` directories on
  one shared `CMAKE_MODULE_PATH`: SushiBLAS's own
  `BuildPolicy`/`CompilerWarnings`/`Optimization`/`Sanitizers.cmake` files
  were renamed to `SushiBLAS`-prefixed names after they shadowed
  SushiRuntime's identically-named files and broke *its* build; and
  `cmake/Runtime.cmake`'s `SUSHISTACK_HOME`-based sibling default gained a
  missing `EXISTS` check (it previously trusted the environment variable's
  path unconditionally, which resolves to a nonexistent directory on a
  machine where `SUSHISTACK_HOME` is set globally but modules are checked out
  flat rather than inside it).
- `SushiRuntimeConfig.cmake.in`'s `find_dependency(IntelSYCL)` (in the
  SushiRuntime sibling, fixed here because SushiBLAS's own SB-6 packaging
  verification is what exercised it for the first time) and its exported
  alias case mismatch (`SushiRuntime::sushiruntime` vs. the expected
  `SushiRuntime::SushiRuntime`) — both blocked `find_package(SushiBLAS)`'s
  transitive `find_dependency(SushiRuntime)` from ever succeeding.

### Removed
- The vendored SushiRuntime snapshot (`include/SushiRuntime/`,
  `lib/{sushiruntime,hwloc-15}.*`).
- The stale `scripts/win/` and `scripts/linux/` build scripts, superseded by
  the `sb`/`sushiblas` CLI.
- `LAYOUT.md`, `STYLE_GUIDE.md`, `TODO.md` as standalone docs (folded into
  `ARCHITECTURE.md` and `CONTRIBUTING.md` — see Added above).

---

<!--
## [x.y.z] — YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing behavior

### Fixed
- Bug fixes

### Removed
- Removed features

[x.y.z]: https://github.com/SushiSystems/SushiBLAS/compare/vX.Y.Z-1...vX.Y.Z
-->

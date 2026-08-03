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
None open.

#### Closed
- **Release-mode test compile failure (reported 2026-08-02) — not reproducible,
  closed 2026-08-03.** The original report was a `sa build --type release` from
  the sushiai repo (which builds this one as a sibling) that reached
  `[153/219]` and failed on `tests/functional/unit/test_constant.cpp`,
  `test_leaky_relu.cpp` and `test_gelu.cpp` under `-O3 -DNDEBUG`. The captured
  log had kept only the `FAILED:` markers and the `clang++` invocations, not
  the diagnostics, so there was nothing to diagnose from.

  Both sides have since been re-run with full output. `sb build --type release`
  here compiles all three under `-O3 -DNDEBUG -Werror` with no diagnostics and
  the suite is green at 182/182. `sa build --type release` from sushiai reaches
  208/208 with those same three files compiling clean at steps 133, 138 and
  140, and `sa test --suite all` passes 155/155. So it was neither these files
  nor a cross-repo configuration difference. The likeliest explanation is a
  stale build tree at the time of the original run, which the reconfigure has
  replaced. Kept on record rather than deleted so a recurrence is recognised as
  a repeat rather than filed as a new bug.

### Added
- **The op-dispatch seam is public API (`namespace SushiBLAS::Dispatch`).** The
  five functor-templated `*_internal.hpp` helpers moved out of `src/` and are
  installed as `include/SushiBLAS/engine/{math/elementwise,math/nonlinear,
  math/reductions,math/random,logic}/dispatch.hpp`, renamed from `_internal`
  now that they are not internal. A downstream framework composes several math
  ops into one plain C++ lambda, hands it to one `execute_binary` /
  `execute_unary_inplace` / `execute_nonlinear_*` / `execute_logic_*` /
  `execute_reduction` call, and gets a single fused SYCL kernel instead of one
  launch per op — which is the whole basis of operator fusion for the layer
  above, and keeps "the consumer never writes SYCL itself" true. Two supporting
  headers came with it: `engine/dispatch.hpp` (the shared `Domain` vocabulary
  and the documented constraints on a device-inlined functor) and
  `engine/math/safe_math.hpp` (the `safe_exp`/`safe_log`/`safe_sqrt`/`safe_abs`
  overload sets that let one functor cover real and complex element types).
  `tests/package/main.cpp` now calls `Dispatch::execute_binary` with its own
  lambda against a `find_package`'d install, so the reachability claim is
  tested from outside the tree rather than asserted.
- **`RandomOps::shuffle` is a real Fisher-Yates permutation.** It was a no-op
  stub, commented out of `src/CMakeLists.txt`, and declared in an installed
  header — so a consumer that called it compiled and then failed to link.
  Implemented as a sequential `single_task` over the engine's Philox stream
  (new `Kernels::SYCL::Random::draw_index`, integer-only so a shuffle does not
  depend on the device's fp64 support), supporting every dtype since it moves
  elements rather than computing on them. Reproducible from a fixed seed.
- **`RandomOps::orthogonal` produces an actually orthogonal matrix.** Same link
  trap as `shuffle`, plus a worse problem: the quarantined implementation
  forwarded to `normal()`, so un-commenting it would have shipped a Gaussian
  matrix under an orthogonal name. Now fills with N(0,1) and orthonormalizes by
  modified Gram-Schmidt (`kernels/sycl/random/orthonormalize.hpp`) — the same Q
  a QR would give, without a QR. FLOAT32/FLOAT64 rank-2 only; everything else
  throws rather than falling back.
- **Complex support where it means something, a loud failure where it does
  not.** `add`/`sub`/`mul`/`div`/`neg`/`exp`/`log`/`sqrt`/`square`/
  `reciprocal`/`abs` now compute on COMPLEX32/COMPLEX64 instead of silently
  doing nothing, as do `equal`/`not_equal`/`and`/`or`/`xor`/`not`/`where`.
- **`Core::to_string(DataType)` and `Core::is_complex_dtype(DataType)`**
  (`core/common.hpp`), so a dtype rejection can name the offending type.
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
- **Seven silent no-ops in dtype switches.** `default: break;` in the
  elementwise (x2), nonlinear (x2) and logic (x3) dispatch helpers meant an
  unsupported dtype produced a command group with no kernel enqueued: `add()`
  on a COMPLEX32 tensor returned successfully and changed nothing. Every one of
  them now throws naming the operation and the dtype, and the reductions helper
  and `execute_random` got the same treatment for their already-throwing
  branches. This is P1 (fail loudly, never silently) applied to the last places
  in the library that violated it.
- **Lexicographic ordering of complex numbers, removed.** `less`/`less_equal`/
  `greater`/`greater_equal` had complex overloads that compared `(real, imag)`
  lexicographically — a confident, well-formed, mathematically meaningless
  answer. Those overloads are gone and the ordering ops declare `Domain::REAL`,
  so a complex operand throws at record time. Same for `max`/`min`/`clamp`/
  `floor`/`ceil`/`round`/`fmod`/`remainder`.
- **`is_complex` / `is_complex_v` declared three times in one namespace.** All
  three lived in `SushiBLAS::Internal` (`engine/blas/utils.hpp` plus the
  elementwise and random private helpers) — harmless only while no translation
  unit saw more than one, a guarantee that promoting the dispatch helpers to
  public headers destroyed. Collapsed into one definition,
  `SushiBLAS::Core::is_complex` in the new `include/SushiBLAS/core/traits.hpp`;
  `utils.hpp` includes it so existing includers still compile. A fourth copy in
  `tests/common/test_utils.hpp` is gone too.
- **RNG seed and offset were being written to task-metadata slots that do not
  exist.** `random_internal.hpp` (and `shuffle.cpp`) called `set_param(10, ...)`
  and `set_param(11, ...)` on a `TaskMetadata` holding eight; `set_param`
  ignores an out-of-range index silently, so every random task recorded a zero
  seed and a zero offset. The kernels were unaffected — they capture seed and
  offset by value, not through the metadata — so no number ever came out wrong,
  but the record of which draw a task made was blank. Now slots 6 and 7 by
  convention (`Dispatch::K_RNG_SEED_PARAM` / `K_RNG_OFFSET_PARAM`), stored as
  raw `uint64_t` rather than lossily cast to `double`, with distribution
  parameters capped at six and a `Dispatch::set_task_params` helper that throws
  rather than dropping an overflow. Fixed entirely on the SushiBLAS side;
  sushiruntime is unchanged.
- **`using namespace SushiRuntime::Graph::Literals;` no longer leaks from a
  public header.** The old private helpers injected it into `namespace
  SushiBLAS` at namespace scope, which was tolerable while they were private
  and is not once they are installed. Each `.cpp` that uses the `_op` literal
  now says so itself.

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

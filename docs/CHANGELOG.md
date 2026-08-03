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
- **GEMM ships two kernel shapes and picks one from the device profile**
  (`include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`, moved there when the
  epilogue seam was made public). `Detail::gemm_local_tiled` is the
  existing cooperative local-memory kernel; `Detail::gemm_no_local` drops the
  `local_accessor` and both barriers entirely and reads the operands straight
  from global memory. The split is portBLAS's own (`gemm_local` vs.
  `gemm_no_local`) and it exists because on a CPU backend the two kernels are
  close to opposites: local memory there is the same cache the global pointer
  already reads through, so staging a tile copies the data twice and every
  `group_barrier` is a real rendezvous between OS threads buying nothing. Until
  now this library ran the GPU shape unconditionally, including on the only
  device it can currently reach. Selection is a plain runtime `if` over two
  compile-time configurations — the accumulator is a fixed-size private array,
  so the two variants are two instantiations, not one function threaded with
  `if constexpr`. The local tile is checked against `local_mem_bytes` and
  `max_work_group_size` at launch and falls back with a warning rather than
  failing the launch, and the variant that ran is logged at `SB_LOG_INFO`.
- **An `Epilogue` template parameter on `Kernels::SYCL::gemm`**, defaulting to
  `GemmIdentityEpilogue` and applied to each element immediately before the
  store. It receives `(value, row, col)`; the coordinates are what make a bias
  broadcast expressible rather than only elementwise work, which is what turns
  it into a real fusion hook — the layer above can fold a bias-add and an
  activation into GEMM instead of paying two more kernel launches. The
  `@warning` on it documents the trap that makes this dangerous rather than
  merely useful: an *activation* epilogue in a forward **training** pass is
  silently wrong unless it also spills its pre-activation input, because the
  backward pass has to differentiate at that input and it no longer exists.
  Nothing throws and nothing produces NaN — the model just trains towards the
  wrong place. It is why cuBLASLt exposes `GELU_AUX_BIAS` and not merely
  `GELU_BIAS`.
- **`SushiBLAS::Device::Profile` — the device described as facts a kernel can
  dispatch on** (`include/SushiBLAS/engine/device_policy.hpp`, included from
  `engine/dispatch.hpp`). Probed once per device and cached. The field that
  carries the weight is `has_dedicated_local_memory`, because `local_mem_type`
  alone cannot answer it: an OpenCL CPU device reports `local_mem_type::local`
  and a plausible `local_mem_size` while backing both with the same cache the
  global pointer already reads through, so a tiled kernel there copies the data
  twice and pays a real thread rendezvous for every `group_barrier`. Also
  supplies `plan_1d`, a grid-stride geometry capped at what the machine holds
  resident rather than one work-item per element, and `vector_width`, which
  returns 1 on CPU deliberately — the OpenCL CPU backend already vectorises
  across work-items and a hand-written `sycl::vec` competes with it for the
  same lanes. Knows nothing about GEMM or elementwise on purpose: only a kernel
  knows what its accumulator costs in registers.
- **AOT device targeting (`SB_SYCL_TARGETS`).** Default builds emit SPIR-V and
  let the runtime JIT it, which reaches an NVIDIA GPU not at all — NVIDIA's
  OpenCL driver implements no `clCreateProgramWithIL`, so it cannot ingest
  SPIR-V, and the resulting "it fell back to the CPU" is not a fallback but the
  only device the binary could ever see. `SB_SYCL_TARGETS` threads
  `-fsycl-targets=` through compile *and* link as PUBLIC, which a STATIC
  library needs: device code is not finalised until the consumer's executable
  links, so a consumer inheriting `-fsycl` without the target list produces a
  binary with no CUDA image and reports it only at run time. `SB_SYCL_DEVICE_FLAGS`
  is the escape hatch for backend switches such as `-Xcuda-ptxas;-v`, which
  reports the register spills that cost a blocked GEMM most of its blocking.
  Configure prints the active targets either way.
- **`sb build -D VAR=VALUE`**, repeatable, validated, appended after every
  default the CLI sets so an explicit `-D` is never outranked. Added because
  without it `SB_SYCL_TARGETS` was unreachable: this project's contributing
  rules route every build through `sb`, and `sb build` had no way to pass a
  cache variable, so the option above would have been unusable by the only
  sanctioned path.
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
- **GEMM's layout/transpose handling moved off the inner loop and onto the
  host.** The four `(layout, transA/transB)` combinations were branched on
  per element, inside the tile-load helpers, in the innermost loop. They are
  only ever two strides, so `Detail::strides_for` resolves them once before the
  launch and the element access becomes `p[row * s.row + col * s.col]` — exact
  for all four cases, branch gone, and with zero extra template instantiations
  because the strides are runtime values rather than template parameters. The
  same pair now describes C, and SYRK addresses its operands through it too.
- **GEMM's register block is derived from `sizeof(T)` instead of fixed at 4x4.**
  The old comment said the halving from portBLAS's default `Tile<8,8,8,8>` was
  to bound private memory "across all five dispatched element types, including
  `complex<double>`" — which penalised `float`, the training path, for the sake
  of a type almost nobody calls. `Detail::register_block_dim` now picks the
  largest square block fitting a byte budget, so 4-byte and narrower types get
  the full 8x8 (256 B of accumulator, exactly what upstream's default spends)
  and the 8- and 16-byte types keep 4x4. `constexpr` arithmetic, not a list of
  explicit specialisations. The no-local variant uses a quarter of that budget
  and caps at 4x4, because on a CPU backend the runtime vectorises *across*
  work-items and every accumulator scalar is really a whole vector register —
  a 4x4 float block is already the entire AVX2 register file.
- **GEMM's A tile is bank-padded and its register block owns a strided lattice
  rather than a contiguous square.** The A read was
  `tileA[(local_row * item_rows + ii) * tile_k + kk]`, and with the widened
  block that stride is `8 * 16 = 128` floats — `128 mod 32 == 0`, so every lane
  in a sub-group differing only in `local_row` hits one bank. The tile's
  leading dimension is now `tile_k + 1 = 17`; `gcd(17, 32) == 1`, so the four
  distinct `local_row` values in a 32-lane sub-group take four distinct banks.
  Ownership changed with it: a work-item now owns tile rows
  `local_row + ii*wg_rows` and columns `local_col + jj*wg_cols`, so
  neighbouring work-items touch neighbouring elements. That is what keeps the B
  read conflict-free without padding (contiguous ownership would have spaced
  those lanes `item_cols == 8` floats apart, banks `{0,8,16,24}` twice) and
  what makes the global C store coalesced. The full arithmetic is written out
  at the read site rather than left as an unexplained `+ 1`.
- **GEMM uses 32-bit integers for tile-local indices**, keeping 64-bit only for
  global coordinates, strides and batch offsets, which can genuinely exceed
  2^31. 64-bit integer arithmetic is markedly slower on a GPU and a tile index
  never comes close to needing it.
- **Only GEMM's output pointer is `__restrict__`-qualified.** BLAS forbids C
  from overlapping A or B, so that is sound. `a` and `b` are deliberately left
  unqualified and a comment says why: a caller computing `C = A * A` passes the
  same pointer for both, which is legal, and `__restrict__` on both would make
  it undefined behaviour.
- **SYRK owns its tile geometry** (`Detail::SyrkTileConfig`) instead of sharing
  GEMM's, so retuning one cannot silently retune the other. SYRK has not been
  given the local/no-local split; it still always takes the local-memory shape,
  which is a named follow-up rather than a regression.
- **Double buffering was considered and deliberately left out.** It would elide
  one of the two barriers per K-step, but the win over a correctly bank-padded
  single-buffered kernel is modest and there is no GPU on this machine to
  measure it on — the only reachable device is an OpenCL CPU, which takes the
  no-local variant and has no barriers to elide. Restructuring a barrier
  sequence that cannot be verified is the wrong risk; the reasoning is recorded
  in the kernel rather than the omission left to look like an oversight.
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
- **Column-major GEMM and SYRK read A and B with row-major indexing.** Both the
  reference and the SYCL kernel addressed a COLUMN_MAJOR operand as
  `a[row*lda + col]` where the standard says `a[row + col*lda]` — the two arms
  of the `layout` test were inverted, which made `layout` a complete no-op for
  A and B (only C respected it). Two consequences. A column-major GEMM computed
  the wrong product. And because the driver correctly passes the column-major
  leading dimension (`lda = m` for an `m x k` A), the wrong index runs off the
  end of the allocation whenever `m > k`: the maximum touched index is
  `(m-1)*m + k-1` against an `m*k` buffer. `Unit_GEMMTest.RectangularColMajor`
  (m=8, k=4) was reading 27 floats past the end of A on every run.

  It survived because the reference kernel — the differential oracle — carried
  the identical defect, so `FORCE_SYCL` and `FORCE_REFERENCE` agreed with each
  other while both disagreed with BLAS. `Reference::gemv`, which was written
  correctly (`ROW_MAJOR ? a[r*lda + c] : a[r + c*lda]`), is what establishes
  that this was a level-3 defect and not a house convention.

  Both kernels now go through one two-stride rule, and the fix is pinned by
  hand-computed expectations rather than by another differential test:
  `Regression_Sentinel.GemmColumnMajorOperandsUseColumnStrides` (both execution
  modes, `m > k`), three `Unit_GEMMKernelTest` absolute pins, and
  `StrideTableMatchesTheStandardIdentity`, which asserts the property that
  would have caught it when it was written — transposing an operand and
  flipping its layout are the same operation, so ROW_MAJOR+transposed and
  COLUMN_MAJOR+untransposed must address identically.
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

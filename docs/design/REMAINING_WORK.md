# Remaining work

Next id: RW-108

## Decisions waiting for the owner

### RW-1 Decide the names of `spmv` and of "packed"
- State: decision
- Done when: the owner has ruled whether the CSR `spmv` or the packed `spmv` is renamed, and
  what "packed" names besides BLAS packed storage
- Source: `docs/agent/2026_10_07_V010/PLAN.md`

`PLAN_EXT`, section 5, point 3.

### RW-2 Delete `blas/sparse.hpp` or move it under `experimental/`
- State: decision
- Done when: `modules/operations/blas/include/SushiBLAS/blas/sparse.hpp` is gone or under
  `experimental/`

### RW-3 Remove `core/export.hpp` and the `<numbers>` include
- State: decision
- Done when: neither is in the tree (C13)
- Where: `modules/operations/nonlinear/source/gelu.cpp`

`gelu.cpp` includes a C++20 header in a C++17 library. A strict C++17 compiler refuses the file.
Cause: it includes `<numbers>`.

### RW-4 Split `IO` into one reader and writer per format
- State: decision
- Done when: the owner has approved the boundary, and each format has a reader and a writer over
  a shared host-copy helper, with a load that returns a status and no `Engine&` (C4)
- Where: `modules/operations/io/include/SushiBLAS/io/io.hpp`, `modules/operations/io/source/io.cpp`

Every `IO` call runs the whole recorded graph first. A print or a save waits for every recorded
task. Cause: `IO` takes an `Engine&`, the one class outside the recorder shape.

### RW-5 Decide whether ELU accepts complex input
- State: decision
- Done when: ELU takes complex input, or the owner has ruled it real-only

### RW-6 Decide the backend of the quarantined FFT and LAPACK declarations
- State: decision
- Done when: the declarations under `modules/operations/experimental/include/SushiBLAS/experimental/`
  have a chosen backend and an implementation

### RW-7 Decide whether a green CI on the tagged commit is a release condition
- State: decision
- Done when: the owner has ruled, and the build jobs build against the tagged runtime

### RW-8 Tag v0.1.0
- State: decision
- Done when: `## Unreleased` is renamed to the release heading `check_version.py` holds to
  `project()`, and v0.1.0 is tagged (L11)

### RW-9 Rule how a function-local `constexpr` is spelt
- State: decision
- Done when: the about 124 local values not spelt `K_UPPER_SNAKE` are renamed, or the owner has
  ruled that a local keeps `snake_case`

`tools/naming/sushiblas_0_1_0.tsv` lists what the rename of 0.1.0 changed.

### RW-107 Decide whether GEMM tiles are chosen per device
- State: decision
- Done when: the owner has ruled whether the plan of the parametric GEMM tiles starts
- Source: `docs/agent/2026_10_19_PARAMETRIC_GEMM_TILES/SPEC.md`

The tiled GEMM runs one tile shape on every device, measured only on the RTX 3080 Ti. The spec
chooses a shape per device from a few compiled ones, by a measured table, an analytic model and
hard limits; `PLAN.md` beside it holds eight tasks. The owner decided its section 7 and holds
the start.

## Verification

### RW-10 Run the benchmark programs other than `benchmark_band_packed`
- State: unmeasured
- Done when: each benchmark program has a recorded run on the module tree

### RW-11 Record why CTest skips tests on the CUDA build
- State: unmeasured
- Done when: each skipped test has its reason written down

### RW-12 Measure the parity allowances of the random reference tests
- State: unmeasured
- Done when: each allowance of `test_random_reference.cpp` that proves too tight is raised with
  the measured value beside it, and `LargeLambdaPoissonGivesTheSameCounts` holds on a second
  device

### RW-13 Check the rotation bricks and `Core::IndexBase` against their sources
- State: unmeasured
- Done when: `givens.hpp` and `modified_givens.hpp` match `drotg.f90`, `zrotg.f90` and
  `drotmg.f` of a LAPACK checkout, and the two values of `Core::IndexBase` match the Sparse BLAS
  header

### RW-14 See each test added since 2026-10-05 fail on the old code
- State: unmeasured
- Done when: every validation, parity and edge-case test added since then fails on the old code
  and passes on the new

### RW-15 See the record-time refusal of a device that cannot host the SYRK tile
- State: unmeasured
- Done when: `require_local_tile` has thrown on a device that cannot host the `syrk` or `syr2k`
  tile

### RW-16 Configure and build as a consumer does
- State: unmeasured
- Done when: the tree configures with `SB_BUILD_TESTS` and `SB_BUILD_EXAMPLES` off, and
  `tests/package/` builds against an installed tree

### RW-17 Compile the kernels for a second target
- State: unmeasured
- Done when: a second ahead-of-time target and a SPIR-V device have compiled the kernels,
  `Kernels::multiply`, `divide` and `conjugate` among them

### RW-18 Settle the open items each design lists
- State: open
- Done when: each open or unverified item of the designs is settled or moved here
- Source: `docs/agent/2026_10_07_V010/SPEC.md`

`DESIGN_RECORDER` section 15, `DESIGN_SUPPORT_MEMORY` 18, `DESIGN_BLAS` 12, `DESIGN_BLAS_L3` 13,
`DESIGN_BLAS_BAND_PACKED` 14, `DESIGN_GEMM_CPU` 10.3 and 13, `DESIGN_SPARSE` 18, `DESIGN_YOLO26`
13, and section 7 of `REPORT.md` in the same folder.


### RW-101 State and test that tensors may be created from several threads
- State: open
- Done when: `TensorFactory::create` and `Engine::create_tensor` state it, and a test creates and
  releases tensors from eight threads

Section 4 of SushiAI's `docs/design/GRAPH_CAPABILITIES.md`, B7.

### RW-102 State and test host access to a Shared tensor between executions
- State: open
- Done when: a plan compiled once copies a source the host rewrites between three executions, and
  the destination matches each time

Section 4 of SushiAI's `docs/design/GRAPH_CAPABILITIES.md`, B8.

## Benchmarks and tuning

### RW-19 Tune the blocking of the packed GEMM on the CPU
- State: unmeasured
- Done when: `benchmark_gemm_blocking` has run on the CPU device and the blocking follows the
  procedure of `DESIGN_GEMM_CPU`, section 10.4, answering the eight unknowns of section 10.3

### RW-20 Settle the solve block widths
- State: unmeasured
- Done when: `benchmark_triangular` has set the two widths, 256 and 128 today, and whether a small
  TRMV needs a serial arm (`DESIGN_BLAS_L3`, sections 8 and 13)

### RW-21 Decide from the runs whether `ScratchShelf` gets size classes above 4 KiB
- State: blocked
- Blocked by: RW-19, RW-20
- Done when: the runs show whether a recorded `trmv`, a packed GEMM and a scattering sparse
  product need classes above 4 KiB, and the shelf follows
- Where: `modules/engine/engine/include/SushiBLAS/engine/scratch_shelf.hpp`

A lease above 4 KiB allocates device memory at every record. A recorded `trmv`, a packed GEMM and a
transposed or one-triangle sparse product allocate and free at each record. Cause: the scratch pool
has no size class above 4 KiB.

### RW-22 Run the benchmarks on a device other than the CPU
- State: unmeasured
- Done when: the benchmark figures exist for a GPU; those of 2026-10-07 are from one Ryzen 5
  7600X on the tree before the break

## GPU validation

### RW-23 Confirm the device profile on the card
- State: unmeasured
- Done when: `Device::Profile` reports `has_dedicated_local_memory = true` on the RTX 3080 Ti
- Source: `docs/design/GPU_VALIDATION.md`

### RW-24 Measure the tiled GEMM's bank conflicts and register spill
- State: unmeasured
- Done when: the bank conflicts and the spill of the 8x8 block are measured
- Source: `docs/design/GPU_VALIDATION.md`

### RW-25 Run the 128-bit vector path and the grid-stride launch
- State: unmeasured
- Done when: both shapes of `launch_1d.hpp` have run on a GPU
- Source: `docs/design/GPU_VALIDATION.md`

### RW-26 Measure where an axis reduction switches to one work-group per segment
- State: unmeasured
- Done when: the switch point is measured on a GPU
- Source: `docs/design/GPU_VALIDATION.md`

### RW-27 Run SushiAI's determinism regressions on the GPU
- State: unmeasured
- Done when: the regressions have passed on the card
- Source: `docs/design/GPU_VALIDATION.md`

### RW-28 Run the group-per-row GEMV and SYMV paths
- State: unmeasured
- Done when: both paths have run on a device that chooses them; on the CPU only a 4 x 16384 GEMV
  reaches them and SYMV never does
- Source: `docs/design/GPU_VALIDATION.md`

### RW-29 Confirm the device memory alignment on a run
- State: unmeasured
- Done when: `Storage` is seen to receive 128-byte aligned memory on CUDA, and on Level Zero when
  a machine with it is at hand
- Source: `docs/design/GPU_VALIDATION.md`

### RW-30 Measure the weight gradient of `window_dot`
- State: unmeasured
- Done when: the cost of its two 64-bit divisions per term is measured
- Source: `docs/design/GPU_VALIDATION.md`

## Kernels and reductions

### RW-31 Give SYRK an epilogue and a variant without local memory
- State: open
- Done when: SYRK runs on a device that cannot host its tile; `syr2k` and `her2k` stay on the
  local-memory shape on a CPU

### RW-32 Share GEMM's local-memory padding and store interleave with SYRK
- State: open
- Done when: both are helpers outside `gemm.hpp` that SYRK uses

### RW-33 Add `trmm`
- State: open
- Done when: `trmm` exists, built from `TrmvTerm` and `TriangleRow`

### RW-34 Add a `rot` that takes c and s as tensors
- State: open
- Done when: `rotg` chains into `rot` inside one graph

### RW-35 Coalesce the reads of TRSM's diagonal-block solve
- State: open
- Done when: the diagonal-block solve of `trsm.hpp` reads coalesced; lines 126 to 163 map one
  work-item per vector today

### RW-36 Close the gap between the SYCL `tpsv` and the dense `trsv`
- State: open
- Done when: a 128-row block of `tpsv` costs what `trsv` does; 0.30 ms against 0.19 ms up to
  order 1024 and 0.42 ms against 0.20 ms from 2048 (FLOAT32, RTX 3080 Ti)

### RW-37 Measure where a narrow-band `tbsv` is cheaper on the host
- State: unmeasured
- Done when: the crossing point against a GPU is measured

### RW-38 Measure the two bounds of `plan_depth_split` on a second GPU
- State: unmeasured
- Done when: both bounds, set from one RTX 3080 Ti, are measured on another card

### RW-39 Give the `gemv` with few strided rows a benchmark row
- State: open
- Done when: the benchmark measures it

### RW-40 Merge the mirrored symmetric reads into one brick
- State: open
- Done when: `symv`, `SymmetricAccess` and `symmetric_product` share one text, and the dense
  `symv`, `trmv` and one-vector solve run on the structured algorithm bricks

### RW-41 Move the reference rank updates onto `ReferenceAccumulator<T>`
- State: open
- Done when: they accumulate through it and the parity tests have a complex truth

### RW-42 Add an implicit `BLASScalar` from `std::complex<float>`, and `rotmg` for `HALF`
- State: open
- Done when: both exist with tests

### RW-43 Move the complex operators onto `safe_mul` and `safe_div`
- State: open
- Done when: the elementwise and nonlinear kernels use them for complex values

### RW-44 Bring the batch feed into the compiled plan
- State: blocked
- Blocked by: sushiruntime RW-59
- Done when: a batch step costs no extra dispatch round

### RW-45 Batch level 1 and level 2 operations over tensors above rank 1
- State: open
- Done when: each level 1 and level 2 operation takes a batched tensor

### RW-46 Record a plan once
- State: open
- Done when: `Engine::execute` keeps the compiled graph instead of discarding it on every call

### RW-47 Fuse kernels as a pass over the recorded graph
- State: open
- Done when: the pass exists and is measured

### RW-48 Write a parallel `shuffle`, then freeze the permutation a seed gives
- State: open
- Done when: `shuffle` runs in parallel and a test pins the permutation of a seed
- Where: `modules/operations/random/source/shuffle.cpp`

The permutation `shuffle` draws for a seed is not frozen. A seed's permutation may change between
versions. Cause: no test pins it, and a parallel shuffle would change it.

### RW-49 Design what SushiAI's gather-based loss asks for
- State: open
- Done when: broadcasting binaries, a concatenation along a non-channel axis and a row gather
  with a gather-form backward are designed, with a general group count for `window_dot` and the
  softmax pair (`DESIGN_YOLO26`, section 14, point 10)
- Source: `docs/agent/2026_10_07_V010/SPEC.md`

### RW-50 Round host `HALF` in the subnormal range
- State: open
- Done when: the host conversion rounds below 2^-13 as a device does

### RW-93 Fill a random tensor from a stream position read when the task runs
- State: open
- Done when: `PositionedRandomOps` fills from a position held in a tensor operand, and a plan
  replayed twice with the position rewritten between runs draws two different fills

Section 4 of SushiAI's `docs/design/GRAPH_CAPABILITIES.md`, B1. SushiAI's dropout waits on it
(`sushiai RW-94`).

### RW-94 Add an index family: rows by index and one element per row
- State: open
- Done when: `IndexOps` gathers rows by an INT32 or INT64 index with a defined result out of range
  and a gather-form backward, with tests

Section 4 of SushiAI's `docs/design/GRAPH_CAPABILITIES.md`, B2, which also states the two benchmarks
it lands with. RW-49 designed part of it.

### RW-95 Add a broadcast seam, one value per row or per column under any functor
- State: open
- Done when: `Dispatch::execute_broadcast` and `BroadcastOps` exist with tests, and no existing
  plan records different tasks

Section 4 of SushiAI's `docs/design/GRAPH_CAPABILITIES.md`, B3.

### RW-96 Add bilinear upsampling and its gradient
- State: open
- Done when: `upsample_bilinear` and its backward exist in both kernels, with tests

Section 4 of SushiAI's `docs/design/GRAPH_CAPABILITIES.md`, B4.

### RW-97 Add the index of the maximum along one axis
- State: open
- Done when: `argmax_along` folds through `Reduce::RunningArgMax`, the lowest index winning a tie,
  with tests

Section 4 of SushiAI's `docs/design/GRAPH_CAPABILITIES.md`, B5.

### RW-98 Add GELU by the error function
- State: open
- Done when: `gelu_erf` and `gelu_erf_backward` exist beside `gelu`, whose formula stays, with tests

Section 4 of SushiAI's `docs/design/GRAPH_CAPABILITIES.md`, B6.

### RW-99 Add a mask of the k largest of each row
- State: open
- Done when: `ReductionOps::top_k_mask_along` exists with tests, the lowest index winning a tie

Section 1 of SushiAI's `docs/design/GRAPH_CAPABILITIES.md`, B9.

### RW-100 Add the index of the rows a mask selects
- State: blocked
- Blocked by: RW-94
- Done when: `IndexOps::select_rows` exists with tests

Section 1 of SushiAI's `docs/design/GRAPH_CAPABILITIES.md`, B10.

### RW-103 Add a bfloat16 `DataType`
- State: open
- Done when: `Core::DataType` holds bfloat16 and the operations that take HALF take it, with tests

SushiAI's mixed precision waits on it (`sushiai RW-109`).

## Kernel audit

### RW-51 Land the correctness fixes of the kernel audit
- State: open
- Done when: the six correctness items each land with the test the audit names
- Source: `docs/agent/2026_10_19_KERNEL_AUDIT/REPORT.md`

HALF normals in `normal_block`; SYRK tile counts through `Detail::group_count` and empty
batches; a 4x4x4 GEMM with batch 65 536 and a GER with m = 70 000 on the card; the event of an empty spatial input; `sycl::vec` loads in place of
`reinterpret_cast`; the shadowed `k` of `gemv.hpp`.

### RW-53 Land the optimisations of the kernel audit
- State: open
- Done when: each optimisation is its own `perf` commit with its kernel's tests green
- Source: `docs/agent/2026_10_19_KERNEL_AUDIT/PLAN.md`

GEMM: the staging order from each operand's contiguous axis; full tiles without bounds checks
and with 32-bit offsets; 128-bit staged loads; a double-buffered K loop; `K_TILE_K` and the
`GemmLocalTile` shape; then the HALF `joint_matrix` path, tasks 6 to 10 of the audit's
`PLAN.md`. Elsewhere: the TRSM trailing update through `gemm_accessed`; each B panel of the CPU
GEMM packed once per depth block; the CPU register block per ISA, on an AVX-512 host; unit-stride
level 1 through `Launch::map` and `zip_inplace`; GER, SYR, SYR2, SPR and SPR2 on the contiguous
axis and the stored triangle only; `row_reduction` ending in a sub-group shuffle; ‖vᵢ‖² once per
step of `orthonormalize`; a 3-D or 32-bit decode in the spatial kernels; a tiled transpose in
`layout_convert`; the transposed `spmv` and `spsymv` scatter past 256 work-items; and
`Device::profile_for` without a mutex and a scan.

### RW-92 Measure the optimisations of RW-53 in one session
- State: blocked
- Blocked by: RW-53
- Done when: the tree of commit `f5173a7` and the tree after RW-53 have run the benchmarks the
  optimisations touch, in at most an hour, and each change that moves no row is reverted
- Source: `docs/agent/2026_10_19_KERNEL_AUDIT/PLAN.md`

### RW-54 Land the refactors of the kernel audit
- State: open
- Done when: each refactor item lands with its tests unchanged in result
- Source: `docs/agent/2026_10_19_KERNEL_AUDIT/REPORT.md`

One tiled-product brick with a Store parameter for `gemm_local_tiled`, `syrk` and `syr2k`; A
through `OperandAccess` in ger, syr and syr2; one image-index brick in the spatial kernels;
`partial_capacity` and `K_LEVEL1_PARTIAL_BYTES` into reduction, dropping `blas_kernels` from the
sparse kernels' `DEPENDS`; every dot product through `multiply_add`, tasks 1 to 5 of the audit's
`PLAN.md`; `[[nodiscard]]`, namespace closers, five Doxygen blocks, `deps` last, `is_upper()` in
`tbsv` and the include grouping of `gemm.hpp`; the 211 kernel lines over 100 columns; one
`no_local` warning per device.

### RW-106 Find why 13 bit-equality tests fail on the Ryzen 5 7600X CPU device
- State: open
- Done when: the cause is named and the 13 tests pass on `opencl:cpu` of that machine, or carry
  the `quarantined` label with an entry in `KNOWN_ISSUES.md`
- Source: `docs/agent/2026_10_19_KERNEL_AUDIT/REPORT.md`

`Medium_BlasBand.AFullBandEqualsTheDenseSibling` and `Medium_BlasPacked.APackedTriangleEqualsTheDenseSibling`
in the real and complex dtypes, and three `Medium_TriangularParallel` tests in the complex
dtypes, differ by one ulp. They pass on CI's AMD EPYC 7763 at commit `067eb86` and on `cuda:0`,
and still fail here with the level 1 launch shapes, the `profile_for` fast path and the
row-reduction shuffle all taken out. A build of commit `f3e6382` on this machine would show
whether they failed before the kernel audit.

## Sparse

### RW-55 Parallelise `sptrsv` by level scheduling
- State: open
- Done when: an analysis routine, a type for its result and a solve that takes it exist

### RW-56 Add `sptrsm`, a symmetric `spmm` and a one-triangle builder entry
- State: open
- Done when: the three exist with tests

### RW-57 Assemble on the device and convert a dense tensor in a recording
- State: open
- Done when: both run as recorded tasks

### RW-58 Parallelise long rows and the transposed `spmm`
- State: open
- Done when: a long row gets a work-group and the transposed `spmm` is parallel inside a column

### RW-59 Add CSC and COO operands, sparse-sparse products and mixed index widths
- State: open
- Done when: each exists with tests

### RW-60 Add a recordable structure check and a one-device check
- State: open
- Done when: a caller can record a structure check, and one call refuses operands on two devices

### RW-61 Call one `SparseOps` routine from the package test
- State: open
- Done when: `tests/package/main.cpp` calls one, so the sparse headers are shown to install
- Where: `tests/package/main.cpp`

The package test does not show that the sparse headers install. A broken sparse install passes the
package test. Cause: it calls no `SparseOps` routine.

## IO

### RW-62 Save through the task graph
- State: open
- Done when: `save` is asynchronous

### RW-63 Route `Shared` storage through the queue for a print and a save
- State: open
- Done when: a print and a save of `Shared` storage read through the queue, not the host pointer

### RW-64 Print a rank-0 tensor as its value
- State: open
- Done when: a rank-0 tensor prints its value and `to_vector` reads without zero-filling first

### RW-65 Test the four CLI services
- State: open
- Done when: each service under `cli/sushiblas/services/` has tests (C18)

## Code shape

### RW-104 Carry out SushiBLAS's share of layered execution
- State: open
- Done when: tasks B1 to B13 of `docs/design/LAYERED_EXECUTION.md` are closed
- Source: `docs/design/LAYERED_EXECUTION.md`

B3 closes RW-4; B13 closes KI-26 with RW-89. The programme's entry is `sushiai RW-130`; B5 waits on
its A0, B10 on `sushiruntime RW-65`.

### RW-66 Reword the old bare names in comments and messages
- State: open
- Done when: no comment or failure message spells an enumerator or a constant by its old bare
  name; `kLarge` remains in two tests

### RW-67 Move the recorder's identity counter into a source file
- State: open
- Done when: two binaries that link the static library cannot share an identity

### RW-68 Route every runtime dtype switch through one table
- State: open
- Done when: `constant.cpp`, `reductions/dispatch.hpp` and the rest use one table (C1)

### RW-69 Give reduction kernels one home
- State: open
- Done when: every reduction kernel lives in one module (C9)

### RW-70 Split the level 1 kernel headers per routine
- State: open
- Done when: each level 1 routine has its own header (C12)

### RW-71 Split `test_regression.cpp` per defect
- State: open
- Done when: each defect has its own test file and every test states its tolerance (C17)

### RW-72 Widen the reference tests of levels 1 and 2 and TRSM
- State: open
- Done when: they cover column-major and other dtypes, and the remaining single-case smoke tests
  carry the edge cases the five edited ones carry

### RW-73 Give the eight older spatial operations an overlap check
- State: open
- Done when: each refuses a written tensor that overlaps a read one, as the `window_dot` triple
  does

### RW-74 Bring the record-time errors to the error table of `cpp-code-style`
- State: open
- Done when: an operand that breaks a rule stops the process through `SB_VERIFY` with a
  replaceable handler, and `UnsupportedExecution` is returned by the next `execute()`

Its own plan, after 0.2.0. It touches the helpers of `operand_check.hpp`, `SB_THROW_IF`, 339
`EXPECT_THROW` lines in 50 test files, two examples and the catch ladder of SushiAI's
`seam.cpp`.

### RW-75 Group the includes of every source file
- State: open
- Done when: every file groups its includes as the contributing guide states

The runtime carries the same order and groups its own in step (`sushiruntime RW-61`).

### RW-76 Include each header of a header-only module first and alone in a test
- State: open
- Done when: a header that forgot an include fails in its own test
- Where: `modules/foundation/`, `modules/kernels/`, `modules/engine/operands/`,
  `modules/engine/functor_map/`

No test includes each header of a header-only module first and alone. A header that forgot an
include builds until a source of another module includes it first. Cause: no such test exists.

### RW-77 Spell the remote in `sushi-module.toml` as GitHub does
- State: open
- Done when: `sushi-module.toml` names `SushiSystems/SushiBLAS`
- Where: `sushi-module.toml`

`sushi-module.toml` spells the remote in lower case. The file differs from `SushiSystems/SushiBLAS`.
Cause: the remote was written by hand.

### RW-78 Show the dependency rule on a kernel that drops its events
- State: open
- Done when: a test shows a kernel that accepts `deps` and never uses it, and the engine seam
  check reports it
- Where: `tools/layering/check_engine_seams.py`,
  `tests/library/umbrella/test_SushiBLAS/test_task_dependencies.cpp`

The dependency rule reads text. No test shows a kernel that drops its dependency events. Cause:
`rule_task_dependencies` passes a kernel that accepts `deps` and never uses it, and
`ARecordedTaskComputesItsResult` runs one operation of each seam alone.

## Command line

### RW-79 Take build, test, run and clean from the shared registrar
- State: blocked
- Blocked by: sushicore RW-15
- Done when: the four commands come from sushicore (command line C1, C12)

### RW-80 Rename `--sort` to `--pick` with a hidden alias
- State: open
- Done when: `sb run --pick` works and `--sort` stays as a hidden alias (C4)

### RW-82 Link the colour, theme and exit code page from `cli/README.md`
- State: blocked
- Blocked by: sushicore RW-40
- Done when: `cli/README.md` links the one page sushicore writes (C14)

### RW-83 State the default run target once
- State: open
- Done when: the default run target is written in one place (C18)

### RW-84 Let `sb test --package` report a locked DLL
- State: open
- Done when: a locked DLL beside the consumer ends in a console error, not a traceback

## Continuous integration

### RW-85 Fix the failing jobs and build the benchmarks
- State: open
- Done when: the Windows job finds a SYCL device past its filter, and the nightly Benchmarks and
  Quarantined jobs configure (`cmake/SushiBLASSushiLog.cmake:23`) and build the benchmarks
- Where: `.github/workflows/ci.yml`, `.github/workflows/nightly.yml`

The Windows job, added by commit `3fe1a91`, has never passed. Its runner has the OpenCL ICD
loader that `sb setup` installs and no OpenCL driver, so `TestEnvironment::SetUp` finds no
device, `Runtime::create()` fails, and every test of a binary that uses the environment fails
with it, Small ones included: 350 of 376 on run 39269551240. The job now installs Intel's CPU
runtime for OpenCL through `.github/actions/setup-opencl-cpu/action.yml`; no run has shown it
green yet.

The Windows job and two nightly jobs fail. Those three jobs are red; the Linux, sanitizer,
ThreadSanitizer and SushiAI jobs pass. Cause: no SYCL device passes the Windows job's hardware
filter, and the nightly Benchmarks and Quarantined jobs fail at configure in
`cmake/SushiBLASSushiLog.cmake`.

### RW-86 Bring the Linux job of the small and medium tests under 45 minutes
- State: open
- Done when: the job runs within the pre-merge budget and its timeout drops from 90 minutes
- Where: `.github/workflows/ci.yml`

The Linux job of the small and medium tests runs over its budget. Its timeout is 90 minutes until
the ccache figures are known. Cause: the build took 48 minutes without a compiler cache, against a
pre-merge budget of 45.

### RW-87 Size the medium suites that reach nothing
- State: open
- Done when: each `Medium_` suite whose tests reach no runtime, queue, tensor or file is `Small_`

### RW-88 Run the release workflow once
- State: unmeasured
- Done when: `.github/workflows/release.yml` has run before the tag depends on it

## Release

### RW-89 Pin the runtime
- State: blocked
- Blocked by: sushiruntime RW-3
- Done when: `cmake/SushiBLASConfig.cmake.in` finds SushiRuntime 0.4, `cmake/Runtime.cmake`
  names the same version, a sibling checkout's version is checked, and CI builds on the tag

### RW-90 Keep the experimental module out of the install
- State: open
- Done when: `experimental` is `INTERNAL`, the package test fails on an installed tree that holds
  that folder or a document among its headers, and `sb test --package` passes

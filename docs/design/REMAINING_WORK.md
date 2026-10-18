# Remaining work

The single backlog of this repository. A defect or an unverified claim that is already in the
tree is in [the known issues](../reference/KNOWN_ISSUES.md); this page holds what is to be done.
Finding numbers such as C4 or L11 refer to `docs/agent/2026_10_05_ESTATE_AUDIT/REPORT.md`. A name
such as `DESIGN_BLAS_L3`, `PLAN` or `PLAN_EXT` is a heading of `SPEC.md` or `PLAN.md` in
`docs/agent/2026_10_07_V010/`.

## Full run of the v0.1.0 tree

The whole suite ran on the module tree on 2026-10-09, on the CUDA build with an RTX 3080 Ti:
2 049 tests, 2 014 passed, none failed, 35 skipped.

- Run `sb build --benchmarks` and `sb test` on the OpenCL CPU device, where this tree has never
  run, and fix what it finds.
- Run the seven benchmark programs that have not run; only `benchmark_band_packed` has.
- Read why CTest lists tests as skipped on the CUDA build and record each reason.
- If the link fails on a `HostKernel` symbol, turn the generic lambda arms of `all.cpp`,
  `any.cpp`, `shuffle.cpp` and `orthogonal.cpp` into named classes shaped like
  `Detail::RandomArm`.
- Raise any parity allowance of `test_random_reference.cpp` that proves too tight, with the
  measured value beside it.
- Compare `givens.hpp` and `modified_givens.hpp` with `drotg.f90`, `zrotg.f90` and `drotmg.f`
  from a LAPACK checkout, and the two values of `Core::IndexBase` with the Sparse BLAS header.
- Confirm that each validation, parity and edge-case test added since 2026-10-05 fails on the
  old code and passes on the new; no new test has been seen failing before its fix.

## Benchmarks and tuning

- Run `benchmark_gemm_blocking` on the CPU device and tune the blocking of the packed GEMM by the
  procedure of `DESIGN_GEMM_CPU`, section 10.4; it also answers the eight unknowns of its
  section 10.3.
- Run `benchmark_triangular` and settle the two solve block widths and whether a small TRMV needs a
  serial arm (`DESIGN_BLAS_L3`, sections 8 and 13).
- Run `benchmark_band_packed` and `benchmark_sparse`; neither has produced a number.
- Decide from those runs whether `ScratchShelf` gets size classes above 4 KiB. A recorded
  `trmv`, a packed GEMM and a scattering sparse product each lease more than the pool holds.
- Run the benchmarks on a device other than the CPU; the numbers of 2026-10-07 are from one
  Ryzen 5 7600X on the tree before the break.

## GPU validation

Two functional runs exist, both on one RTX 3080 Ti. [GPU_VALIDATION.md](GPU_VALIDATION.md)
records them and holds the steps that have not run.

- Confirm `Device::Profile` reports `has_dedicated_local_memory = true` on the card.
- Measure the bank conflicts of the tiled GEMM and the register spill of its 8x8 block.
- Run the 128-bit vector path and the grid-stride launch shape of `launch_1d.hpp`.
- Measure where an axis reduction should switch to one work-group per segment.
- Run the determinism regressions of SushiAI on the GPU.
- Run the group-per-row GEMV and SYMV paths on a device where they are chosen; on the CPU only a
  4 x 16384 GEMV reaches them and SYMV never does.
- Confirm on a run that `Storage` receives 128-byte aligned device memory on CUDA, and on Level
  Zero when a machine with it is at hand.
- Measure the weight gradient of `window_dot`, whose terms each decode a window with two 64-bit
  divisions.

## Kernels and reductions

- Give SYRK an epilogue and a variant without local memory for a device that cannot host its
  tile; `syr2k` and `her2k` stay on the local-memory shape on a CPU.
- Give SYRK GEMM's local-memory padding and store interleave by lifting them out of `gemm.hpp`
  into shared helpers.
- Add `trmm`; `TrmvTerm` and `TriangleRow` are the bricks it starts from.
- Add a `rot` that takes c and s as tensors, so `rotg` chains into it inside one graph.
- Coalesce the right-side reads of TRSM.
- Close the gap between the SYCL `tpsv` and the dense `trsv`: one 128-row block costs 0.30 ms
  against 0.19 ms up to order 1024 and 0.42 ms against 0.20 ms from order 2048 (FLOAT32,
  RTX 3080 Ti); the cost per block does not grow with the order beyond that step.
- Measure where a narrow-band `tbsv`, which runs as one sequential device task, is cheaper on
  the host than on a GPU.
- Measure the two bounds of `plan_depth_split`, fewer LOCAL_TILED work-groups than compute units
  and a K of at least 16 times the width, on a second GPU. Both are set from one RTX 3080 Ti,
  where a 255 x 256 product with a K of 1352 stays on LOCAL_TILED at 250 GFLOP/s.
- Measure the `gemv` with few strided rows: it has a test and no benchmark row.
- Merge the three texts of the mirrored read of a symmetric matrix (`symv`, `SymmetricAccess`,
  `symmetric_product`) into one brick, and express the dense `symv`, `trmv` and one-vector solve
  through the structured algorithm bricks.
- Move the reference rank updates onto `ReferenceAccumulator<T>` and give the parity tests a
  complex truth.
- Add an implicit `BLASScalar` from `std::complex<float>`, and `rotmg` for `HALF`.
- Move the complex operators of the elementwise and nonlinear kernels onto `safe_mul` and
  `safe_div`.
- Confirm whether `BLASOps::nrm2` uses the scaled form that `ReductionOps::nrm2` uses
  (`Reduce::ScaledSumOfSquares`); the plain two-norm overflows above about 1e19 in `FLOAT32`.
- Bring the batch feed into the compiled plan; it costs one extra dispatch round per step and
  needs an operand-address late binding that SushiRuntime does not offer.
- Add batch support for tensors above rank 1 to the level 1 and level 2 BLAS operations.
- Record a plan once: `Engine::execute` compiles and discards the graph on every call.
- Add automatic kernel fusion as a pass over the recorded graph.
- Write a parallel `shuffle` and then freeze the permutation a seed gives.
- Design what SushiAI's gather-based loss asks for: broadcasting binaries, a concatenation along
  a non-channel axis and a row gather with a gather-form backward (`DESIGN_YOLO26`,
  section 14, point 10), with a general group count for `window_dot` and the softmax pair.
- Give host `HALF` a conversion that rounds in the subnormal range.
- Decide whether ELU accepts complex input; it is real-only today.
- Decide the backend of the quarantined FFT and LAPACK declarations
  (`modules/operations/experimental/include/SushiBLAS/experimental/`), then implement them.

## Kernel audit

From the audit of 2026-10-19 (`docs/agent/2026_10_19_KERNEL_AUDIT/REPORT.md`). Each
optimisation lands only with five runs before and five after, release build, same machine, and
is reverted if the median does not move beyond the spread.

Correctness:

- Draw HALF normals from a float uniform with a log argument in (0, 1] and cast once at the end;
  a `test_philox.cpp` case draws 2²⁰ HALF normals and finds every one finite.
- Clamp the SYRK tile counts with `Detail::group_count`; a `test_syrk.cpp` case runs n = 0 and
  batch 0 and finds C untouched.
- Run a 4x4x4 GEMM with batch 65 536 and a GER with m = 70 000 on the CUDA card; if either fails,
  fold the batch into a grid-stride loop and swap the GER dimensions.
- Order the event of an empty spatial input behind its `deps`.
- Replace the `reinterpret_cast` vector access of `packed_micro_kernel.hpp` and `launch_1d.hpp`
  with `sycl::vec::load` and `store`.
- Rename the shadowed `k` of `gemv.hpp`.

Measurement:

- Give `benchmark_gemm` a RowMajor sweep beside the ColumnMajor one.
- Add `benchmark_level2` (gemv in both layouts, ger, symv, syr, trmv), `benchmark_syrk` (syrk and
  syr2k, orders 256 to 4096), `benchmark_spatial` and `benchmark_random`.

Optimisation, each gated on its benchmark:

- Pick the staging order of the GPU GEMM from each operand's contiguous axis.
- Give full GEMM tiles a load path without bounds checks and with 32-bit in-tile offsets.
- Load staged GEMM tiles with 128-bit accesses when the contiguous axis is aligned.
- Double-buffer the GEMM K loop to one barrier per step.
- Tune `K_TILE_K` and the work-group shape of `GemmLocalTile`.
- Give HALF GEMM a `joint_matrix` path, tasks 6 to 10 of
  `docs/agent/2026_10_19_KERNEL_AUDIT/PLAN.md`, after the four GEMM tasks above.
- Route the TRSM trailing update through `gemm_accessed`.
- Pack each B panel of the CPU GEMM once per depth block and share it across row blocks.
- Size the CPU register block per ISA, measured on an AVX-512 host.
- Run unit-stride Level 1 through `Launch::map` and `zip_inplace`.
- Map the fast dimension of GER, SYR, SYR2, SPR and SPR2 to the contiguous axis and launch only
  the stored triangle.
- Finish `row_reduction` with a fixed-order sub-group shuffle.
- Compute ‖vᵢ‖² once per step of `orthonormalize`.
- Replace the int64 index decode of the spatial kernels with a 3-D launch or a 32-bit path when
  the extents fit.
- Give `layout_convert` a local-memory tiled transpose.
- Widen the transposed `spmv` and `spsymv` scatter past 256 work-items on a GPU.
- Cache `Device::profile_for` so a launch does not take a mutex and scan.

Refactor:

- Run `gemm_local_tiled`, `syrk` and `syr2k` on one tiled-product brick with a Store parameter;
  the three tests stay bit-identical.
- Read A through `OperandAccess` in ger, syr and syr2, SYCL and reference.
- Give spatial_kernels one image-index brick for the flat decode and the inverse-window walk.
- Move `partial_capacity` and `K_LEVEL1_PARTIAL_BYTES` into reduction and drop `blas_kernels`
  from the `DEPENDS` of sparse_kernels.
- Accumulate every dot-product kernel through `multiply_add`, tasks 1 to 5 of
  `docs/agent/2026_10_19_KERNEL_AUDIT/PLAN.md`.
- Add `[[nodiscard]]` and the namespace closers in spatial, Doxygen on the five Level 1
  routines, `deps` last in `orthonormalize`, `is_upper()` in `tbsv` and the include grouping of
  `gemm.hpp`; teach `check_source_comments.py` to report a function with no block.
- Wrap the 211 kernel lines over 100 columns.
- Warn once per device, not per call, when GEMM falls back to `no_local`.

## Sparse

- Parallelise `sptrsv` by level scheduling: an analysis routine, a type for its result and a
  solve that takes it.
- Add `sptrsm`, a symmetric `spmm`, and a builder entry that keeps one triangle.
- Assemble on the device, and add a recorded conversion from a dense tensor.
- Give a long row a work-group, and parallelise the transposed `spmm` inside a column.
- Add CSC and COO operands, sparse-sparse products and mixed index widths.
- Add a structure check a caller can record, and a check that the operands of one call live on
  one device.
- Call one `SparseOps` routine from `tests/package/main.cpp`.
- Decide whether the CSR `spmv` or the packed `spmv` is renamed, or both names stay
  (`PLAN_EXT`, section 5, point 3). Owner decision.

## IO

- Split `IO` into one reader and writer per format over a shared host-copy helper, with a load
  that returns a status, and narrow it off `Engine&` in the same change (C4). The boundary needs
  the owner's approval.
- Make `save` asynchronous through the task graph (`modules/operations/io/include/SushiBLAS/io/io.hpp`).
- Route `Shared` storage through the queue for a print and a save, or fix the fault in
  SushiRuntime; today the host pointer is read.
- Print a rank-0 tensor as its value and stop `to_vector` zero-filling before it reads.
- Add tests for the four CLI services (C18).

## Code shape

- Rename the function-local `constexpr` values that are not spelled `K_UPPER_SNAKE`, or rule
  that a local keeps `snake_case`; the naming sweep left 136 of them
  (`tools/naming/sushiblas_0_1_0.tsv` lists what it renamed).
- Reword the comments and the failure messages that still spell an enumerator or a constant
  by its old bare name (`FORCE_REFERENCE`, `kLarge`); the sweep changed qualified names only.
- Delete `blas/sparse.hpp` or move it under `experimental/`. Owner approval.
- Move the recorder's identity counter into a source file.
- Ask SushiRuntime for a public "run after" edge, `Recorder::task` overloads that take a span of
  regions and a small-buffer callable, and a statement of what concurrent `Recorder::task` guarantees
  (`DESIGN_RECORDER`, section 15).
- Route every runtime dtype switch through one table; `constant.cpp` and
  `reductions/dispatch.hpp` come first (C1).
- Add categories, TRACE and FATAL to the logger and send log output to stderr (C3, C6).
- Give reduction kernels one home (C9).
- Split the Level 1 kernel headers into one file per routine (C12).
- Remove `core/export.hpp` and the `<numbers>` include (C13). Owner approval.
- Shorten the two comment blocks `check_source_comments.py` still reports, then drop
  `--rule rule_license_block` from the check list of `AGENTS.md` (C14).
- Split `test_regression.cpp` per defect and state every test tolerance (C17).
- Add column-major and other-dtype cases to the reference tests of BLAS levels 1 and 2 and TRSM,
  and raise the remaining single-case smoke tests to the edge cases the five edited ones carry.
- Give the eight older spatial operations the overlap check the `window_dot` triple has.
- Bring the record-time errors to the error table of the `cpp-code-style` skill, as its own plan
  after 0.2.0: an operand that breaks a rule (`std::invalid_argument` today) stops the process
  through an `SB_VERIFY` with a handler a host can replace, and a refusal of the mode or the device
  (`UnsupportedExecution`) is kept by the recorder and returned by the next `execute()`. It
  touches the three helpers of `operand_check.hpp`, `SB_THROW_IF`, 310 `EXPECT_THROW` lines in 36
  test files, two examples and the catch ladder of SushiAI's `seam.cpp`.
- Group the includes of every source file as the contributing guide states, together with
  SushiRuntime, whose files carry the same ungrouped order, so the two trees stay shaped alike.

## Command line

- Take build, test, run and clean from a shared sushicore registrar (command line C1, C12).
- Rename `--sort` to `--pick` with a hidden alias, in step with the sibling CLIs (C4).
- Settle the panel names, short options and defaults across the CLIs (C5, C6).
- Document colour, theme and exit codes once in sushicore and link it from `cli/README.md` (C14).
- Remove the unused `find_executables` and `select_interactive` re-exports (C17). Owner approval.
- State the default run target once (C18) and choose one Python floor for the CLIs (C22).

## Documentation

- Decide what to do about the code of conduct: four pages of the Contributor Covenant were read
  on 2026-10-09 and none states a licence for its text (`NOTICE.md`, "Adapted text"). Owner.

## Layout, needing the owner

- SushiRuntime, SushiDSP and SushiAI still name their source folder `src/` and are one
  module each; this repository moved to `modules/` on 2026-10-09 (L8).

## Continuous integration

- Run every job of `ci.yml` and `nightly.yml` once and fix what the first runs show.
- Bring the Linux job of the small and medium tests under the 45-minute pre-merge budget.
- Read each `Medium_` suite and rename to `Small_` the ones whose tests reach no runtime, queue, tensor or file.
- Run the release workflow once before the tag depends on it.

## Release

In the order the tag needs them.

- Tag SushiRuntime v0.4.0 with the aligned shared and device allocation and
  `Graph::add_emitter` in it; both were uncommitted in the runtime's working tree on 2026-10-07.
  Who cuts it, and when, is the owner's decision.
- Pin the runtime: `find_dependency(SushiRuntime 0.4)` in `cmake/SushiBLASConfig.cmake.in`, the
  same version in `cmake/Runtime.cmake`, and a version check of a sibling checkout (`PLAN`,
  section 3).
- Install without `modules/operations/experimental/include/SushiBLAS/experimental/` and make the package consumer test fail if
  the installed tree holds that folder or a document among its headers (`PLAN`, section 3),
  then run `sb test --package`.
- Migrate SushiAI onto the views, the enums and `BLASScalar`. The sushiai session does it, once,
  after this tree has built.
- Get CI green on the commit to be tagged, with the build jobs on the tagged runtime.
  Whether that is a release condition is the owner's decision.
- Cut the changelog: rename `## Unreleased` to the release heading, which `check_version.py`
  holds to `project()`.
- Tag v0.1.0 (L11). Owner decision.

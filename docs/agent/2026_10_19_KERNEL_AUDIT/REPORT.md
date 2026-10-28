# Kernel audit

A read-only inspection of every header under `modules/kernels/` (127 files, 15 549 lines): the
`sycl/` and `reference/` paths of blas_kernels, blas_seam_kernels, spatial_kernels,
sparse_kernels, random_kernels and packed_product_harness, with the foundation headers they lean
on (`launch_1d.hpp`, `device_policy.hpp`, `reduce_1d.hpp`, `accumulator.hpp`,
`multiply_add.hpp`, `running_solution.hpp`). Nothing was built, tested or benchmarked. The plan
that follows from it is in [the backlog](../../backlog/README.md), section "Kernel audit".

**[D]** is a defect visible in the code. **[M]** is a performance hypothesis that needs a
before and after benchmark before it is accepted. Every [M] item is unmeasured. The CUDA grid
limit and zero-range defects are inferred from the code and from the seam README's own rule;
the tests in the plan confirm or refute them.

## Context checked

- The tiled GEMM work-group is a fixed 8x8 (`gemm.hpp:113-119`).
- The K loop has no double buffering and two barriers per step (`gemm.hpp:319-366`). The seam
  README calls the omission deliberate because no GPU was available
  (`blas_seam_kernels/README.md:326-332`); the RTX 3080 Ti is that GPU now.
- Every load is bounds-checked through `load_operand` (`gemm.hpp:63-71`, called at 329, 338,
  467-471, 661-663; `syrk.hpp:289,296`; `syr2k.hpp:220-232`).
- No GPU kernel uses sub-groups, vector loads or matrix extensions. The only `sycl::vec` is in
  the CPU micro-kernel (`packed_micro_kernel.hpp:53-109`).
- Of eight benchmark programs only `benchmark_band_packed` has run (`KNOWN_ISSUES.md:15`).
  Level 2, SYRK and SYR2K, spatial and random have no benchmark.

## blas_seam_kernels

### GEMM on the GPU, `sycl/level3/gemm.hpp`

- [M] Staging ignores the operand's contiguous axis. A is staged k-fastest (`:326-329`) and B
  n-fastest (`:335-338`). For ColumnMajor NoTrans both loads stride by `lda` or `ldb` across
  neighbouring work-items: 16 cache lines per 64 floats against 2 when coalesced. The `no_local`
  B loads (`:470`) and the `depth_split` loads (`:661-663`) do the same. `benchmark_gemm` runs
  ColumnMajor only (`benchmark_gemm.cpp:137`). The CPU packer already picks its loop order from
  the access (`packed_panel.hpp:59-72,129`). This is the largest hypothesis of the audit.
- [M] Interior tiles pay two compares and two int64 multiplies per element
  (`operand_access.hpp:50-53`).
- [M] The 8x8 register block holds 64 accumulators and 16 operands per item; spill and occupancy
  are unmeasured. `K_TILE_K = 16` (`:134`) and the group shape are untuned; at 8 448 B of local
  memory per group, registers limit occupancy first.
- [M] Double buffering, 128-bit global loads and `joint_matrix` for HALF are untried.
- [D] The batch sits in nd_range dimension 0 (`:273-276`, `syrk.hpp:245`, `syr2k.hpp:165`). On
  CUDA that is grid z, capped at 65 535 groups, so `batch_size > 65535` should fail to launch.
- [D] `SB_LOG_WARN` fires on every call that falls back to `no_local` (`:791-795`).
- [D] `gemm.hpp:43` includes `reduction/reduce_1d.hpp` only for `Reduce::group_width`; the
  include grouping breaks the style guide (`:37-47`); 51 lines pass 100 columns.

### GEMM on the CPU, packed path

- [M] Each macro-block packs its own A and B panels (`packed_product.hpp:135-145`), so A is
  re-packed `col_blocks` times and B `row_blocks` times.
- [M] The register block is 6 x (64 B / sizeof(W)) (`packed_product_plan.hpp:61-64`): 12 ymm
  on AVX2, 6 of 32 zmm on AVX-512.
- [D] `load_lanes` and `store_lanes` dereference a `reinterpret_cast<sycl::vec*>`
  (`packed_micro_kernel.hpp:60,70`; also `launch_1d.hpp:175,187`), a strict-aliasing breach.
  `sycl::vec::load` and `store` are the defined route.

The reference GEMM, the access bricks and the Givens rotations show no defect.

## blas_kernels

### Level 1

- [M] axpy, copy, scal, swap and rot launch through `Launch::indexed` (`level1.hpp:54-114`) and
  never take the 128-bit path of `Launch::map` and `zip_inplace`.
- [D] Those five and their reference twins carry no Doxygen block, and
  `check_source_comments.py` does not report it.
- When scratch is missing, `reduce_indexed` runs one serial `single_task` over n elements
  (`reduce_1d.hpp:760-773`).

### Level 2

- [D] `gemv.hpp:103` declares a lambda parameter `k` that `:107` shadows with the loop variable.
- [M] GER, SYR, SYR2, SPR and SPR2 run j fastest over `range<2>(m, n)`, so ColumnMajor writes
  stride by `lda`; the symmetric four launch n² items and half return at once.
- [D] Dimension 0 of that `range<2>` is CUDA grid y, capped at 65 535.
- [D] ger, syr, syr2, symv and the reference ger, symv, syr, syr2, syrk hard-code a layout
  ternary instead of `OperandAccess`.
- [M] `row_reduction.hpp:129-137` folds through local memory with log2(width) barriers; a
  fixed-order sub-group shuffle would keep determinism with fewer.
- [D] `tbsv.hpp:66` computes `op_upper` by hand where `tpsv.hpp:66` asks `access.is_upper()`.
- [M] Each block of a one-vector solve submits two launches and runs as one `single_task`
  (`triangular_solve.hpp:250-278`).

### Level 3

- [D] SYRK computes its tile counts without the `group_count` clamp (`syrk.hpp:240-248`), so
  n == 0 or batch 0 builds a zero-sized nd_range, which fails a launch.
- [M] SYRK and SYR2K tiles have no padding: a 4-way bank conflict by the README's arithmetic.
- [D] `gemm_local_tiled`, `syrk` and `syr2k` are three copies of one staged product that differ
  only in the store and the conjugated second factor.
- [D] GEMM fuses its multiply-add; SYRK, SYR2K, GEMV, `row_reduction`, the sparse kernels and
  `window_dot` round twice. SYRK(A) is not bit-equal to GEMM(A, Aᵀ) on its triangle.
- [M] The TRSM trailing update (`trsm.hpp:179-233`) is a rank-width update with no tiling.

## spatial_kernels

- [M] Every SYCL kernel decodes its flat index with 6 to 8 int64 divisions or modulos per
  element, and the backward kernels add one per tap.
- [M] `layout_convert` (`broadcast.hpp:200-210`) writes strided for NHWC to NCHW.
- [D] The SYCL entry points lack `[[nodiscard]]` and the `// namespace` closers.
- [D] On empty input they return `sycl::event()`, which is not ordered behind `deps`.
- [D] 24 functions repeat the stride unpack and the index decode.

Reference and SYCL agree in formula for every spatial kernel.

## sparse_kernels

- [M] Non-transposed spmv and spmm run one work-item per row; transposed spmv and spsymv run at
  most 256 serial work-items (`row_blocks.hpp:50`).
- [D] `sycl/level1.hpp:42` includes the BLAS Level 1 kernel header only for
  `partial_capacity`, which is the only reason `sparse_kernels` depends on `blas_kernels`.

## random_kernels

- [D] HALF normals from the kernel produce infinities. `uniform01_block<sycl::half>` rounds a
  float in [0, 1) to half (`philox.hpp:136`), and every u ≥ 1 − 2⁻¹² rounds to 1.0.
  `normal_block` then takes `log(0)` (`:173`): about one infinity per 4 096 pairs. The
  operations refuse HALF draws (`random/dispatch.hpp:101`), so only a direct kernel call
  reaches it.
- [D] `draw_index` returns `raw % bound` (`philox.hpp:192`), biased below bound/2⁶⁴, while its
  brief says uniform.
- [M] `orthonormalize` submits 2·(count−1)+1 kernels, and each work-group recomputes ‖vᵢ‖².
- [D] Its three functions take `deps` second, where every other kernel takes it last.

## Across the tree

- [D] 211 lines pass 100 columns.
- [M] `Device::profile_for` takes a mutex and scans linearly on every launch
  (`device_policy.hpp:207-221`).

## Build, tests and the one measurement of 2026-10-19

The optimisations of B10 to B26 were written unmeasured and built once, as the owner ruled. Two
builds stopped in their first two minutes on a variable and a type alias that only the NVPTX body
of `gemm_joint_matrix_as` read, so the SPIR-V pass flagged them (`71e202e`, `f4033fa`); the third
built all 489 steps in 29 minutes with no warning.

### Tests

| Run | Result |
| --- | --- |
| `sb test --jobs 4`, default device, the RTX 3080 Ti | 2718 of 2718 passed, 39 skipped by the device profile, 629 s |
| `sb test --jobs 4 --label large`, `cuda:0` | 2 of 2 passed: the HALF matrix field, and `joint_matrix` bit for bit against the local-tiled kernel |
| `sb test --jobs 4`, `opencl:cpu` | 2705 of 2718 passed, 671 s; a rerun of the 13 failed the same way |

The 13 that fail on the CPU device compare two kernels for equal bits, and each pair runs through
code this work did not change: `Medium_BlasBand.AFullBandEqualsTheDenseSibling` (`sbmv` against
`symv`, four dtypes), `Medium_BlasPacked.APackedTriangleEqualsTheDenseSibling` (`spmv` against
`symv`, three dtypes; every failing element is one of the 17 of the output vector, so the rank
updates B20 changed compare equal) and six complex cases of `Medium_TriangularParallel`. The
values differ in the last bit, 2.5764460563659668 against 2.5764462947845459. Whether they failed
on the CPU before this work is not known: the earlier binary is gone, and no CPU run of these
suites is recorded. They are not in `KNOWN_ISSUES.md`.

### Measurement

One session of 32 minutes, release build, 5 runs of the four programs the kept changes touch on
both devices, then one run of every other program. The "before" is the first round of the stopped
baseline, B9, on the tree of `f5173a7`: one run, so it has no spread. Five medium processes ran
on the machine throughout, as the owner set. Figures are the median of the runs in milliseconds.

The programs no change touched set the noise. On the GPU their rows moved by a median of 0.0 %
(triangular), +1.5 % (sparse), -0.2 % (reductions) and +0.4 % (random), single rows by up to
15 %. On the CPU they moved by +5 % to +30 % throughout, the packed GEMM included, which no change
touched: the CPU's before ran under a heavier load than its after, and no CPU row below is read as
a result.

| Row, RTX 3080 Ti | Before | After, median [range] | Change |
| --- | --- | --- | --- |
| HALF GEMM ColumnMajor 1024 | 0.471 | 0.127 [0.125, 0.131] | 3.7x faster |
| HALF GEMM ColumnMajor 2048 | 2.419 | 0.794 [0.791, 0.794] | 3.0x faster |
| HALF GEMM RowMajor 2048 | 3.625 | 0.750 [0.745, 0.764] | 4.8x faster |
| FLOAT32 GEMM ColumnMajor 512 | 0.266 | 0.177 [0.175, 0.196] | 33 % faster |
| FLOAT32 GEMM RowMajor 1024 | 0.783 | 0.482 [0.477, 0.488] | 38 % faster |
| FLOAT32 GEMM ColumnMajor 2048 | 3.358 | 4.600 [4.055, 4.744] | 37 % slower |
| FLOAT32 GEMM RowMajor 2048 | 3.084 | 4.054 [3.635, 4.084] | 31 % slower |
| FLOAT64 GEMM ColumnMajor 2048 | 29.107 | 30.939 [28.366, 31.039] | within the noise |
| `layout_convert` to NCHW, FLOAT32, side 128 | 0.313 | 0.105 | 67 % faster |
| `layout_convert` to NHWC, FLOAT32, side 128 | 0.150 | 0.104 | 23 % faster |
| `ger` ColumnMajor FLOAT32, order 8192 | 6.105 | 0.681 | 9.0x faster |
| `syr` ColumnMajor FLOAT32, order 8192 | 3.457 | 0.595 | 5.8x faster |
| `spr` ColumnMajor FLOAT32, order 8192 | 3.831 | 0.422 | 9.1x faster |
| `syr` RowMajor FLOAT32, order 8192 | 0.379 | 0.606 | 60 % slower |
| `syr2` RowMajor FLOAT32, order 8192 | 0.385 | 0.637 | 66 % slower |
| `spr2` RowMajor FLOAT64, order 8192 | 0.730 | 0.927 | 27 % slower |
| `copy` FLOAT32, 2^24 elements | 0.236 | 0.186 | 20 % faster |
| `axpy`, `scal`, `rot` FLOAT32, 2^24 elements | | | within the noise |

The HALF rows are the `joint_matrix` path, which every square size takes. The FLOAT32 GEMM gains
up to order 1024 and loses at 2048; the measurement cannot say which of B10 to B13 does which,
since they were built together. The rank updates gain five to nine times in ColumnMajor, where the
old two-dimensional range wrote across the leading dimension, and lose in RowMajor, where the old
range already wrote along it and the triangle walk adds a root and its corrections per element.
Under the plan's rule each change that moves no row, or moves rows backwards, is reverted; which
of the FLOAT32 GEMM at 2048 and the RowMajor rank updates that covers is for the owner to rule.

The logs, one per run, and the full comparison tables, 710 rows for the four programs and the
rest per program, were written outside the repository to the session's temporary folder and are
not kept with it; the rows above are the record. `sb run` of the same programs reproduces them.

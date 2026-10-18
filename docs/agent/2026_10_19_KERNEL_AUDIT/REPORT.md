# Kernel audit

A read-only inspection of every header under `modules/kernels/` (127 files, 15 549 lines): the
`sycl/` and `reference/` paths of blas_kernels, blas_seam_kernels, spatial_kernels,
sparse_kernels, random_kernels and packed_product_harness, with the foundation headers they lean
on (`launch_1d.hpp`, `device_policy.hpp`, `reduce_1d.hpp`, `accumulator.hpp`,
`multiply_add.hpp`, `running_solution.hpp`). Nothing was built, tested or benchmarked. The plan
that follows from it is in [the backlog](../../design/REMAINING_WORK.md), section "Kernel audit".

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

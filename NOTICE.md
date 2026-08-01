# NOTICE

This project includes algorithms ported from **portBLAS**
(https://github.com/codeplaysoftware/portBLAS), archived January 2025, whose
lineage continues at the UXL Foundation's **generic-sycl-components**
(https://github.com/uxlfoundation/generic-sycl-components).

```
portBLAS / oneMath SYCL BLAS
Copyright (C) Codeplay Software Limited.
Licensed under the Apache License, Version 2.0.
```

Per SushiBLAS's own convention (P3: "when you do fork, fork on purpose and
own it"), the source below was rewritten from portBLAS's C++ expression-tree
executor framework into SushiBLAS's flat template-function convention. There
is no free rebase from upstream after this point - tuning and maintenance of
these kernels is SushiBLAS's own responsibility from here on.

## Ported files

- **`kernels/sycl/blas/level3/gemm.hpp`**
  Tiling/blocking algorithm ported from
  `src/operations/blas3/gemm_local.hpp` (local-memory-tiled GEMM): the
  work-group cooperatively loads an `A`/`B` tile into local (shared) memory,
  each work-item accumulates an `item_rows x item_cols` block of `C` in
  private registers via an outer-product loop over the K-tile, guarded by a
  barrier between load and compute phases.

  **Deliberate deviations from the upstream source** (first-pass
  simplifications, not correctness concerns - each is a documented follow-up
  opportunity, not a gap):
  - A single fixed tile configuration (`item_rows=4, item_cols=4, wg_rows=8,
    wg_cols=8, tile_k=16`) rather than portBLAS's full `Tile<>` template
    parameterization and per-backend tuning table (`Tile<8,8,8,8>` for large
    matrices, `Tile<2,2,8,8>` for small, separate Intel-GPU configs, etc.).
  - `tile_k` is a fixed constant, not portBLAS's cache-line-width-driven
    `cl_elems` sizing.
  - No vectorized `sycl::vec` packet load/store (`Packetize<>`) - scalar
    element-at-a-time loads throughout.
  - No "internal vs. edge tile" template-bool fast-path split - every tile
    takes the bounds-checked path uniformly.
  - No double buffering - one local-memory buffer, two barriers per
    K-iteration instead of portBLAS's buffer-flip trick that elides one.
  - Native `sycl::nd_range<3>` (batch, row-tile, col-tile) instead of
    portBLAS's manual 1-D flattened index decomposition.
  - `sycl::group_barrier()` instead of the older `nd_item::barrier()` member
    form; no `decorated::legacy` pointer-decoration compiler-compat hedges.

- **`kernels/sycl/blas/level3/syrk.hpp`**
  **No dedicated SYRK tiling kernel exists upstream** - portBLAS's own
  `doc/MissingFeatures.md` lists SYRK as unimplemented. `C =
  alpha*op(A)*op(A)^T + beta*C` is mathematically `GEMM(A, A, transA=trans,
  transB=!trans)` restricted to one triangle of `C`, so this file reuses
  `gemm.hpp`'s tile-loading primitives (`Detail::gemm_load_a/b`,
  `Detail::GemmTileConfig`) with the second GEMM operand set to `A` itself,
  and adds a triangular output-write mask (per-tile early skip when the
  whole work-group block falls outside the requested triangle, plus a
  per-element mask on the final store). This composition pattern mirrors how
  portBLAS's own SYMM (`src/interface/symm_interface.hpp`) reuses its GEMM
  kernel for a different symmetric operation - but the triangular-output
  logic itself is original SushiBLAS work, not a port.

## Not ported

FFT/DFT and LAPACK (`SVD`, `QR`, eigendecomposition, etc.) remain quarantined
per decision D1 - see `include/SushiBLAS/experimental/`. The backend
strategy for those is a named follow-up, not solved here.

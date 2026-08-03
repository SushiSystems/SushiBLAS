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

- **`include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`**
  Tiling/blocking algorithms ported from `src/operations/blas3/gemm_local.hpp`
  and `src/operations/blas3/gemm_no_local.hpp`, and the reason upstream ships
  both. `Detail::gemm_local_tiled` is the local-memory form: the work-group
  cooperatively loads an `A`/`B` tile into local (shared) memory, each
  work-item accumulates an `item_rows x item_cols` block of `C` in private
  registers via an outer-product loop over the K-tile, guarded by a barrier
  between load and compute phases. `Detail::gemm_no_local` is the generic form:
  the same register blocking with the operands read straight from global
  memory, no `local_accessor` and no barrier, which is what a CPU backend wants
  because its "local memory" is the same cache the global pointer already reads
  through. Which one runs is decided at launch from a `Device::Profile`.

  **Deliberate deviations from the upstream source** (simplifications, not
  correctness concerns - each is a documented follow-up opportunity, not a
  gap):
  - No per-backend tuning table. The register block is derived from `sizeof(T)`
    against a byte budget rather than selected from upstream's set of hand-
    tuned `Tile<>` instantiations (`Tile<8,8,8,8>` for large matrices,
    `Tile<2,2,8,8>` for small, separate Intel-GPU configs, etc.). The
    work-group is a fixed 8x8 in the local-tiled variant, and comes from
    `Device::work_group_size` in the no-local one.
  - `tile_k` is a fixed constant, not portBLAS's cache-line-width-driven
    `cl_elems` sizing.
  - No vectorized `sycl::vec` packet load/store (`Packetize<>`) - scalar
    element-at-a-time loads throughout.
  - No "internal vs. edge tile" template-bool fast-path split - every tile
    takes the bounds-checked path uniformly.
  - No double buffering - one local-memory buffer, two barriers per
    K-iteration instead of portBLAS's buffer-flip trick that elides one. Left
    out on purpose rather than by oversight: the project's only reachable
    device is an OpenCL CPU, which takes the no-local variant and has no
    barriers to elide, so a restructured barrier sequence could not be
    verified here at all. See the comment at the K-loop for the full reason.
  - Native `sycl::nd_range<3>` (batch, row-tile, col-tile) instead of
    portBLAS's manual 1-D flattened index decomposition.
  - `sycl::group_barrier()` instead of the older `nd_item::barrier()` member
    form; no `decorated::legacy` pointer-decoration compiler-compat hedges.

  **SushiBLAS's own additions:**
  - The `Epilogue` template parameter, applied per element immediately before
    the store and defaulting to an identity functor. It takes the element's
    `(row, col)` so a bias broadcast is expressible, which is what makes it a
    fusion hook rather than an elementwise map. Not modelled on anything
    upstream: portBLAS's GEMM has no epilogue seam.

  **Worked out independently rather than carried over** (upstream may or may
  not do the equivalent - these were derived from this kernel's own geometry,
  not read off `gemm_local.hpp`, so no claim is made either way):
  - Bank padding on the A tile's leading dimension, and a strided-lattice
    register-block ownership so that neighbouring work-items address
    neighbouring elements. The arithmetic for both is written out at the read
    site.

- **`kernels/sycl/blas/level3/syrk.hpp`**
  **No dedicated SYRK tiling kernel exists upstream** - portBLAS's own
  `doc/MissingFeatures.md` lists SYRK as unimplemented. `C =
  alpha*op(A)*op(A)^T + beta*C` is mathematically `GEMM(A, A, transA=trans,
  transB=!trans)` restricted to one triangle of `C`, so this file reuses
  `gemm.hpp`'s operand-addressing primitives (`Detail::strides_for`,
  `Detail::load_operand`) with the second GEMM operand set to `A` itself,
  and adds a triangular output-write mask (per-tile early skip when the
  whole work-group block falls outside the requested triangle, plus a
  per-element mask on the final store). This composition pattern mirrors how
  portBLAS's own SYMM (`src/interface/symm_interface.hpp`) reuses its GEMM
  kernel for a different symmetric operation - but the triangular-output
  logic itself is original SushiBLAS work, not a port.

  It deliberately does **not** reuse GEMM's tile geometry: `SyrkTileConfig`
  is SYRK's own, so retuning GEMM's register block cannot silently retune
  SYRK's. SYRK has also not been given GEMM's local/no-local split - it
  always takes the local-memory shape, which is what it has always done and
  is a named follow-up rather than a regression.

## Not ported

FFT/DFT and LAPACK (`SVD`, `QR`, eigendecomposition, etc.) remain quarantined
per decision D1 - see `include/SushiBLAS/experimental/`. The backend
strategy for those is a named follow-up, not solved here.

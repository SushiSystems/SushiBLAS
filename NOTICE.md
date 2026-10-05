# Notices

SushiBLAS is Copyright (c) 2026 Sushi Systems and licensed under the PolyForm Noncommercial
License 1.0.0; see `LICENSE`. The material below keeps its own licence.

## Ported files

| File | Source | Version | Licence |
| --- | --- | --- | --- |
| `include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp` | portBLAS, https://github.com/codeplaysoftware/portBLAS, `src/operations/blas3/gemm_local.hpp` and `gemm_no_local.hpp` | unverified; the repository was archived in January 2025 | Apache-2.0, Copyright (C) Codeplay Software Limited; text in `third_party/licenses/Apache-2.0.txt` |
| `kernels/sycl/blas/level3/syrk.hpp` | The GEMM tiling above, composed for SYRK; portBLAS has no SYRK kernel | as above | Apache-2.0 for the tiling it carries over, as above |

```
portBLAS / oneMath SYCL BLAS
Copyright (C) Codeplay Software Limited.
Licensed under the Apache License, Version 2.0.
```

portBLAS continues at the UXL Foundation's generic-sycl-components,
https://github.com/uxlfoundation/generic-sycl-components. The Sushi Systems licence covers the
changes to these two files, not the upstream algorithm. The sections "Port notes" and "Not
ported" below state what was changed.

The license block of both files carries the upstream lines. This command writes every license
block in the repository and keeps them:

```
python D:/Projects/sushiskills/tools/licensing/write_license_block.py D:/Projects/sushiblas --project "SushiBLAS - https://github.com/SushiSystems/SushiBLAS" --upstream "include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp=Tiling ported from portBLAS, Copyright (C) Codeplay Software Limited;Apache-2.0, see NOTICE.md" --upstream "kernels/sycl/blas/level3/syrk.hpp=Tiling ported from portBLAS, Copyright (C) Codeplay Software Limited;Apache-2.0, see NOTICE.md"
```

## Algorithms written from a published reference

| File | Reference | Licence of the reference |
| --- | --- | --- |
| `kernels/sycl/random/philox.hpp` | Philox4x32-10 from Salmon, Moraes, Dror and Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", SC'11; round constants checked against Random123, https://github.com/DEShawResearch/random123, `include/Random123/philox.h` | unverified |

Whether `philox.hpp` was written from the paper or adapted from the Random123 source is not
recorded in the tree.

## Adapted text

| File | Source | Version | Licence |
| --- | --- | --- | --- |
| `docs/CODE_OF_CONDUCT.md` | Contributor Covenant, https://www.contributor-covenant.org/version/2/1/code_of_conduct.html | 2.1 | unverified; the audit names CC BY 4.0 |

## Third-party components

Nothing is vendored, nothing is fetched at configure time and no binary is redistributed. The
build and the command line interface resolve these from the machine; none of their files sits
in this repository, and no licence below was read from its source.

| Component | Used by | Source | Version | Licence |
| --- | --- | --- | --- | --- |
| SushiRuntime | `cmake/Runtime.cmake` | https://github.com/SushiSystems/SushiRuntime | sibling checkout or installed package, not pinned | Sushi Systems' own; see that repository's `LICENSE` |
| SYCL headers and runtime, Intel oneAPI DPC++ | every `<sycl/...>` include, `CMakePresets.json` | https://github.com/intel/llvm | not pinned | unverified |
| oneTBB | `CMakePresets.json` library path | Intel oneAPI | not pinned | unverified |
| GoogleTest | `tests/CMakeLists.txt` | https://github.com/google/googletest | not pinned | unverified; the audit names BSD-3-Clause |
| hwloc | through SushiRuntime, from vcpkg on Windows (`cmake/Vcpkg.cmake`) | https://github.com/open-mpi/hwloc | not pinned | unverified; the audit names BSD-3-Clause |
| sushicore | `cli/pyproject.toml` | PyPI | >=0.7.0 | Sushi Systems' own; see that package's licence |
| typer | `cli/pyproject.toml` | PyPI | >=0.12 | unverified; the audit names MIT |
| rich | `cli/pyproject.toml` | PyPI | >=13.0 | unverified; the audit names MIT |
| tomli | `cli/pyproject.toml`, Python below 3.11 | PyPI | >=2.0 | unverified; the audit names MIT |
| setuptools | `cli/pyproject.toml`, build backend | PyPI | >=61.0 | unverified |
| pytest, pyflakes | `.github/workflows/ci.yml` | PyPI | not pinned | unverified |
| `intel/oneapi-basekit` container image | `.github/workflows/ci.yml` | Docker Hub | `latest` | unverified; the audit names the Intel EULA |

## Port notes

Per SushiBLAS's own convention (P3: "when you do fork, fork on purpose and
own it"), the source below was rewritten from portBLAS's C++ expression-tree
executor framework into SushiBLAS's flat template-function convention. There
is no free rebase from upstream after this point - tuning and maintenance of
these kernels is SushiBLAS's own responsibility from here on.

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

## Earlier versions

No version was tagged or released before this licence. From git: the first commit, `8c64d41`,
carries no licence file. The commits from `e8ada31` (2026-02-19) up to and including `c381046`
carried the MIT License. The commits from `d66c872` (2026-08-01) up to the commit
that introduced this file's licence statement carried the Apache License 2.0; of those,
`origin/main` held the ones up to and including `aee576a` (2026-09-22) on 2026-10-05. Anyone
who received a copy keeps the licence that copy carried.

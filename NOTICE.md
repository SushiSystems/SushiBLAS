# Notices

SushiBLAS is Copyright (c) 2026-present Mustafa Garip & Sushi Systems and licensed under the PolyForm Noncommercial
License 1.0.0; see `LICENSE`. The material below keeps its own licence.

## What ships in this repository

Three third-party works have code or text in the tracked files: portBLAS, the Contributor
Covenant and the reference BLAS that LAPACK distributes, whose Fortran the two rotation bricks
were transcribed from. Random123 is listed with them: the owner does not recall whether
`philox.hpp` was adapted from its source, so the file is treated as if it was. Licence texts live
in `third_party/licenses/`.

| Work | Files here | Licence | Licence text |
| --- | --- | --- | --- |
| portBLAS, Copyright (C) Codeplay Software Limited | `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp`, `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level3/syrk.hpp`, `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level3/syr2k.hpp` | Apache-2.0 | `third_party/licenses/Apache-2.0.txt` |
| Random123, Copyright 2010-2012, D. E. Shaw Research | `modules/kernels/random_kernels/include/SushiBLAS/random_kernels/sycl/philox.hpp` | BSD-3-Clause | `third_party/licenses/BSD-3-Clause-Random123.txt` |
| Reference BLAS, as LAPACK distributes it, Copyright (c) 1992-2025 The University of Tennessee and The University of Tennessee Research Foundation, (c) 2000-2025 The University of California Berkeley, (c) 2006-2025 The University of Colorado Denver | `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/givens.hpp`, `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/modified_givens.hpp` | modified BSD, SPDX `BSD-3-Clause-Open-MPI` | `third_party/licenses/BSD-3-Clause-Open-MPI-LAPACK.txt` |
| Contributor Covenant 2.1 | `.github/CODE_OF_CONDUCT.md` | not established; see "Adapted text" below | not in `third_party/licenses/` |

No other tracked file carries a third-party copyright line. Outside `LICENSE`, this file,
`docs/`, `third_party/` and the fixtures under `tools/`, a search of every tracked file for
`Copyright` found a holder other than Sushi Systems only in the two portBLAS headers on
2026-10-07, before `syr2k.hpp` was written; that file carries the same portBLAS line, and the
search was not repeated.

## Ported files

| File | Source | Version | Licence |
| --- | --- | --- | --- |
| `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp` | portBLAS, https://github.com/codeplaysoftware/portBLAS, `src/operations/blas3/gemm_local.hpp` and `gemm_no_local.hpp` | no upstream commit is recorded in this tree; the repository was archived in January 2025 | Apache-2.0, Copyright (C) Codeplay Software Limited; text in `third_party/licenses/Apache-2.0.txt` |
| `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level3/syrk.hpp` | The GEMM tiling above, composed for SYRK; portBLAS has no SYRK kernel | as above | Apache-2.0 for the tiling it carries over, as above |
| `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level3/syr2k.hpp` | The same tiling with two register blocks, composed for SYR2K and HER2K | as above | Apache-2.0 for the tiling it carries over, as above |

```
portBLAS / oneMath SYCL BLAS
Copyright (C) Codeplay Software Limited.
Licensed under the Apache License, Version 2.0.
```

The `LICENSE` file on portBLAS's `master` branch was read on 2026-10-07 and is the Apache License,
Version 2.0. portBLAS continues at the UXL Foundation's generic-sycl-components,
https://github.com/uxlfoundation/generic-sycl-components. The Sushi Systems licence covers the
changes to these three files, not the upstream algorithm. The sections "Port notes" and "Not
ported" below state what was changed.

Lines 21 and 22 of each file's license block carry the upstream lines. This command, run from the
repository root, writes every license block in the repository and keeps them:

```
python tools/licensing/write_license_block.py . --project "SushiBLAS - https://github.com/SushiSystems/SushiBLAS" --upstream "modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp=Tiling ported from portBLAS, Copyright (C) Codeplay Software Limited;Apache-2.0, see NOTICE.md" --upstream "kernels/sycl/blas/level3/syrk.hpp=Tiling ported from portBLAS, Copyright (C) Codeplay Software Limited;Apache-2.0, see NOTICE.md" --upstream "kernels/sycl/blas/level3/syr2k.hpp=Tiling ported from portBLAS, Copyright (C) Codeplay Software Limited;Apache-2.0, see NOTICE.md"
```

The command was extended with the `syr2k.hpp` argument on 2026-10-07 and has not been run in
that form.

## Algorithms written from a published reference

| File | Reference | Licence of the reference |
| --- | --- | --- |
| `modules/kernels/random_kernels/include/SushiBLAS/random_kernels/sycl/philox.hpp` | Philox4x32-10 from Salmon, Moraes, Dror and Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", SC'11; round constants checked against Random123, https://github.com/DEShawResearch/random123, `include/Random123/philox.h` | The paper carries no code licence. Random123 is BSD-3-Clause, Copyright 2010-2012, D. E. Shaw Research; its `LICENSE` on `main` was fetched on 2026-10-09 and is in `third_party/licenses/` |
| `modules/kernels/random_kernels/include/SushiBLAS/random_kernels/sycl/poisson.hpp`, `poisson_ptrs` | Hoermann's transformed rejection with squeeze (PTRS), Insurance: Mathematics and Economics 12, 1993, as `kernels/README.md` cites it | A journal article; no code licence |
| `modules/kernels/random_kernels/include/SushiBLAS/random_kernels/sycl/poisson.hpp`, `poisson_knuth` | Knuth's multiplication loop for a Poisson draw, as `kernels/README.md` names it | A textbook algorithm; no code licence |
| `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/givens.hpp` | The `xROTG` algorithm LAPACK 3.10 introduced, transcribed from the reference BLAS sources `drotg.f90` and `zrotg.f90` as a web fetch returned them on 2026-10-07 | LAPACK's `LICENSE` on `master`, fetched on 2026-10-09 and kept in `third_party/licenses/` |
| `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/modified_givens.hpp` | The modified Givens construction, transcribed from the reference BLAS source `drotmg.f` the same way | as above |

Whether `philox.hpp` was written from the paper or adapted from the Random123 source is not
recorded, and the owner does not recall. The file is treated as adapted: its license block names
Random123 and the BSD-3-Clause text is in `third_party/licenses/`.

The two rotation bricks were written from source text, through a fetch that passes the page
through a summarising model, so they follow the reference BLAS line by line as far as that fetch
was faithful. Each file's license block names LAPACK, and LAPACK's licence text is in
`third_party/licenses/`.

## Adapted text

| File | Source | Version | Licence |
| --- | --- | --- | --- |
| `.github/CODE_OF_CONDUCT.md` | Contributor Covenant, https://www.contributor-covenant.org/version/2/1/code_of_conduct.html | 2.1 | not established on 2026-10-09 |

The file's "Attribution" section names the Covenant and its version. The Covenant's 2.1 text,
fetched on 2026-10-07, states no licence in its body. On 2026-10-09 the Covenant's home page,
its 2.1 page and its FAQ stated no licence for the text either, and the `LICENSE.md` of its
repository, https://github.com/EthicalSource/contributor_covenant, was the Hippocratic License 3.0,
which the site's footer applies to the site's source code. The CC BY 4.0 the estate audit names
was not found in any of the four. The licence of the adapted text is therefore not established.

## Third-party components that do not ship

Nothing is vendored, nothing is fetched at configure time and no binary is redistributed. The
build and the command line interface resolve these from the machine; none of their files sits
in this repository. The licence column says where each licence was read.

| Component | Used by | Source | Version | Licence |
| --- | --- | --- | --- | --- |
| SushiRuntime | `cmake/Runtime.cmake` | https://github.com/SushiSystems/SushiRuntime | the package the CLI built, at the tag `cli/sushistack.deps.toml` pins | PolyForm Noncommercial 1.0.0, from the checkout's `LICENSE` |
| SYCL headers and runtime, Intel oneAPI DPC++ | every `<sycl/...>` include; `intel-oneapi-compiler-dpcpp-cpp` from Intel's apt repository in `.github/actions/setup-oneapi/action.yml` | https://github.com/intel/llvm | not pinned | unverified |
| Intel CPU Runtime for OpenCL Applications | the SYCL CPU device of the Windows CI job; `w_opencl_runtime_p` installed by `.github/actions/setup-opencl-cpu/action.yml` | https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html | 2026.0.0.946, pinned with its SHA256 | unverified |
| GoogleTest | `cmake/SushiBLASTests.cmake`; `libgtest-dev` in `.github/actions/setup-oneapi/action.yml` | https://github.com/google/googletest | not pinned | BSD-3-Clause, Copyright 2008, Google Inc., from the workspace's vcpkg `share/gtest/copyright` |
| hwloc | through SushiRuntime, from vcpkg on Windows (`cmake/Vcpkg.cmake`); `libhwloc-dev` in `.github/actions/setup-oneapi/action.yml` | https://github.com/open-mpi/hwloc | not pinned | a BSD licence, from the workspace's vcpkg `share/hwloc/copyright`; the audit names BSD-3-Clause |
| sushicore | `cli/pyproject.toml` | PyPI | >=0.9.7 | Sushi Systems' own; the 0.5.0 installed on this machine declares Apache-2.0, and no 0.7.0 or later was read |
| typer | `cli/pyproject.toml` | PyPI | >=0.12 | MIT, from the metadata of the installed 0.20.0 |
| rich | `cli/pyproject.toml` | PyPI | >=13.0 | MIT, from the metadata of the installed 15.0.0 |
| tomli | `cli/pyproject.toml`, Python below 3.11 | PyPI | >=2.0 | MIT, from the metadata of the installed 2.4.0 |
| setuptools | `cli/pyproject.toml`, build backend | PyPI | >=61.0 | MIT, from the metadata of the installed 82.0.1 |
| pytest | `.github/workflows/ci.yml`, `cli/tests/` | PyPI | not pinned | MIT, from the metadata of the installed 9.1.1 |
| pyflakes | `.github/workflows/ci.yml` | PyPI | not pinned | unverified; not installed on this machine |

## Port notes

Per SushiBLAS's own convention (P3: "when you do fork, fork on purpose and
own it"), the source below was rewritten from portBLAS's C++ expression-tree
executor framework into SushiBLAS's flat template-function convention. There
is no free rebase from upstream after this point - tuning and maintenance of
these kernels is SushiBLAS's own responsibility from here on.

- **`modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp`**
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
  - `sycl::vec` loads in place of `Packetize<>`: 16-byte loads of whole,
    aligned blocks of FLOAT32, FLOAT64 and HALF operands, chosen per operand
    at run time; every other block is read element by element.
  - The whole-tile fast path is a run-time branch, uniform across a
    work-group, in place of a template bool.
  - No double buffering: one local buffer of each tile and two barriers per
    K-iteration, which measured faster on the RTX 3080 Ti at order 2048.
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

- **`modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level3/syrk.hpp`**
  **No dedicated SYRK tiling kernel exists upstream** - portBLAS's own
  `doc/MissingFeatures.md` lists SYRK as unimplemented. `C =
  alpha*op(A)*op(A)^T + beta*C` is mathematically `GEMM(A, A, transA=trans,
  transB=!trans)` restricted to one triangle of `C`, so this file reuses
  `gemm.hpp`'s operand-addressing primitives (`OperandAccess`,
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

LAPACK beyond LU (`SVD`, `QR`, eigendecomposition, etc.) is declared and not
defined, see `modules/operations/experimental/include/SushiBLAS/experimental/`.

## Earlier versions

No version was tagged or released before this licence. From git: the first commit, `8c64d41`,
carries no licence file. The commits from `e8ada31` (2026-02-19) up to and including `c381046`
carried the MIT License. The commits from `d66c872` (2026-08-01) up to the commit
that introduced this file's licence statement carried the Apache License 2.0; of those,
`origin/main` held the ones up to and including `aee576a` (2026-09-22) on 2026-10-05. Anyone
who received a copy keeps the licence that copy carried.

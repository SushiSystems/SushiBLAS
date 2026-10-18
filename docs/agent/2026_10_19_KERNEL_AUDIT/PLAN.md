# Plan

Implements the two decisions the kernel audit left open (`REPORT.md`). The owner delegated both
to the engineer on 2026-10-19. `draw_index` keeps its modulo; its brief and
`modules/kernels/README.md` now state the bias.

## Decision 1: fused multiply-add everywhere

Every dot-product kernel accumulates through `Kernels::Detail::multiply_add`, on the SYCL path
and on the reference path alike. One rounding per term is more accurate than two, costs one
instruction instead of two on every target the library runs on, and makes SYRK(A) equal
GEMM(A, Aᵀ) bit for bit on its triangle. GEMM already accumulates this way (commit 39bc0a4).

| # | Task | Done when |
| --- | --- | --- |
| 1 | Accumulate SYRK and SYR2K through `multiply_add`, SYCL and reference | A `test_syrk.cpp` case finds SYRK(A) bit-equal to GEMM(A, Aᵀ) on the stored triangle |
| 2 | Accumulate GEMV and `row_reduction` through `multiply_add`, SYCL and reference | `test_gemv` parity holds with its tolerance unchanged or tightened |
| 3 | Accumulate the sparse products (spmv, spmm, spsymv) through `multiply_add`, SYCL and reference | `test_sparse` parity holds with its tolerance unchanged or tightened |
| 4 | Accumulate `window_dot` and its two gradients through `multiply_add`, SYCL and reference | `test_spatial` parity holds with its tolerance unchanged or tightened |
| 5 | State the policy once in `docs/architecture/` and link it from each kernel module README | `check_docs_truth.py` passes and a grep finds no `+= multiply(` in a dot-product kernel |

Each task is one build and one `sb test` run, which the owner starts.

## Decision 2: `joint_matrix` for HALF GEMM

HALF GEMM gets a third GPU path built on `sycl::ext::oneapi::experimental::matrix`. Matrix
units are the only way SYCL reaches the tensor cores of a card such as the RTX 3080 Ti, and the
scalar tiled path cannot come near them for HALF. The path accumulates in FLOAT32 and rounds to
HALF once, in the epilogue.

Its boundaries:

- It lives in its own header, `blas_seam_kernels/sycl/level3/gemm_joint_matrix.hpp`, compiled
  only under `SYCL_EXT_ONEAPI_MATRIX`. No public operation header names an extension type, so
  the C++17 ABI of the stack does not change.
- `Device::Profile` gains a field that says whether the device offers a HALF x HALF -> FLOAT32
  combination, read from `ext_oneapi_matrix_combinations`. The GEMM dispatch chooses the path
  from that field and from nothing else.
- The local-tiled path stays the fallback on every device and the parity truth for the tests.
- The work starts after the coalescing, full-tile and double-buffer tasks of the backlog, so it
  is measured against a tuned baseline and not against today's kernel.

| # | Task | Done when |
| --- | --- | --- |
| 6 | Add the matrix-combination field to `Device::Profile` | A `test_device_policy.cpp` case reads it on the CUDA card and on the CPU device |
| 7 | Write `gemm_joint_matrix` for HALF, NoTrans x NoTrans, tile multiples only | A `test_gemm.cpp` case matches the local-tiled result within the HALF tolerance at 1024 |
| 8 | Extend it to every transpose and to edge tiles through the local-tiled path | Every HALF case of `test_gemm` passes on that path |
| 9 | Route HALF GEMM to it when the profile allows | `benchmark_gemm` HALF median, five runs before and after on the RTX 3080 Ti, rises beyond the spread; otherwise the routing is reverted and the path stays unused |
| 10 | Record the path in `blas_seam_kernels/README.md` and the changelog | `check_docs_truth.py` and `check_changelog.py` pass |

Tasks 6 to 10 run one after another; each is one build. How the plan runs, inline or in waves,
is the owner's choice.

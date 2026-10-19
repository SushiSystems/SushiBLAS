# Plan

**Status:** Open — tasks 1 to 10 and B1 to B27, in RW-52, RW-53 and RW-54.

Implements the two decisions the kernel audit left open (`REPORT.md`), then the benchmarks and
the measured optimisations of RW-52 and RW-53. The owner delegated both decisions to the
engineer on 2026-10-19. `draw_index` keeps its modulo; its brief and
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

## Benchmarks and measured optimisations (RW-52, RW-53)

The owner chose on 2026-10-19 to run RW-52 and then RW-53 inline, in one session, in the order
below. Each build and each benchmark run waits for the owner's go-ahead. A build runs with
`CMAKE_BUILD_PARALLEL_LEVEL=4` and a test run with `sb test --jobs 4`.

### How a task is measured

- The build is release, configured with `SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86"` and
  `--benchmarks`. The GPU is the RTX 3080 Ti, selected with `ONEAPI_DEVICE_SELECTOR=cuda:0`; the
  CPU is the Ryzen 5 7600X, selected with `opencl:cpu`.
- One run is one invocation of `sb run benchmark_<name>`. Each program already times 10
  iterations after 2 warm-ups and prints their median. A figure here is the median of five
  runs' medians, and its spread is the lowest and highest of the five.
- The "before" of a task is the "after" of the last task kept on the same kernel, or the
  baseline of B9 when there is none. Only the "after" costs five new runs.
- A change is kept when its median moves beyond the spread of its "before". Otherwise it is
  reverted in the same task and the task is recorded as measured and reverted.
- A kept change carries the small and medium tests green (`sb test --jobs 4`), and the large
  tests of the kernel it touched, before its `perf(...)` commit, whose body gives both figures.

### Setup and benchmarks (RW-52)

The owner chose on 2026-10-19 to extend `benchmark_band_packed` and `benchmark_triangular` rather
than write `benchmark_level2` and `benchmark_syrk`: the first already times dense gemv, symv,
trmv, syr and spr, and the second syrk and syr2k at orders 256, 1024 and 4096, both in
ColumnMajor only.

| # | Task | Done when | State |
| --- | --- | --- | --- |
| B1 | Configure the build for `spir64;nvidia_gpu_sm_86` with the benchmarks | `benchmark_gemm` prints the RTX 3080 Ti under `cuda:0` and the CPU under `opencl:cpu` | done |
| B2 | Sweep RowMajor in `benchmark_gemm` beside ColumnMajor | Each section of `benchmark_gemm` names its layout and both layouts print | done |
| B3 | Time HALF in `benchmark_gemm`, which the harness fills and names | `benchmark_gemm` prints HALF rows on both devices | done |
| B4 | Time copy, scal, swap and rot in `benchmark_level1` | The program prints one row per routine and length on both devices | open |
| B5 | Time `benchmark_band_packed` in both layouts, with gemv Trans, ger, syr2 and spr2 added | The program prints one section per layout, with the four new rows, on both devices | open |
| B6 | Time `benchmark_triangular` in both layouts | The program prints one section per layout, its syrk and syr2k rows among them, on both devices | open |
| B7 | Write `benchmark_spatial` over the spatial kernels | The program prints one row per operation on both devices | open |
| B8 | Write `benchmark_random` over the draws and `orthonormalize` | The program prints one row per routine on both devices | open |
| B9 | Record the baseline: five runs of every benchmark on both devices | `REPORT.md` holds the median and spread of every row | open |

### GEMM on the GPU (RW-53)

| # | Task | Measured on | State |
| --- | --- | --- | --- |
| B10 | Stage each operand along its contiguous axis in the tiled, `no_local` and `depth_split` loads | `benchmark_gemm`, `cuda:0` | open |
| B11 | Load full tiles without bounds checks, with 32-bit offsets | `benchmark_gemm`, `cuda:0` | open |
| B12 | Stage through 128-bit loads | `benchmark_gemm`, `cuda:0` | open |
| B13 | Double-buffer the K loop | `benchmark_gemm`, `cuda:0` | open |
| B14 | Tune `K_TILE_K` and the `GemmLocalTile` shape | `benchmark_gemm`, `cuda:0` | open |
| B15 | Tasks 6 to 10 above: the HALF `joint_matrix` path | `benchmark_gemm` HALF, `cuda:0` | open |

### Elsewhere (RW-53)

| # | Task | Measured on | State |
| --- | --- | --- | --- |
| B16 | Run the TRSM trailing update through `gemm_accessed` | `benchmark_triangular`, both devices | open |
| B17 | Pack each B panel of the CPU GEMM once per depth block | `benchmark_gemm`, `opencl:cpu` | open |
| B18 | Choose the CPU register block per ISA, AVX-512 on the 7600X | `benchmark_gemm_blocking`, `opencl:cpu` | open |
| B19 | Run unit-stride level 1 through `Launch::map` and `zip_inplace` | `benchmark_level1`, both devices | open |
| B20 | Run GER, SYR, SYR2, SPR and SPR2 along the contiguous axis, on the stored triangle only | `benchmark_band_packed`, both devices | open |
| B21 | End `row_reduction` in a fixed-order sub-group shuffle | `benchmark_band_packed` gemv, both devices | open |
| B22 | Compute ‖vᵢ‖² once per step of `orthonormalize` | `benchmark_random`, both devices | open |
| B23 | Decode the spatial flat index in 3-D or 32-bit | `benchmark_spatial`, both devices | open |
| B24 | Transpose through local-memory tiles in `layout_convert` | `benchmark_spatial`, both devices | open |
| B25 | Scatter the transposed `spmv` and `spsymv` over more than 256 work-items | `benchmark_sparse`, both devices | open |
| B26 | Read `Device::profile_for` without a mutex and a scan | `benchmark_level1` small sizes, both devices | open |
| B27 | Close RW-52 and RW-53: changelog, backlog, module READMEs, this status | The checkers of `AGENTS.md` pass | open |

Every task from B2 on is one build. B15 is five builds, one per task 6 to 10.

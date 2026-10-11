# benchmarks

Eight standalone programs that time SushiBLAS operations. Six go through the public API: an
`Engine`, `create_tensor`, `IO::from_vector` and the operation views (`BLASOps`, `SparseOps`,
`ReductionOps`, `ElementwiseOps`, `NonLinearOps`). `benchmark_gemm_blocking` calls the GEMM kernel
directly, and `benchmark_triangular` calls the TRSM kernel directly for some of its rows. They use no
test framework and never pass or fail on a number.

| File | Holds |
| --- | --- |
| `benchmark_common.hpp` | The argument, environment, fill, timing and table helpers the programs share |
| `benchmark_gemm.cpp` | Square GEMM through `BLASOps` |
| `benchmark_gemm_blocking.cpp` | One square GEMM kernel call under a cache blocking given on the command line |
| `benchmark_level1.cpp` | `axpy`, `dot`, `nrm2`, `asum`, `iamax` |
| `benchmark_band_packed.cpp` | The band and packed Level 2 routines beside their dense siblings |
| `benchmark_triangular.cpp` | The triangular and symmetric BLAS routines, and the TRSM kernel at chosen block widths |
| `benchmark_sparse.cpp` | `spmv`, `spsymv`, `spmm`, `sptrsv`, `axpyi`, `doti` |
| `benchmark_reductions.cpp` | `sum`, `max`, `var` over the whole tensor |
| `benchmark_elementwise.cpp` | `add`, in-place `exp`, in-place `relu` |

## Build and run

```
sb build --benchmarks
sb run benchmark_gemm
sb run benchmark_level1
sb run benchmark_reductions
sb run benchmark_elementwise
sb run benchmark_band_packed
sb run benchmark_triangular
sb run benchmark_sparse
sb run benchmark_gemm_blocking -- f32 2048
```

The benchmark targets are off by default. `--benchmarks` adds them to the build and the setting stays
in the build tree until `--no-benchmarks` or `--clean` removes it.

Arguments for the program follow `--`: `sb run <program> -- <sizes>`, for example
`sb run benchmark_triangular -- 512 2048`. Seven programs take an optional list of sizes and fall
back to their defaults when none are given. A size that is not a positive integer stops the
program with exit code 1. `benchmark_gemm_blocking` takes its own arguments, described below.

The device is the one the runtime selects; `sb` has no device option. The environment variable
`ONEAPI_DEVICE_SELECTOR` pins one, for example `ONEAPI_DEVICE_SELECTOR=opencl:cpu`. Every program
prints the name of the device it ran on.

## What each program measures

| Program | Operations | Default sizes | Rate |
| --- | --- | --- | --- |
| `benchmark_gemm` | `gemm`, C = A * B, square, column-major | n = 128, 512, 1024, 2048 | GFLOP/s, 2 * n^3 flops |
| `benchmark_level1` | `axpy`, `dot`, `nrm2`, `asum`, `iamax` | 2^16, 2^20, 2^24 elements | GB/s |
| `benchmark_reductions` | `sum`, `max`, `var` over the whole tensor | 2^16, 2^20, 2^24 elements | GB/s |
| `benchmark_elementwise` | `add`, in-place `exp`, in-place `relu` | 2^16, 2^20, 2^24 elements | GB/s |
| `benchmark_band_packed` | `gemv`, `gbmv`, `symv`, `sbmv`, `spmv`, `trmv`, `tbmv`, `tpmv`, `trsv`, `tbsv`, `tpsv`, `syr`, `spr`, column-major | n = 1024, 4096, 8192 | Melem/s |
| `benchmark_triangular` | `gemm`, `symm`, `syrk`, `syr2k`; `trmv` and `trsv` under `ForceSYCL` and `ForceReference`; `trsm` with 8 and with n right-hand sides; the TRSM kernel at block widths 32 to 1024 and at one block | n = 256, 1024, 4096 | GFLOP/s |
| `benchmark_sparse` | `spmv` (plain, transposed, conjugate-transposed), `spsymv`, `spmm` (plain, transposed), `sptrsv` (plain, transposed), `axpyi`, `doti`; 16 entries per row | n = 2^14, 2^17, 2^20 | GB/s |

Every operation runs in FLOAT32 and then FLOAT64.

### `benchmark_gemm_blocking`

```
sb run benchmark_gemm_blocking -- <f32|f64> <n> [kernel=packed|unpacked] [inner=BYTES]
    [outer=BYTES] [stream=BYTES] [workers=PER_UNIT] [units=COUNT]
```

It times one square GEMM of order `n` by calling `Kernels::SYCL::Detail::packed_gemm` or
`Kernels::SYCL::gemm` on device USM it allocates itself, with no graph around the call. `kernel=` chooses the packed kernel or the
one that takes no scratch. The other arguments override fields of the `Device::Profile` the
device reports: `inner`, `outer` and `stream` are the three byte budgets of `CacheBlocking`,
`workers` its work-items per compute unit and `units` the compute-unit count. It prints the plan
those values give (block rows, columns and depth, block count, workers, scratch bytes) and one
row in GFLOP/s. It exists to find a blocking by measurement on one machine.

The byte count behind GB/s is one read of each input element and one write of each output
element: `axpy` 3n, `dot` 2n, `nrm2`, `asum`, `iamax`, `sum`, `max` and `var` n, `add` 3n,
`exp` and `relu` 2n, each times the element size. A kernel that makes more than one pass, or
writes partial results to scratch, moves more bytes than this, so the printed rate is a lower
bound on the traffic the hardware served.

In-place `exp` runs on the same tensor every iteration, so its values overflow to infinity after
a few iterations; the timing then covers infinite inputs.

## How a number is produced

Each program prints the device name, the build configuration (`release` when `NDEBUG` is defined,
`debug` otherwise), the compute units, cache bytes and cache blocking of the device profile, and the
engine's execution mode. It fills the inputs once, runs 2 warm-up iterations, then times 10
iterations with `std::chrono::steady_clock`. One iteration of a program that goes through the public
API records the operation on the task graph and calls `Engine::execute`, so graph compilation and
dispatch are inside the measured time. The table gives the median and minimum time per iteration in
milliseconds, and the rate computed from the median.

The shared argument, environment, fill, timing and table code is in `benchmark_common.hpp`.

## Caveats

This repository is built and tested on the OpenCL CPU device only. No GPU number has been
measured here.

The four programs added on 2026-10-07 (`benchmark_gemm_blocking`, `benchmark_band_packed`,
`benchmark_triangular`, `benchmark_sparse`) had not been built or run when this page was written. What
they print is described from their source.

A number is meaningful only for the machine, device and build that printed it. Compare two
numbers only when the device and build lines above them match.

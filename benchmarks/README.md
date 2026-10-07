# benchmarks

Four standalone programs that time SushiBLAS operations through the public API (`Engine`,
`create_tensor`, `IO::from_vector`, `blas()`, `reductions()`, `elementwise()`, `nonlinear()`).
They use no test framework and never pass or fail on a number.

## Build and run

```
sb build --benchmarks
sb run bench_gemm
sb run bench_level1
sb run bench_reductions
sb run bench_elementwise
```

The benchmark targets are off by default. `--benchmarks` adds them to the build and the setting stays
in the build tree until `--no-benchmarks` or `--clean` removes it.

Each program takes an optional list of sizes as arguments and falls back to its defaults when
none are given. A size that is not a positive integer stops the program with exit code 1.

## What each program measures

| Program | Operations | Default sizes | Rate |
| --- | --- | --- | --- |
| `bench_gemm` | `gemm`, C = A * B, square, column-major | n = 128, 512, 1024, 2048 | GFLOP/s, 2 * n^3 flops |
| `bench_level1` | `axpy`, `dot`, `nrm2`, `asum`, `iamax` | 2^16, 2^20, 2^24 elements | GB/s |
| `bench_reductions` | `sum`, `max`, `var` over the whole tensor | 2^16, 2^20, 2^24 elements | GB/s |
| `bench_elementwise` | `add`, in-place `exp`, in-place `relu` | 2^16, 2^20, 2^24 elements | GB/s |

Every operation runs in FLOAT32 and then FLOAT64.

The byte count behind GB/s is one read of each input element and one write of each output
element: `axpy` 3n, `dot` 2n, `nrm2`, `asum`, `iamax`, `sum`, `max` and `var` n, `add` 3n,
`exp` and `relu` 2n, each times the element size. A kernel that makes more than one pass, or
writes partial results to scratch, moves more bytes than this, so the printed rate is a lower
bound on the traffic the hardware served.

In-place `exp` runs on the same tensor every iteration, so its values overflow to infinity after
a few iterations; the timing then covers infinite inputs.

## How a number is produced

Each program prints the device name, the build configuration (`release` when `NDEBUG` is
defined, `debug` otherwise) and the engine's execution mode. It fills the inputs once, runs
2 warm-up iterations, then times 10 iterations with `std::chrono::steady_clock`. One iteration
records the operation on the task graph and calls `Engine::execute`, so graph compilation and
dispatch are inside the measured time. The table gives the median and minimum time per
iteration in milliseconds, and the rate computed from the median.

The shared argument, environment, fill, timing and table code is in `bench_common.hpp`.

## Caveats

This repository is built and tested on the OpenCL CPU device only. No GPU number has been
measured here.

A number is meaningful only for the machine, device and build that printed it. Compare two
numbers only when the device and build lines above them match.

# Tests

Every test of SushiBLAS, in a tree that mirrors `modules/`. The tests of the module
`modules/<tier>/<module>/` are under `tests/<tier>/<module>/`, and the test file of a public
header `<SushiBLAS/<module>/<path>/<name>.hpp>` is `tests/<tier>/<module>/<path>/test_<name>.cpp`.
A header whose tests do not fit one file has a folder in its place, `test_<name>/`, with one file
per aspect. A header has the file or the folder, never both.

| Path | Holds |
| --- | --- |
| `<tier>/<module>/` | The tests and benchmarks of one module, and a README with the reasoning of its tests |
| `<tier>/<module>/test_<name>.cpp` | The tests of the header `<name>.hpp` of that module |
| `<tier>/<module>/test_<name>/` | The tests of one header, one file per aspect |
| `<tier>/<module>/<name>.hpp` | A helper only the tests of that module share; the module's README lists it under Helpers |
| `<tier>/<module>/benchmark_<name>.cpp` | A benchmark program; see [Benchmarks](#benchmarks) |
| `applications/<path>/` | The tests of the application under `applications/<path>/`, one file per header it holds |
| `package/` | A consumer of an installed SushiBLAS, configured on its own; see [its README](package/README.md) |

A test that crosses modules belongs to the highest-tier module it uses, in the test file of the
header whose behaviour it checks. The test-only modules count as used: a test built on
`SushiBLASTest` belongs to the `engine` tier or above. The umbrella `<SushiBLAS/SushiBLAS.hpp>` does
not count, because it is what a consumer includes. A test that checks several operation families
together belongs to `library/umbrella/test_SushiBLAS/`, the module that depends on all of them.

The helpers tests of several modules share are internal test-only modules under `modules/`,
listed in [`modules/README.md`](../modules/README.md). A helper the tests of one module share
sits in that module's folder, and a test includes it by its quoted relative path.

## One program per module

`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake` builds every program. A module
calls it in its own `CMakeLists.txt` with the files of its mirror and gets the `sushiblas_<module>_tests`. Every program lands in `build/tests/`. A new test file is
listed by hand in that call; nothing globs the folders. An application's tests are built the same
way by `sushiblas_add_application_test_program()`, called in the application's `CMakeLists.txt`,
which compiles the application's sources into the program.

## Sizes are labels

What a test needs to run is its size, and the size is its one CTest label. CI chooses its stages
by size, and CTest stops a test that outlives the timeout of its size.

| Size | The test may use | Timeout |
| --- | --- | --- |
| `small` | One process and one thread; no network, no sleep, no device, no file outside the build tree | 60 s |
| `medium` | Threads, localhost, files in a temporary folder, the OpenCL or another CPU SYCL device | 300 s |
| `large` | A real GPU, or a run longer than `medium` allows | 900 s |

A test's size is the prefix of its GoogleTest suite name: `Small_`, `Medium_` or `Large_`, as in
`TEST(Small_ComplexArithmetic, ...)`. No build file lists sizes. Each program registers the three
prefixes as three filters, each with its label and timeout, and a suite with none of them stops
the configure with its name.

A program whose suites are all `Small_` is linked with
`modules/foundation/test_runner/source/small_test_main.cpp`, which creates no runtime, so it
opens no device and starts no worker thread. A program with a `Medium_` or `Large_` suite creates
the shared runtime. A suite whose tests reach the runtime, a queue, a tensor or a file is
`Medium_`; the host reference kernels, the arithmetic and the type checks are `Small_`. No suite
is `Large_`.

A test that fails intermittently and cannot be fixed the day it is seen is named in the
`QUARANTINED` list of its program's call, as `Suite.Case`. It keeps its size and gains the label
`quarantined`, which takes it out of every stage that gates; `docs/reference/KNOWN_ISSUES.md`
names it, its owner and the date it is fixed by. No test is quarantined.

## Running

Tests run through `sb`, which sets the loader path the test program needs.

| Command | Runs |
| --- | --- |
| `sb test` | The `small` and `medium` tests, without the quarantined ones |
| `sb test --label small` | The `small` tests; `medium` and `large` work the same way, and the flag repeats |
| `sb test --label quarantined` | The quarantined tests only |
| `sb test --package` | Installs the library, then configures, builds and runs `package/` against the install |
| `sb test --cli` | The pytest tests of the `sb` CLI, by the same labels |
| `sb test --filter <pattern>` | The tests whose name matches the pattern |
| `sb run sushiblas_<module>_tests` | One module's program; flags after `--` go to GoogleTest |
| `sb bench` | Every benchmark program, after `sb build --benchmarks` |
| `sb run benchmark_<name>` | One benchmark program, after `sb build --benchmarks` |

[`cli/README.md`](../cli/README.md) describes each flag, and
[the contributing guide](../docs/CONTRIBUTING.md) states what a new operation's tests carry.

## What the tests depend on

A test program links the `sushiblas` library, GoogleTest and the include roots of the internal
modules, and takes the project's build policy. A test can therefore include a kernel from
`modules/kernels/` and call it directly, and it sees more than a consumer does. `package/` is the
one place that sees only the installed surface.

## Shared reasoning

The source comments state what a test checks. A module's README holds why a test is written the
way it is: the defect it pins, the number it was measured against, the quirk of the machine it ran
on. The reasoning below holds for tests of several modules.

A test builds its operation views on the fixture's engine, as in `BLASOps(*engine)`, or on a
`GraphProbe` when it asserts on the recorded graph; `modules/engine/engine_fixture/README.md`
and `modules/engine/graph_probe/README.md` describe them. A refusal at record time is asserted as
`std::invalid_argument`, and a mode or device refusal as `SushiBLAS::UnsupportedExecution`.

### Expectations are worked out by hand

Where a file says its expectations are hand-computed, they do not come from another path in
this library. A differential oracle agrees with the code under test in every case where both
are wrong the same way. That is how an inverted `ColumnMajor` addressing arm in GEMM survived
for months: the SYCL kernel and the reference kernel had the identical defect, so the
differential tests agreed.

### Comparing two kernels under the two forced modes

The `TheSYCLKernelAgreesWithTheReferenceKernel` tests fill two buffers under
`ExecutionMode::ForceSYCL` and `ExecutionMode::ForceReference` and compare them. That is how
the library checks kernel agreement: `ExecutionMode` has no mode that runs both, and
`Engine::set_mode` is `noexcept`.

A BLAS kernel that sums is not compared this way. The reference sums in `double` and the SYCL
kernel in the element type, FLOAT32 for HALF, so the BLAS parity tests measure each against a
`TruthSum` within a rounding bound; see `blas_truth.hpp` in `modules/foundation/blas_test_oracle/README.md`. A kernel
that only moves elements is still compared bit for bit.

### Comparing floats by bit pattern

The `*_float_bits` helpers and `bits_of` copy a float into a `uint32_t` with `std::memcpy`.
`std::bit_cast` is C++20 and this tree is C++17; `std::memcpy` between two trivially copyable
objects of equal size is the same operation with the same guarantees, and every compiler this
project targets folds it to a register move. Reading a float through a `uint32_t` lvalue would
be undefined behaviour the compiler is free to optimise on. The comparison is on bits because
`==` on floats says two NaNs differ and says +0 and -0 agree, and "the same answer" means
neither. `EXPECT_FLOAT_EQ` and a zero tolerance both compare values, so they have the same
blind spot.

### Ordering is asserted on in-degree

The ordering tests assert on a recorded task's in-degree, which `GraphProbe` reports as
`RecordedTask::dependency_count`, never on an output value. With no dependency
edge the two tasks may still come out in the right order by luck, so a correct number proves
nothing about ordering. `Medium_Sentinel.ReductionsOrderAgainstViewsOfOneAllocation`
explains this at length.

### Views at a non-zero offset

The ordering and storage tests put their operands partway into an arena. At offset zero a bare
`data()` pointer and a real `region()` name the same resource, so an operation that declared the
wrong one would still get its edge and the test would pass without proving anything. Partway
into an arena they diverge: `data()` is `base + offset`, and a region built from it widens to
the whole allocation of that address, which the tracker files under a resource the producer
never touched. Two views of one arena then look like two unrelated resources and the tracker
emits no edge at all. That cost SushiAI's first arena-packed training run.

## Benchmarks

Ten programs that time SushiBLAS operations, one per `benchmark_*.cpp` file. Eight go through the
public API: an `Engine`, `create_tensor`, `IO::from_vector` and the operation views (`BLASOps`,
`SparseOps`, `ReductionOps`, `ElementwiseOps`, `NonLinearOps`, `SpatialOps`, `RandomOps`). `benchmark_gemm_blocking` calls the
GEMM kernel directly, and `benchmark_triangular` calls the TRSM kernel directly for some of its
rows. They use no test framework, carry the target label `benchmark` and never pass or fail on a
number. Each sits beside the tests of the module it times, and `sushiblas_add_test_program()`
builds it when `SB_BUILD_BENCHMARKS` is on.

| File | Holds |
| --- | --- |
| `operations/blas/benchmark_gemm.cpp` | GEMM through `BLASOps`, in both layouts |
| `operations/blas/benchmark_gemm_blocking.cpp` | One square GEMM kernel call under a cache blocking given on the command line |
| `operations/blas/benchmark_level1.cpp` | `axpy`, `copy`, `scal`, `swap`, `rot`, `dot`, `nrm2`, `asum`, `iamax` |
| `operations/blas/benchmark_band_packed.cpp` | The band and packed Level 2 routines beside their dense siblings, in both layouts |
| `operations/blas/benchmark_triangular.cpp` | The triangular and symmetric BLAS routines, and the TRSM kernel at chosen block widths, in both layouts |
| `operations/sparse/benchmark_sparse.cpp` | `spmv`, `spsymv`, `spmm`, `sptrsv`, `axpyi`, `doti` |
| `operations/reductions/benchmark_reductions.cpp` | `sum`, `max`, `var` over the whole tensor |
| `operations/random/benchmark_random.cpp` | The `RandomOps` draws, `shuffle` and `orthogonal` |
| `operations/spatial/benchmark_spatial.cpp` | Every `SpatialOps` operation on a batch of NHWC images |
| `library/umbrella/benchmark_elementwise.cpp` | `add`, in-place `exp`, in-place `relu`; it crosses `elementwise` and `nonlinear` |

### Build and run

```
sb build --benchmarks
sb run benchmark_gemm
sb run benchmark_level1
sb run benchmark_reductions
sb run benchmark_elementwise
sb run benchmark_band_packed
sb run benchmark_triangular
sb run benchmark_sparse
sb run benchmark_spatial
sb run benchmark_random
sb run benchmark_gemm_blocking -- f32 2048
```

The benchmark targets are off by default. `--benchmarks` adds them to the build and the setting stays
in the build tree until `--no-benchmarks` or `--clean` removes it.

Arguments for the program follow `--`: `sb run <program> -- <sizes>`, for example
`sb run benchmark_triangular -- 512 2048`. Nine programs take an optional list of sizes and fall
back to their defaults when none are given. A size that is not a positive integer stops the
program with exit code 1. `benchmark_gemm_blocking` takes its own arguments, described below.

The device is the one the runtime selects; `sb` has no device option. The environment variable
`ONEAPI_DEVICE_SELECTOR` pins one, for example `ONEAPI_DEVICE_SELECTOR=opencl:cpu`. Every program
prints the name of the device it ran on.

### What each program measures

| Program | Operations | Default sizes | Rate |
| --- | --- | --- | --- |
| `benchmark_gemm` | `gemm`, C = A * B, square, every section once per layout: ColumnMajor, then RowMajor | n = 128, 512, 1024, 2048 | GFLOP/s, 2 * n^3 flops |
| `benchmark_gemm`, rows `gemm deep` and `gemm strided` | `gemm` in FLOAT32 with a small C and a long K: five shapes of C = A^T * B, where K is contiguous in memory under ColumnMajor, and six of C = A * B^T, where K is the leading axis under ColumnMajor; the size column is K | fixed | GFLOP/s, 2 * m * n * k flops |
| `benchmark_level1` | `axpy`, `copy`, `scal`, `swap`, `rot`, `dot`, `nrm2`, `asum`, `iamax` | 2^16, 2^20, 2^24 elements | GB/s |
| `benchmark_reductions` | `sum`, `max`, `var` over the whole tensor | 2^16, 2^20, 2^24 elements | GB/s |
| `benchmark_elementwise` | `add`, in-place `exp`, in-place `relu` | 2^16, 2^20, 2^24 elements | GB/s |
| `benchmark_band_packed` | `gemv` (plain, transposed), `gbmv`, `symv`, `sbmv`, `spmv`, `trmv`, `tbmv`, `tpmv`, `trsv`, `tbsv`, `tpsv`, `ger`, `syr`, `syr2`, `spr`, `spr2`; every order once per layout, ColumnMajor then RowMajor | n = 1024, 4096, 8192 | Melem/s |
| `benchmark_triangular` | `gemm`, `symm`, `syrk`, `syr2k`; `trmv` and `trsv` under `ForceSYCL` and `ForceReference`; `trsm` with 8 and with n right-hand sides; the TRSM kernel at block widths 32 to 1024 and at one block; every size once per layout, ColumnMajor then RowMajor | n = 256, 1024, 4096 | GFLOP/s |
| `benchmark_sparse` | `spmv` (plain, transposed, conjugate-transposed), `spsymv`, `spmm` (plain, transposed), `sptrsv` (plain, transposed), `axpyi`, `doti`; 16 entries per row | n = 2^14, 2^17, 2^20 | GB/s |
| `benchmark_spatial` | `im2col`, `col2im` and the three `window_dot` kernels on a 3x3 window padded by one; `pool_max`, `pool_avg` and their backwards on 2x2 windows of stride 2; `upsample_nearest` by 2 and its backward; `concat_channels`, `slice_channels`, `channel_affine`, `broadcast_spatial`; `layout_convert` to NCHW and back; 8 images of 64 channels | H = W = 32, 64, 128 | GB/s, each input read once and each output written once |
| `benchmark_random` | `uniform`, `normal`, `log_normal`, `exponential`, `poisson` at rates 4 (Knuth) and 30 (PTRS), `discrete_uniform` on [0, 9], `truncated_normal`, `bernoulli`, and `shuffle` up to 2^20 elements; then `orthogonal` on square matrices of order 64, 256 and 1024, sized by n | 2^16, 2^20, 2^24 elements | Melem/s |

Every operation runs in FLOAT32 and then FLOAT64; the square rows of `benchmark_gemm` also run in
HALF, and skip a dtype whose aspect the device lacks.

#### `benchmark_gemm_blocking`

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
element: `axpy` 3n, `copy` and `scal` 2n, `swap` and `rot` 4n, `dot` 2n, `nrm2`, `asum`,
`iamax`, `sum`, `max` and `var` n, `add` 3n, `exp` and `relu` 2n, each times the element size. A kernel that makes more than one pass, or
writes partial results to scratch, moves more bytes than this, so the printed rate is a lower
bound on the traffic the hardware served.

In-place `exp` runs on the same tensor every iteration, so its values overflow to infinity after
a few iterations; the timing then covers infinite inputs.

### How a number is produced

Each program prints the device name, the build configuration (`release` when `NDEBUG` is defined,
`debug` otherwise), the compute units, cache bytes and cache blocking of the device profile, the
engine's execution mode and its layout. It fills the inputs once, runs 2 warm-up iterations, then times 10
iterations with `std::chrono::steady_clock`. One iteration of a program that goes through the public
API records the operation on the task graph and calls `Engine::execute`, so graph compilation and
dispatch are inside the measured time. The table gives the median and minimum time per iteration in
milliseconds, and the rate computed from the median.

The shared argument, environment, fill, timing and table code is the internal module
`benchmark_harness`; see `modules/operations/benchmark_harness/README.md`.

### Caveats

The functional tests of this tree have run on one NVIDIA RTX 3080 Ti. Which benchmark programs
have been built and run is in `docs/reference/KNOWN_ISSUES.md`; what a program prints is
described here from its source.

A number is meaningful only for the machine, device and build that printed it. Compare two
numbers only when the device and build lines above them match.

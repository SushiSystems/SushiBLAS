# benchmark_harness

The helpers every benchmark program shares: the size arguments, the environment lines, the fill
of an input, the timing loop and the table a program prints. The module is internal and
benchmark-only: its headers are not installed, and it is added only when `SB_BUILD_BENCHMARKS`
is on.

It depends on `core`, `launch`, `tensor`, `engine` and `io`. Its header is included as
`<SushiBLAS/benchmark_harness/benchmark_harness.hpp>`.

| File | Declares |
| --- | --- |
| `benchmark_harness.hpp` | The argument, environment, fill, timing and table helpers the benchmarks share |

## benchmark_harness.hpp

`print_environment` prints the device name, the build configuration (`release` when `NDEBUG` is
defined, `debug` otherwise), the compute units, cache bytes and cache blocking of the device
profile, and the engine's execution mode. `time_iterations` runs 2 warm-up iterations, then
times 10 with `std::chrono::steady_clock` and returns the median and the minimum. The benchmark
section of `tests/README.md` describes the programs that use it.

# benchmark_harness

The helpers every benchmark program shares: the size arguments, the environment lines, the fill
of an input, the timing loop and the table a program prints. The module is internal and
benchmark-only: its headers are not installed, and it is added only when `SB_BUILD_BENCHMARKS`
is on.

It depends on `core`, `launch`, `tensor`, `engine`, `io`, `recording_graph`, `profile` and `profiling`. Its header is included as
`<SushiBLAS/benchmark_harness/benchmark_harness.hpp>`.

| File | Declares |
| --- | --- |
| `benchmark_harness.hpp` | The argument, environment, fill, timing and table helpers the benchmarks share |
| `profile_session.hpp` | `ProfileSession`, which records a benchmark program under SushiProfiler when `SB_BENCH_PROFILE` names a folder |

A benchmark program records on a `RecordingGraph` of its own, from
`modules/engine/recording_graph/README.md`: the operation views of the program record through
`graph.recorder()` and `graph.run()` runs what they recorded and leaves the graph empty. The
module has no graph class of its own, because `RecordingGraph` includes no test framework.

## benchmark_harness.hpp

`print_environment` prints the device name, the build configuration (`release` when `NDEBUG` is
defined, `debug` otherwise), the compute units, cache bytes and cache blocking of the device
profile, the default recording mode and the engine's layout. `time_iterations` runs 2 warm-up
iterations, then times 10 with `std::chrono::steady_clock` and returns the median and the minimum.
`measure(graph, record)` is `time_iterations` over one record call and one `graph.run()`. Each
iteration records on an empty graph, runs it and clears it, so recording, plan compilation and the
run are all inside the time. It does not replay one compiled plan: a graph keeps its nodes and its
plan after a run, and the runtime rebuilds a plan only when the recording changed, so replaying one
recording would time the run alone. The number is the cost of recording and running an operation
once. The environment block prints `record mode:`, the mode a `RecordingGraph` records under unless
a program sets another (`K_DEFAULT_RECORDING_MODE`). The benchmark
section of `tests/README.md` describes the programs that use it.

## Under the profiler

`label_graph(graph, scenario)` names the graph a program owns after the scenario it measures, so the spans of its runs carry the name and `roofline` separates graphs. `print_environment(engine, scenario)` starts a `ProfileSession` when the environment variable `SB_BENCH_PROFILE` names a folder. The session records the runtime's device spans and the library's cost and ceilings facts, marks one period of the timeline `bench.iteration` for each iteration of `time_iterations`, and writes `<scenario>.sprec.json` into the folder when the process ends. Each iteration is also the zone `bench.iteration`. Without the variable the program times exactly what it timed before: a null check and a zone with no collector open.

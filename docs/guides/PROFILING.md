# Profiling a program that uses SushiBLAS

How to record a run with SushiProfiler so each kernel shows its FLOPs, its bytes and how close it ran to the device's limits. Read [the architecture page](../architecture/PROFILING.md) for how the numbers get there.

## Measure the device once

```
sb build --benchmarks
sb bench device_ceilings -- --out build/ceilings
```

Source: not compiled

`device_ceilings` writes `<device>.ceilings.json` into the folder. It needs the GPU for numbers that count: a CPU device's file is marked smoke-only. [The format](../reference/CEILINGS_FORMAT.md) says what the file holds.

## Register the parts and start a session

```cpp
SushiProfiler::Registry registry;
SushiRuntime::RuntimeProfilerParts runtime_parts(registry, runtime);
SushiBLAS::Profiling::SushiBLASProfilerParts blas_parts =
    SushiBLAS::Profiling::register_profiler_parts(registry, runtime, "build/ceilings");

SushiProfiler::ProfilerConfig config;
config.timeline = "step";
config.sources = {SushiProfiler::NamedSettings{"sushiruntime.device",
                                              SushiProfiler::JsonValue::object()}};
config.analyses = {"roofline"};
SushiProfiler::SessionStart started = SushiProfiler::Session::start(config, registry);
```

Source: not compiled

Keep both parts objects until every session started with the registry has stopped. A registry takes one set of SushiBLAS parts; a second `register_profiler_parts` throws `std::invalid_argument`. Mark one period of the timeline per step with `started.session->timeline("step").mark()`, run the graph, and call `stop()` and `report()` at the end.

## Read the table

The `roofline` table has a row per node, with the achieved FLOP/s and bytes/s, the intensity, and, where the device has ceilings in the record, the bound, the share of the bound reached and whether the node is compute or memory limited. A node with no cost is not in the table; the record's spans show it with no `flops` attribute. A device with no ceilings file is stated `unmeasured` in the record's facts and its rows have no bound.

## Name what you record

`TaskRecorder::origin(label)` labels the nodes recorded while its scope lives:

```cpp
const SushiBLAS::TaskRecorder::OriginScope layer = recorder.origin("layer.3");
SushiBLAS::BLASOps(recorder).gemm(a, b, c);
```

Source: not compiled

The span of that GEMM names `layer.3`, so a profile of a network has a row per layer. Scopes nest and restore.

## Without a profiler

Build with `SB_ENABLE_PROFILER=OFF` and every macro of the library expands to nothing and the `profiling` module is not built. A benchmark program records only when `SB_BENCH_PROFILE` names a folder.

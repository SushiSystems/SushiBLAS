# profiling

What SushiBLAS adds to a SushiProfiler registry: an environment contributor that states the library's version and settings and the ceilings of each device, the file those ceilings are read from, and `register_profiler_parts`, the one function a host calls. Its headers are included as `<SushiBLAS/profiling/<name>.hpp>`.

It depends on [`core`](../../foundation/core/README.md), [`launch`](../../foundation/launch/README.md), [`profile`](../../foundation/profile/README.md), SushiRuntime and SushiProfiler. It does not depend on [`engine`](../engine/README.md): the default execution mode is stated as the literal `auto`, which `TaskRecorder` takes when it is given none.

| File | Declares |
| --- | --- |
| `profiler_parts.hpp` | `SushiBLASProfilerParts`, the two `register_profiler_parts` overloads |
| `blas_environment.hpp` | `BlasEnvironment`, the contributor, and `DeviceFacts` |
| `ceilings.hpp` | `DeviceCeilings`, `ComputeCeiling`, `BandwidthCeiling`, `CeilingsMethod` |
| `ceilings_file.hpp` | `read_ceilings`, `write_ceilings`, `ceilings_file_name`, `CeilingsReadStatus` |
| `profiling_log_category.hpp` | `K_PROFILING_LOG_CATEGORY` |

## Registering the parts

```cpp
SushiProfiler::Registry registry;
SushiRuntime::RuntimeProfilerParts runtime_parts(registry, runtime);
Profiling::SushiBLASProfilerParts blas_parts =
    Profiling::register_profiler_parts(registry, runtime, "build/ceilings");
```

Source: not compiled

The registry holds the contributor by reference, so the caller keeps `SushiBLASProfilerParts` until every session started with the registry has stopped. The object is neither copyable nor movable; the function returns it as a prvalue. A registry takes one set of SushiBLAS parts: a second call throws `std::invalid_argument` and adds nothing.

The overload that takes a `Runtime` names each device's track with `SushiRuntime::device_track_name`, the one place that builds the string. The overload that takes a `std::vector<DeviceFacts>` is for a host that names its devices itself and for tests.

## What the contributor states

| Fact | Where | Value |
| --- | --- | --- |
| `sushiblas.version` | build | The library version |
| `sushiblas.profile_level` | build | The deepest zone level the library compiled in |
| `sushiblas.default_mode` | environment | `auto` |
| `sushiblas.gemm_tile_pin` | environment | The value of `SB_GEMM_TILE`, or `unpinned` |
| `sushiblas.ceilings.<track>` | environment | `measured`, `smoke_only`, `unmeasured` or `refused` |
| `device.<track>.peak_flops.<dtype>` | environment | The compute ceiling of the device in that dtype, in FLOP/s, for each dtype in its file |
| `device.<track>.peak_bytes_per_s` | environment | The bandwidth ceiling, in B/s |

The ceiling facts are built with `SushiProfiler::DeviceSpanKeys`, and `<track>` is the name the runtime puts the device's spans on, so the `roofline` analysis finds them. A device with no file, or a file the reader refuses, gets its status fact and no ceiling fact: no analysis judges it against a number nobody measured. A missing file is an `INFO` record and a refused one a `WARN` naming the file and the reason, each once per session start.

## The ceilings file

`<device>.ceilings.json`, named by `ceilings_file_name` from the device's name, in the folder passed to `register_profiler_parts`. [`docs/reference/CEILINGS_FORMAT.md`](../../../docs/reference/CEILINGS_FORMAT.md) holds the format. The `device_ceilings` program writes it with `write_ceilings`; `read_ceilings` validates it and returns a status, never an exception. The module needs SushiProfiler's JSON reader and writer, which is why the reader is here and not in `core`.

# 8. The SushiRuntime dependency

SushiBLAS does not vendor SushiRuntime and does not build it. `cmake/Runtime.cmake` calls
`find_package(SushiRuntime CONFIG REQUIRED)`: SushiRuntime is a prebuilt package, which `sb build`
builds or reuses at the version `cli/sushistack.deps.toml` pins and hands to CMake as
`SushiRuntime_DIR`. SushiLog (`cmake/SushiBLASSushiLog.cmake`) and, with `SB_ENABLE_PROFILER` on,
SushiProfiler (`CMakeLists.txt`) are found the same way. No dependency is added as a
subdirectory and no source of one is compiled in this tree.

Every consumer links the same namespaced target, `SushiRuntime::SushiRuntime`. SushiBLAS
inherits SushiRuntime's `sushiruntime_deterministic_fp` contract through that link. The glossary
calls the step D3.

SushiRuntime is below 1.0.0 and a MINOR release of it may break its interface, so the package
a build finds has to be the one this tree was written against; the pin in
`cli/sushistack.deps.toml` is what keeps them together.

## What SushiBLAS uses of it

SushiBLAS includes `<SushiRuntime/sushiruntime.hpp>` and headers of SushiRuntime's public
modules by their own path, such as `<SushiRuntime/core/span.hpp>` and
`<SushiRuntime/region/buffer_region.hpp>`. It names nothing in `SushiRuntime::Internal`.

| SushiRuntime name | Where SushiBLAS uses it |
| --- | --- |
| `Runtime` | `Engine` and `TensorFactory` take a `Runtime&`, as do `RuntimeScratchSource` and `make_runtime_storage`. `ScratchPool` takes a `ScratchBlockSource&` and `Storage` takes a `StorageBlock`; `RuntimeBlock` allocates or borrows through the runtime. An `Engine` and a `TensorFactory` also have a host form that needs none |
| `Runtime::buffer`, `Buffer<std::byte>`, `Residency`, `DeviceIndex` | `RuntimeBlock` allocates through them, or borrows a buffer another owner keeps alive (`tensor/runtime_block.hpp`) |
| `AdvancedControls` (`Runtime::advanced()`) | `Storage::queue()` and the device count the factory checks |
| `Graph`, `Graph::plan()`, `RunReport` | The program owns the graph, records on it through a `TaskRecorder` and runs it |
| `Recorder::task`, `Access`, `NodeOptions`, `OperationID` | `TaskRecorder` records every tensor operation through them (`engine/task_recorder.hpp`) |
| `Recorder::refusal()` | `TaskRecorder` reads it to report a recording the runtime refused |
| `BufferRegion` | `Tensor::region()` and `OperandSet` |
| `GlobalHandle` | `Storage::global_handle()`, the key of every `BufferRegion` (`tensor/storage.hpp`) |
| `NodeCost` | `node_cost_of`, passed as `NodeOptions::cost` (`engine/task_cost.hpp`) |
| `LivenessToken` | The owner token of a late count's borrowed words (`engine/late_count_keeper.hpp`) |
| `Interop` | `LateCountHold` borrows the words of a late count through `Runtime::interop()` (`engine/late_count_hold.hpp`) |
| `Result`, `Error`, `ErrorCode` | The return of the graph primitives |
| `sushi_ptr`, `RefCounted`, `span` | `Storage` ownership and the span parameters of the public headers |

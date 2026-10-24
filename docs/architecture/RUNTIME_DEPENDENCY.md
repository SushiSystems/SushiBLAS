# 8. The SushiRuntime dependency

SushiBLAS does not vendor SushiRuntime. `cmake/Runtime.cmake` resolves it in three steps, the
pattern the whole stack uses:

1. **A target guard.** If a superproject already defines `SushiRuntime::SushiRuntime`, that
   target is used.
2. **`find_package(SushiRuntime QUIET)`.** An installed package.
3. **A sibling `add_subdirectory`.** `../sushiruntime` beside this checkout, or
   `$SUSHISTACK_HOME/sushiruntime` inside a workspace.

Whichever step succeeds, every consumer links the same namespaced target,
`SushiRuntime::SushiRuntime`. SushiBLAS inherits SushiRuntime's
`sushiruntime_deterministic_fp` contract through that link. The glossary calls this resolution
D3.

No step names a version: `find_package(SushiRuntime QUIET)` takes any installed version and the
sibling step takes the checkout as it is. SushiRuntime is below 1.0.0 and a MINOR release of it
may break its interface, so the SushiRuntime a build finds has to be the one this tree was
written against.

## What SushiBLAS uses of it

SushiBLAS includes `<SushiRuntime/sushiruntime.hpp>` and headers of SushiRuntime's public
modules by their own path, such as `<SushiRuntime/core/span.hpp>` and
`<SushiRuntime/region/buffer_region.hpp>`. It names nothing in `SushiRuntime::Internal`.

| SushiRuntime name | Where SushiBLAS uses it |
| --- | --- |
| `Runtime` | `Engine`, `TensorFactory`, `Storage` and `ScratchPool` are each built on a `Runtime&` |
| `Runtime::buffer`, `Buffer<std::byte>`, `Residency`, `DeviceIndex` | `Storage` allocates through them (`tensor/storage.hpp`) |
| `AdvancedControls` (`Runtime::advanced()`) | `Storage::queue()` and the device count the factory checks |
| `Graph`, `Graph::plan()`, `RunReport` | The program owns the graph, records on it through a `TaskRecorder` and runs it |
| `Recorder::task`, `Access`, `NodeOptions`, `OperationID` | `TaskRecorder` records every tensor operation through them (`engine/task_recorder.hpp`) |
| `Recorder::refusal()` | `TaskRecorder` reads it to report a recording the runtime refused |
| `BufferRegion` | `Tensor::region()` and `OperandSet` |
| `Result`, `Error`, `ErrorCode` | The return of the graph primitives |
| `sushi_ptr`, `RefCounted`, `span` | `Storage` ownership and the span parameters of the public headers |

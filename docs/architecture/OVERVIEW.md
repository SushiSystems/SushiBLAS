# 1. What SushiBLAS is

SushiBLAS is a C++17 tensor and BLAS library on
[SushiRuntime](https://github.com/SushiSystems/SushiRuntime)'s task graph. The scheduler, the
execution engine and the USM allocator are SushiRuntime's. SushiBLAS adds a `Tensor`, an `Engine`
that owns one graph, and eight operation families: dense BLAS, sparse BLAS, elementwise
arithmetic, non-linear activations, logic and comparison, reductions, random generation and
spatial (windowed) operations.

Two commitments shape every operation.

1. **An operation is a recorded task.** `SushiBLAS::BLASOps(engine).gemm(A, B, C)` adds one task
   to the engine's graph, with `A` and `B` declared as reads and `C` as a write, and returns a
   `TaskHandle`. Nothing runs. SushiRuntime's dependency tracker orders the task against every
   other recorded task from those declarations. `engine.execute()` runs the graph, returns when
   every task has finished, and reports the run as a `SushiRuntime::Result<void>`.
2. **No vendor math library.** Every kernel is plain C++ or SushiBLAS's own SYCL. The local-memory
   tiling of GEMM and SYRK derives from portBLAS (Apache-2.0, attributed in
   [`NOTICE.md`](../../NOTICE.md)); the rest is written here, the Philox4x32-10 random generator
   included.

The library is built at C++17, the standard SushiRuntime is built to. The spans in its headers
are `SushiRuntime::span` (`<SushiRuntime/core/span.hpp>`), one class template at every standard,
which at C++20 also converts from and to `std::span`. `tests/package/` is a consumer set to
C++20. The pieces C++20 would supply are stand-ins: SushiLog's `format_text` for `std::format`, and
the value tag `Dispatch::ElementTag` where a templated lambda would go.

A SushiBLAS program starts from SushiRuntime: it creates a `SushiRuntime::Runtime` and builds an
`Engine` on it. SushiBLAS reaches the runtime through `<SushiRuntime/sushiruntime.hpp>` and the
headers of its public modules, and names nothing in `SushiRuntime::Internal`.

In the stack SushiBLAS computes and SushiRuntime orchestrates: the runtime holds the graphs and the
memory and decides where each node runs, and computes nothing of its own. A caller that owns a
graph, such as SushiAI's compiled step, builds a `TaskRecorder` over that graph's recorder and the
operation families over the `TaskRecorder`, and SushiBLAS records into it. Removing the engine's
own graph, so that every caller does this, is
[`LAYERED_EXECUTION.md`](../design/LAYERED_EXECUTION.md).

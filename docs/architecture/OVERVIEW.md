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
   other recorded task from those declarations. `engine.execute()` compiles the graph, runs it and
   returns when every task has finished.
2. **No vendor math library.** Every kernel is plain C++ or SushiBLAS's own SYCL. The local-memory
   tiling of GEMM and SYRK derives from portBLAS (Apache-2.0, attributed in
   [`NOTICE.md`](../../NOTICE.md)); the rest is written here, the Philox4x32-10 random generator
   included.

The library is built at C++17, the standard SushiRuntime is built to. `SushiRuntime::span`
(`core/span.hpp`) is `std::span` at C++20 and SushiRuntime's own type below it, chosen on
`__cplusplus`, and that choice reaches SushiRuntime's compiled ABI. Every span-taking entry point
in the SushiBLAS headers is inline, so the choice is made again in the consumer's translation
unit and never has to agree with a symbol compiled elsewhere. That is what a consumer at C++20
relies on. `tests/package/` is that consumer, set to C++20, and it has not been built against this
tree. The
pieces C++20 would supply are stand-ins: `core/format.hpp` for `std::format`,
`Core::Detail::SourceLocation` for `std::source_location`, and the value tag `Dispatch::ElementTag`
where a templated lambda would go.

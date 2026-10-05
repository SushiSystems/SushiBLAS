# SushiBLAS {#mainpage}

A SYCL tensor and BLAS engine built on [SushiRuntime](https://github.com/SushiSystems/SushiRuntime).
Every operation records a task on a graph rather than executing immediately, so dependencies are
tracked by byte range and the runtime decides the order. Determinism is a contract: reductions use
hand-written fixed-order folds, never `sycl::reduction`.

This is the generated API reference. It documents what the headers say; the reasoning behind the
design lives in the prose docs, and those are the better place to start:

- `README.md` — what the library is and how to build it.
- `docs/architecture/ARCHITECTURE.md` — the engine, the dispatch seam and the kernel layers.
- `docs/guides/INTEGRATION.md` — consuming SushiBLAS from your own CMake project.
- `cli/README.md` — every `sb` command.
- `docs/CONTRIBUTING.md` — how a change is built, tested and documented.

## Where to start reading

| If you want to | Read |
|---|---|
| Hold data | `SushiBLAS::Tensor`, `SushiBLAS::Storage` — `tensor.hpp`, `storage.hpp` |
| Call an operation | `SushiBLAS::Engine` — `engine.hpp` and the family headers under `engine/` |
| Fuse several ops into one kernel | `SushiBLAS::Dispatch` — `engine/dispatch.hpp` and the five `*/dispatch.hpp` |
| Attach an epilogue to a GEMM | `SushiBLAS::Dispatch::execute_gemm` — `engine/blas/dispatch.hpp` |
| Write a deterministic reduction | `SushiBLAS::Reduce` — `engine/reduce_1d.hpp`, `engine/reduce_axis.hpp` |

## Not in this reference

`include/SushiBLAS/experimental/` is excluded. Decision D1 quarantined the FFT and linear-algebra
headers: they are outside the umbrella header and not every declared op has a definition, so
listing them here would advertise an API that does not link.

## A caution about the reference

Doxygen lists every public symbol, which flattens a deliberate hierarchy. Two headers carry
warnings that matter more than their signatures do:

- `engine/blas/dispatch.hpp` — why an epilogue's extra operands are declared as a `Tensor` and
  never as a `ResourceRegion`. A region converts implicitly from a bare pointer, and a bare pointer
  widens to the whole allocation, so two views of one arena become unrelated resources and **no
  dependency edge is emitted**. That bug has cost this stack two silently wrong training runs.
- `engine/math/reductions.hpp` — why a count is an `INT64` and not a value in the input's own dtype.

# Layered execution

**Status:** Open — designed and ruled on 2026-10-19; nothing built
**About:** SushiBLAS's share of the stack's layered execution: the library becomes the one door to computation for SushiAI and SushiEngine.
**Modules:** modules/engine/engine

## Problem

SushiBLAS's share of the stack's layered execution: the library becomes the one door to
computation for SushiAI and SushiEngine, records only into the graph its caller owns, offers a
per-index seam with declared reads and writes, gains two general operations, and carries the
tensor side of a training run over several machines.

Each repository of the stack holds its own share in a document of this name:

| Repository | Its share |
| --- | --- |
| SushiRuntime | Placement class and key, scopes, topology, placement rules, the distributed layer's needs |
| SushiBLAS | This document |
| SushiAI | Data-parallel training over five machines; it also holds the programme's backlog entry |
| SushiEngine | Two lanes behind one contract, SushiBLAS in the runtime lane; placement scopes; physics islands on several machines in the runtime lane |

What exists in the source, read on 2026-10-19:

Read from the source on 2026-10-19.

- `Engine` (`modules/engine/engine/include/SushiBLAS/engine/engine.hpp`) holds the runtime, a
  `ScratchPool`, a `RecordedGraph`, a `TaskRecorder` over that graph's recorder, a
  `TensorFactory` and an `RNGStream`. `execute()` returns `graph_.run()`, and
  `discard_recorded()` drops what was recorded.
- `TaskRecorder` already records on any `SushiRuntime::Recorder` it is built over:
  `TaskRecorder(SushiRuntime::Recorder&, ScratchPool&, ExecutionMode = Auto)`. Its private
  `insert()` is the one caller of `Recorder::task()`.
- Every operation family is a view over a `TaskRecorder`: `BLASOps` has a constructor that
  takes `Engine&` and one that takes `TaskRecorder&` (`blas/blas.hpp`); so do the elementwise,
  sparse, spatial, logic, reductions, nonlinear and random families.
- SushiAI's `CompiledStep` already works the way this document asks of everyone: it records on
  its own `SushiRuntime::Graph` through a `TaskRecorder` built over that graph's recorder, and
  borrows only the engine's scratch pool, mode and random stream
  (`sushiai/modules/core/graph/source/compiled_step.cpp`).
- What calls `Engine::execute()` or `discard_recorded()`: `IO` through two helpers at nine call
  sites (`operations/io/source/io.cpp`), the benchmark harness (`benchmark_harness.hpp`), six
  example programs, the package test and about fifteen test files; in SushiAI,
  `run_recorded()` and `discard_recorded()` of `modules/core/execution` and four test files.
  `IO` takes an `Engine&` (`io/io.hpp`), which is why every `IO` call runs the whole recorded
  graph first (RW-4).
- `functor_map` holds the per-element seam: `MapTask` (one output from some inputs), `ZipTask`
  (N tensors advanced together), `bind_scalars`, and `bind_reads` with `read_as_element` and
  `read_as<U>`. A functor writes only its own element. Its headers are installed, but no page
  says they are a public surface.
- No 1-D convolution exists outside the quarantined `experimental/signal` (FFT only); no
  weighted sum of indexed transforms exists; `gemm` batches above rank 2 and nothing batches a
  `gemv` (RW-45).
- The build pins no SushiRuntime version (KI-26, RW-89).

## Decision

### Roles

| Layer | Role | Never |
| --- | --- | --- |
| SushiRuntime | The one scheduler: graphs, memory, dependencies, placement, transport | Computes |
| SushiBLAS | The one door to computation: operations recorded into the recorder it is handed | Runs a graph, orders work, picks a device or a machine |
| SushiAI, SushiEngine | Own a graph per step or per frame, open placement scopes, combine their formulas with SushiBLAS's operations | Write a device kernel outside SushiBLAS, touch a transport |
| The person running it | Writes the placement rules | |

### The owner's rulings

Made on 2026-10-19.

Made on 2026-10-19.

| # | Subject | Ruling |
| --- | --- | --- |
| 1 | The engine's graph | `Engine` keeps no graph and has no `execute()`. Every operation records into a recorder its caller owns |
| 2 | General operations | The computations SushiEngine writes by hand today and that other fields can use enter SushiBLAS under general names, never as a field's kernel |
| 3 | Field formulas | A formula of a field, such as a physics constraint projection, stays in its field and runs through a SushiBLAS per-index seam whose reads and writes are declared |
| 4 | Placement | SushiBLAS stamps no placement of its own; its nodes take the class and key of the caller's scope |
| 5 | Grain across machines | SushiBLAS does not offload its operations one at a time |
| 6 | Gradient types | The exchange of a training run works for every data type |
| 7 | Order | SushiRuntime's share comes first, SushiBLAS's second, SushiAI's third |

### SushiEngine's native lane

SushiEngine builds a lane without SYCL for the platforms SushiRuntime cannot reach. The owner
ruled on 2026-10-20 that the lane stays fully supported with a runtime and a BLAS of the engine's
own, bound through the same seams as SushiRuntime and SushiBLAS, and without distributed support.
SushiBLAS gains no build without SYCL for it. What SushiBLAS keeps doing for that lane is what it
does today: its SYCL-free value-type and fold headers, such as `BLAS::fold_fixed_order`, are
included by the engine's native backend, so a change to them is checked against SushiEngine's
native CI job.

## Design

### An engine without a graph

`Engine` keeps what is per process: the runtime, the scratch pool, the tensor factory, the
random stream, the execution mode and the layout. It loses `RecordedGraph`, its `TaskRecorder`,
`execute()`, `discard_recorded()` and `recorder()`.

A caller that owns a graph builds a `TaskRecorder` over that graph's recorder and the engine's
scratch pool, and builds the families over it, as `CompiledStep` does:

```cpp
SushiRuntime::Graph graph = runtime.graph();
SushiBLAS::TaskRecorder recorder(graph.recorder(), engine.scratch_pool(), engine.mode());
SushiBLAS::BLASOps blas(recorder);
blas.gemm(a, b, c);
graph.run();
```

The family constructors that take `Engine&` go with the engine's recorder. A caller that wants
one graph for a few operations, a test or an example, writes those four lines; the tests share a
fixture that holds them.

`IO` stops taking an `Engine`. A read or a save is a host action on a tensor whose writer has
run, and running it is the caller's business: `IO` reads what the storage holds when it is
called. RW-4 closes with this change.

### The per-index seam

```cpp
SushiBLAS::IndexOps index(recorder);
index.for_each(n,
    SushiBLAS::bind_writes(
        SushiBLAS::bind_reads(projection, read_as<Constraint>(constraints), read_as_element(inverse_mass)),
        write_as_element(positions)),
    SushiBLAS::OperationOptions{"pgs_project", SushiRuntime::Determinism::Bitwise});
```

`IndexOps::for_each(count, functor, options)` records one task that calls `functor(i, reads...,
writes...)` for each `i` below `count`, on the device or on the host by the execution mode, with
the same functor on both paths as `MapTask` does. `bind_writes` is the counterpart of
`bind_reads`: it checks each tensor, declares it as a write region on the task and hands the
functor a typed pointer. A tensor named in both is declared once as read and written.

Two things the seam cannot check are part of its contract and written on it:

- **Disjoint writes.** Two indices of one call may write the same element only if the caller
  orders them, as a constraint colouring does. The seam declares whole tensors as written; it
  does not see which element an index writes. Under `ForceReference` the indices run in order on
  the host, which gives the result a test compares the device against.
- **A count known late.** A call may take a capacity and a count read at run time instead of a
  fixed count, for the ECS chunks and matter passes of SushiEngine whose size a previous node
  writes. It records through the runtime's dynamic node.

`OperationOptions` (`engine/operation_options.hpp`), which `IndexOps`, `SignalOps` and `BlendOps`
all take and check through one engine function, carries the node's name and a `SushiRuntime::Determinism` (`Unstated`, the
default, `Bitwise`, `Tolerant` or `Cosmetic`), which rides the task description into the runtime
node's `NodeOptions::determinism`, so a caller's contract travels with the node; the owner ruled
on 2026-10-24 that the class lives on the runtime node. A late count is a `UInt32` or `Int32`
tensor in host memory (SushiEngine's counts are `uint32`, B15), read by the runtime's dynamic node
through a buffer borrowed over the tensor's storage; an `Int64` count is refused rather than
narrowed.

`IndexOps` is a new operation family in `operations/index/`, beside the index operations RW-94
asks for. `write_binding.hpp` and `index_task.hpp` join `functor_map`.
`docs/design/PUBLIC_SEAM/README.md`'s list of what is public gains `functor_map`'s bindings.

### Two general operations

| Operation | Computes | Its first user |
| --- | --- | --- |
| `SignalOps::convolve_1d(signal, taps, out[, options])` | For each row, `out[n] = Σ_k signal[n + K - 1 - k] · taps[k]`, valid mode, rows batched over the leading dimensions | SushiEngine's audio FIR, which feeds its history as the first `K - 1` samples |
| `BlendOps::blend_transforms(points, transforms, indices, weights, out[, options])` | For each point `p`, `out[p] = Σ_j weights[p][j] · transforms[indices[p][j]] · points[p]`, with 4×4 or 3×4 transforms | SushiEngine's linear blend skinning; morph and attachment blends after it |

Both have a serial reference path and a SYCL path, and take `Half`, `Float32` and `Float64`. Both
take a trailing `OperationOptions` for the node's name and determinism; without it the node is
named after the operation and `Unstated`. The blend's indices may be `Int32`, `Int64` or `UInt32`.
Dual-quaternion skinning has no general form and goes through the per-index seam.

### The cluster module

The tensor side of a training run over several machines. The design is in SushiAI's
`docs/design/LAYERED_EXECUTION/README.md`, section 6; this is what lands here.

```
modules/engine/cluster/                  built only when SB_ENABLE_DISTRIBUTED is on
    include/SushiBLAS/cluster/
        tensor_pack.hpp                  TensorPack
        cluster_binding.hpp              ClusterBinding
        remote_task.hpp                  record_remote(), bind_views()
modules/operations/elementwise/source/
    ordered_sum.cpp                      ElementwiseOps::ordered_sum()
```

| Brick | Does one thing |
| --- | --- |
| `TensorPack` | Lays an ordered list of tensors out in one flat buffer, one segment per type, and records the gather into it and the scatter out of it; hands out `tensor(index)` and `segment_tensor(dtype)` views over its buffer, so a sum reads the packs in place; its fingerprint hashes every offset, type, shape and type size |
| `ClusterBinding` | Holds the cluster and the kernel registry one engine works with and keeps a pack's buffer shared for the pack's life; it exposes only what the bricks use, never the whole `Cluster` |
| `record_remote()` | Records a registered task over packs: the tensor form of the runtime's `record_registered_task()` |
| `bind_views()` | On a worker, checks each `BufferView` against a pack's layout and hands the segments back typed |
| `ordered_sum()` | Adds `n` tensors of one shape and type in index order, each with a weight, in one kernel |

`ordered_sum()` accumulates `Half` in `Float32` and rounds once after the last input; `Float32`,
`Float64` and both complex types accumulate in their own type; `Int32` and `Int64` add exactly
and wrap, and refuse a weight other than 1.

### Placement

SushiBLAS adds nothing to a node's placement. A node it records inside the runtime's
`Recorder::within(scope, ...)` carries the scope's class and key, because `TaskRecorder::insert()`
builds `NodeOptions` without them and the runtime stamps every node that has none. A test pins
this.

## Phases

Hierarchy first: what callers stand on before the callers move, and the engine's graph goes only
when nothing calls it. Each task has one acceptance criterion and one build. The manual pages a
task makes false (`docs/architecture/ENGINE_AND_RECORDER.md`, the module READMEs, the examples
README) change in its own commit.

| Id | Phase | State |
| --- | --- | --- |
| B1 | A test fixture that owns a runtime graph and a `TaskRecorder` over it; `GraphProbe` rebuilt on it instead of `RecordedGraph`. Done when: every test that uses `GraphProbe` passes with the probe holding no `RecordedGraph`. | building |
| B2 | Tests, examples and the package test record on their own graph. Done when: no file outside `modules/engine/engine/` calls `Engine::execute()` or `discard_recorded()`, and the suites pass. Waits on B1. | building |
| B3 | `IO` without an engine: host actions on a tensor whose writer has run. Done when: `IO`'s header names no `Engine`, its tests pass with the caller running the graph, RW-4 closes. | building |
| B4 | The benchmark harness times a caller's graph. Done when: the harness names no `Engine::execute()` and its benchmarks run. Waits on B1. | building |
| B5 | `Engine` without `RecordedGraph`, `TaskRecorder`, `execute()`, `discard_recorded()`, `recorder()`; the family constructors over `Engine&` removed. Done when: `engine.hpp` declares none of them, every suite passes, and the seam checker forbids them coming back. Waits on B2, B3, B4, `sushiai` A0. | building |
| B6 | `bind_writes`, `write_as_element`, `write_as<U>`. Done when: a test declares a read and a write on one task and the runtime orders a later reader after it. | building |
| B7 | `IndexOps::for_each` with a fixed count and with a late count; `OperationOptions` with the determinism class. Done when: a scatter kernel over a coloured batch equals its `ForceReference` run bit for bit, and a late-count call runs the count a previous node wrote. Waits on B6. | building |
| B8 | `SignalOps::convolve_1d`, in a family of its own (`modules/operations/signal/`, `modules/kernels/signal_kernels/`): the owner ruled on 2026-10-24 that `SpatialOps` stays the operations over an NHWC grid. Done when: each of the three float types equals the serial reference within the test's stated tolerance. | building |
| B9 | `BlendOps::blend_transforms`, in a family of its own (`modules/operations/blend/`, `modules/kernels/blend_kernels/`), as the owner ruled on 2026-10-24 for the same reason as B8. Done when: each of the three float types equals the serial reference within the test's stated tolerance. | building |
| B10 | A node recorded inside a runtime scope carries its class and key. Done when: the test reads the scope's class and key from a SushiBLAS node. Waits on `sushiruntime` R2. | building |
| B11 | `ordered_sum()`. Done when: it equals a serial sum bit for bit for each of the seven types, and refuses a weighted integer sum. | building |
| B12 | The `cluster` module and `SB_ENABLE_DISTRIBUTED`. Done when: a pack of every type round-trips through a worker on `InProcessFabric` with its fingerprint checked. Waits on B11. | building |
| B13 | The runtime pin moves to the tag that carries R1 to R10. Done when: the build resolves that tag; KI-26 closes with RW-89. Waits on `sushiruntime` R10. | planned |
| B14 | A `Storage` that borrows a `SushiRuntime::Buffer<std::byte>` another owner keeps alive, so SushiEngine's BLAS adapter hands operations tensors over its own buffers with no allocation or copy; a task watches the owner. Ruled by the owner on 2026-10-24 for `sushiengine` E5. Done when: a tensor over a borrowed storage records and runs, and a run after the owner lets go is refused as `DanglingResource`. | building |
| B15 | A `UInt32` data type, which the late count of `IndexOps::for_each` takes as it is: SushiEngine's late count is a `uint32` buffer. Ruled by the owner on 2026-10-24 for `sushiengine` E5. Done when: a late-count `for_each` runs over a `UInt32` count, and every operation either takes the type or refuses it in words. | building |
| B16 | `TensorPack` hands out a `Tensor` view of one entry and a 1-D view of one segment, over its own buffer, so SushiAI's exchange sums result packs with `ordered_sum()` in place instead of copying them out first. Ruled by the orchestrator on 2026-10-24 for `sushiai` A3. Done when: each entry view reads the bytes the gather wrote, each segment view of the eight types has the segment's size and offset, and an `ordered_sum()` over the same segment of three packs into a fourth gives the sum. | building |
| B17 | `SignalOps::convolve_1d` and `BlendOps::blend_transforms` take the node's name and determinism, through `OperationOptions`, which moves to the engine module with one shared check; the blend's indices may be `UInt32`. Ruled by the orchestrator on 2026-10-24 for `sushiengine` E5: the engine's adapter keeps every verb's name and determinism on its node. Done when: both nodes carry the name and class a caller gives, a call without options keeps the operation's name and `Unstated`, and the three index types give equal bits. | building |
| B18 | Each graph primitive (`reduce`, `segmented_reduce`, `prefix_scan`, `counting_sort`) takes a trailing window that places each buffer operand at an element offset, declared as that interval of the whole buffer. Ruled by the orchestrator on 2026-10-24 for `sushiengine` E5: SushiRuntime keys an access on its buffer's base, so a borrowed sub-range is never ordered against its parent, and the engine's two lanes must take the same windows. Done when: a window past 0 equals a compact buffer bit for bit, a node over a disjoint window is not ordered after it and an overlapping one is, and a window past its buffer is refused. | building |

B5 waits on SushiAI's A0, which moves SushiAI's two execution helpers and four tests off the
engine's graph. B6 to B9 have disjoint files and could share a wave; the owner chooses inline or
waves.

### Work

<!-- work:begin -->
- [2026_10_24_LAYERED_EXECUTION](../../agent/2026_10_24_LAYERED_EXECUTION/PLAN.md): B14, B15, B16, B17, B18, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13
<!-- work:end -->

## Open questions

- Does the plan run inline or in waves? The owner chooses.

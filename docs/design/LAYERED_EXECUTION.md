# Layered execution
**Status:** Open — designed and ruled on 2026-10-19; nothing built

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
| SushiEngine | Every kernel through SushiBLAS, placement scopes, physics islands on several machines |

## 1. Roles

| Layer | Role | Never |
| --- | --- | --- |
| SushiRuntime | The one scheduler: graphs, memory, dependencies, placement, transport | Computes |
| SushiBLAS | The one door to computation: operations recorded into the recorder it is handed | Runs a graph, orders work, picks a device or a machine |
| SushiAI, SushiEngine | Own a graph per step or per frame, open placement scopes, combine their formulas with SushiBLAS's operations | Write a device kernel outside SushiBLAS, touch a transport |
| The person running it | Writes the placement rules | |

## 2. The owner's rulings

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

## 3. What exists

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
  graph first (KI-7).
- `functor_map` holds the per-element seam: `MapTask` (one output from some inputs), `ZipTask`
  (N tensors advanced together), `bind_scalars`, and `bind_reads` with `read_as_element` and
  `read_as<U>`. A functor writes only its own element. Its headers are installed, but no page
  says they are a public surface.
- No 1-D convolution exists outside the quarantined `experimental/signal` (FFT only); no
  weighted sum of indexed transforms exists; `gemm` batches above rank 2 and nothing batches a
  `gemv` (RW-45).
- The build pins no SushiRuntime version (KI-26, RW-89).

## 4. The design

### 4.1 An engine without a graph

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
called. KI-7 closes with this change.

### 4.2 The per-index seam

```cpp
SushiBLAS::IndexOps index(recorder);
index.for_each(n,
    SushiBLAS::bind_writes(
        SushiBLAS::bind_reads(projection, read_as<Constraint>(constraints), read_as_element(inverse_mass)),
        write_as_element(positions)),
    SushiBLAS::IndexOptions{"pgs_project", SushiBLAS::Determinism::Bitwise});
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

`IndexOptions` carries the node's name and a determinism class with SushiEngine's three values,
`Bitwise`, `Tolerant` and `Cosmetic`, so a caller's contract travels with the node.

`IndexOps` is a new operation family in `operations/index/`, beside the index operations RW-94
asks for. `write_binding.hpp` and `index_task.hpp` join `functor_map`.
`docs/design/PUBLIC_SEAM.md`'s list of what is public gains `functor_map`'s bindings.

### 4.3 Two general operations

| Operation | Computes | Its first user |
| --- | --- | --- |
| `SpatialOps::convolve_1d(signal, taps, out)` | For each row, `out[n] = Σ_k signal[n + K - 1 - k] · taps[k]`, valid mode, rows batched over the leading dimensions | SushiEngine's audio FIR, which feeds its history as the first `K - 1` samples |
| `SpatialOps::blend_transforms(points, transforms, indices, weights, out)` | For each point `p`, `out[p] = Σ_j weights[p][j] · transforms[indices[p][j]] · points[p]`, with 4×4 or 3×4 transforms | SushiEngine's linear blend skinning; morph and attachment blends after it |

Both have a serial reference path and a SYCL path, and take `Half`, `Float32` and `Float64`.
Dual-quaternion skinning has no general form and goes through the per-index seam.

### 4.4 The cluster module

The tensor side of a training run over several machines. The design is in SushiAI's
`docs/design/LAYERED_EXECUTION.md`, section 6; this is what lands here.

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
| `TensorPack` | Lays an ordered list of tensors out in one flat buffer, one segment per type, and records the gather into it and the scatter out of it; its fingerprint hashes every offset, type, shape and type size |
| `ClusterBinding` | Holds the cluster and the kernel registry one engine works with and keeps a pack's buffer shared for the pack's life; it exposes only what the bricks use, never the whole `Cluster` |
| `record_remote()` | Records a registered task over packs: the tensor form of the runtime's `record_registered_task()` |
| `bind_views()` | On a worker, checks each `BufferView` against a pack's layout and hands the segments back typed |
| `ordered_sum()` | Adds `n` tensors of one shape and type in index order, each with a weight, in one kernel |

`ordered_sum()` accumulates `Half` in `Float32` and rounds once after the last input; `Float32`,
`Float64` and both complex types accumulate in their own type; `Int32` and `Int64` add exactly
and wrap, and refuse a weight other than 1.

### 4.5 Placement

SushiBLAS adds nothing to a node's placement. A node it records inside the runtime's
`Recorder::within(scope, ...)` carries the scope's class and key, because `TaskRecorder::insert()`
builds `NodeOptions` without them and the runtime stamps every node that has none. A test pins
this.

## 5. Plan

Hierarchy first: what callers stand on before the callers move, and the engine's graph goes only
when nothing calls it. Each task has one acceptance criterion and one build. The manual pages a
task makes false (`docs/architecture/ENGINE_AND_RECORDER.md`, the module READMEs, the examples
README) change in its own commit.

| Task | Does | Files, mainly | Waits on | Done when |
| --- | --- | --- | --- | --- |
| B1 | A test fixture that owns a runtime graph and a `TaskRecorder` over it; `GraphProbe` rebuilt on it instead of `RecordedGraph` | `engine/graph_probe`, the test environment | | Every test that uses `GraphProbe` passes with the probe holding no `RecordedGraph` |
| B2 | Tests, examples and the package test record on their own graph | `tests/`, `applications/examples/`, `tests/package/` | B1 | No file outside `modules/engine/engine/` calls `Engine::execute()` or `discard_recorded()`, and the suites pass |
| B3 | `IO` without an engine: host actions on a tensor whose writer has run | `operations/io` | | `IO`'s header names no `Engine`, its tests pass with the caller running the graph, KI-7 closes |
| B4 | The benchmark harness times a caller's graph | `operations/benchmark_harness` | B1 | The harness names no `Engine::execute()` and its benchmarks run |
| B5 | `Engine` without `RecordedGraph`, `TaskRecorder`, `execute()`, `discard_recorded()`, `recorder()`; the family constructors over `Engine&` removed | `engine/engine`, the operation families, `tools/` seam checker | B2, B3, B4, `sushiai` A0 | `engine.hpp` declares none of them, every suite passes, and the seam checker forbids them coming back |
| B6 | `bind_writes`, `write_as_element`, `write_as<U>` | `engine/functor_map/write_binding.hpp` | | A test declares a read and a write on one task and the runtime orders a later reader after it |
| B7 | `IndexOps::for_each` with a fixed count and with a late count; `IndexOptions` with the determinism class | `operations/index/`, `engine/functor_map/index_task.hpp` | B6 | A scatter kernel over a coloured batch equals its `ForceReference` run bit for bit, and a late-count call runs the count a previous node wrote |
| B8 | `SpatialOps::convolve_1d` | `operations/spatial` | | Each of the three float types equals the serial reference within the test's stated tolerance |
| B9 | `SpatialOps::blend_transforms` | `operations/spatial` | | Each of the three float types equals the serial reference within the test's stated tolerance |
| B10 | A node recorded inside a runtime scope carries its class and key | `tests/engine/engine/` | `sushiruntime` R2 | The test reads the scope's class and key from a SushiBLAS node |
| B11 | `ordered_sum()` | `operations/elementwise` | | It equals a serial sum bit for bit for each of the seven types, and refuses a weighted integer sum |
| B12 | The `cluster` module and `SB_ENABLE_DISTRIBUTED` | `engine/cluster/`, `cmake/`, `cli/` | B11 | A pack of every type round-trips through a worker on `InProcessFabric` with its fingerprint checked |
| B13 | The runtime pin moves to the tag that carries R1 to R10 | `cli/sushistack.deps.toml` | `sushiruntime` R10 | The build resolves that tag; KI-26 closes with RW-89 |

B5 waits on SushiAI's A0, which moves SushiAI's two execution helpers and four tests off the
engine's graph. B6 to B9 have disjoint files and could share a wave; the owner chooses inline or
waves.

## 6. Open

- **SushiEngine's native lane.** SushiEngine builds a lane without SYCL for platforms SushiRuntime
  cannot reach, which runs its nodes on a thread pool (`sushiengine`'s execution README). Every
  kernel of SushiEngine going through SushiBLAS, which needs SYCL and SushiRuntime, leaves that
  lane no way to compute. SushiEngine's `LAYERED_EXECUTION.md` asks the owner; one answer is a
  SushiBLAS build without SYCL that keeps only the serial reference paths, which would be a
  task of this document.

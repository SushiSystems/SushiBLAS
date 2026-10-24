# Layered execution in SushiBLAS: report

**Status:** Open — B1, B2, B3, B6, B8, B9, B11 and B12 landed on 2026-10-24; not built yet.

## B1

`Test::RecordingFixture` (`modules/engine/engine_fixture/`) owns a `SushiRuntime::Graph` and a
`TaskRecorder` over its recorder, the engine's scratch pool and mode. `GraphProbe` holds one and
no `RecordedGraph`; its public interface is unchanged, so no test changed. `run()` empties the
graph after a run that succeeded and keeps it after one turned away with `InvalidState`, as
`RecordedGraph::run()` did.

Verification: `clang++ -fsyntax-only` of the probe's test, the fixture's test and three probe
users in other modules; the checkers under `tools/` pass. The other 16 probe users were not
syntax-checked. Not built: the owner ruled that each repository is built once at the end.

## B2

The tests record through `SushiBLASTest::recorder()`, a `RecordingFixture` the fixture holds
over its engine, and run with `recording->run()`; `set_mode()` sets the mode of the engine and
of the recorder together. The examples and the package test write the design's four lines and
run `graph.run()`. A runtime graph keeps its nodes after a run and refuses an empty run, so the
examples that record twice call `graph.clear()`.

Left for later tasks: `IO` and its tests (B3), the benchmarks (B4), and the tests whose subject
is `Engine::execute()` or `discard_recorded()` (B5 deletes them). About 120 of the mechanically
migrated test files were not syntax-checked; 38 were, one or more per changed folder.

## B6

`write_binding.hpp` mirrors `read_binding.hpp`. `bind_writes(bind_reads(f, reads...), writes...)`
calls `f(i, reads..., writes..., scalars...)`; the reverse nesting does not compile.
`WriteBound::declare_into` runs after the reads are declared: a tensor already read gains one
write region, a repeated write is skipped, and a write that overlaps a read region without being
it is refused. `AddressedOp::typed` calls a qualified `Detail::narrow_for`, so `scalar_binding.hpp`
now narrows any op with a `narrow<T>()` member. The accessors `element_dtype()` and `IsWriteBound`
that B7 needs land with this change, in the same two headers.

Seen and kept: `write_dtype` duplicates `read_dtype` in a few lines.

## B11

`ordered_sum(span<const Tensor> inputs, span<const double> weights, Tensor& out)` runs one
functor on both paths: each element reads input 0 to n-1 in order, so the work-group size cannot
change the result. `Half` accumulates in `float` and rounds once; a complex input takes its real
weight per component, so no complex product enters the rounding; integers add through
`wrapping_add` and refuse a weight other than 1. The kernel carries the input addresses and
weights by value, which caps n at `K_MAX_ORDERED_SUM_INPUTS` (32); a scratch table would lift
the cap.

## B8

`SignalOps::convolve_1d` lives in `modules/operations/signal/`, with its kernels in
`modules/kernels/signal_kernels/`: a first version went into `SpatialOps`, and the owner moved
it to a family of its own on 2026-10-24. The SYCL kernel runs one work-item per output sample
over `Device::plan_1d` and accumulates in `Accumulator<T>` (`Half` in `float`), k ascending; it
is not tiled, for want of a measurement. The reference kernel sums in double. The family has a
constructor over `TaskRecorder&` only. Its names stay apart from the quarantined
`TransformsOps` of `experimental/signal`.

Seen and kept: the spatial and signal tests copy the same helper block (`tensor_from`,
`elements_from`, `residue_pattern`); the spatial_kernels README names dependencies its CMake
does not list.

## B9

`BlendOps::blend_transforms` lives in `modules/operations/blend/` with its kernels in
`modules/kernels/blend_kernels/`, shaped as `SignalOps` is; the owner ruled it out of
`SpatialOps` on 2026-10-24. A `[P,3]` point is a position (w = 1), a `[P,4]` point is homogeneous
and its output holds w; a 3×4 transform has the implicit row (0,0,0,1). Indices are INT32 or
INT64; weights need not sum to one. One work-item per point, j ascending then component
ascending, `Half` accumulating in `float`. An index outside `[0, T)` adds no term in either mode,
since device-resident indices cannot be checked when the call is recorded.

Seen and kept: signal and blend repeat the dense-rows helper and the kernel structure as the
family pattern.

## B12

`modules/engine/cluster/` builds under `SB_ENABLE_DISTRIBUTED`, which also turns on SushiRuntime's
distributed layer. A pack lays its tensors out in one segment per type, in `Core::DataType`
order, each non-empty segment on a 128-byte boundary; `PackLayout` is its own brick, so a worker
builds a layout from shapes alone. The fingerprint is a 64-bit FNV-1a over a fixed little-endian
byte sequence (version, entry count, per entry type, type size, layout, rank, extents, offset and
size, then each segment and the total); a big-endian host fails to compile. The operation name of
a remote task is `"<name>:<16 hex digits of the fingerprint>"`, so a worker of another layout
never advertises it, and `bind_views()` refuses a view count or a size it does not expect. A pack
allocates shared USM on device 0, so `Cluster::share()` accepts it; gather and scatter record
through `TaskRecorder::record_pair`. `record_remote()` takes the runtime recorder, since
`TaskRecorder` has no registered-task path, and cannot keep packs alive: their life is a
precondition. `ordered_sum()` is not used here; the gradient sum is SushiAI's.

Seen and kept: `PackSpan::as<T>()` uses `Detail::HostDataType` from `tensor`'s internal
namespace; `tensor_pack.cpp` casts away the const of a scatter target's data. No fingerprint
golden is pinned yet: the first run gives the value.

## B3

`IO` needed nothing in place of the engine: it copied through each tensor's own queue and used
the engine only to run its graph first. It now has no constructor and no member. A host or shared
tensor is read in place; a device tensor is copied with `queue.memcpy(...).wait()` on the event
the copy returns. The three host-transfer tests that checked the flush now record, run and read in
the caller's order. `IOStatus::RunFailed`, which nothing returns any more, is removed by the
owner's ruling of 2026-10-24. RW-4 keeps only the split by format.

SushiAI's two checkpoint savers never call `IO`; their unread `Engine&` can go (A0 left it for
this change).

## Dispatches

```
dispatch: B1 — sonnet medium, 31 turns, passed, rung 1
dispatch: B2 — sonnet medium, 112 turns, passed, rung 1
dispatch: B6 — sonnet medium, 46 turns, passed, rung 1
dispatch: B11 — sonnet medium, 48 turns, passed, rung 1
dispatch: B9 — sonnet medium, 77 turns, passed, rung 1
dispatch: B12 — opus low, 99 turns, passed, rung 1
dispatch: B3 — sonnet medium, 53 turns, passed, rung 1
dispatch: B8 — sonnet medium, 55 turns, then B8b sonnet medium, 28 turns, passed, rung 1
```

# Profiling and the cost of a task

What SushiBLAS tells SushiProfiler about the work it records, and how a span of a profile is tied to a number of operations and to a layer of a network. The design behind it is [PROFILING](../design/PROFILING/README.md). This page describes the tree as it is.

## The path of a cost

1. A seam states facts it already holds in a `CostShape`: the extents of a GEMM, the element count of an elementwise operation, the window of a pool. It sets `TaskDescription::shape` and records.
2. `TaskRecorder::insert_node` calls `cost_of(description.operation, description.shape)`. The table of the `cost` module returns FLOPs and a dtype class, or the reason the task states none.
3. The recorder adds the bytes the operands declared (`OperandSet::declared_bytes`) and passes the result to SushiRuntime as `NodeOptions::cost`, mapping the dtype to the runtime's class.
4. The runtime keeps the cost with the node, through plan rebuilds and replays, and its device source writes `flops`, `bytes` and `dtype` on the span of each dispatch.
5. SushiProfiler's `roofline` reads those attributes and the ceiling facts of the record and states, per node, the achieved rate and its share of the bound.

A task with no cost carries no `NodeCost`, so its span has no attributes and `roofline` skips it. A late-count task is such a task, because its width is read on the device at run time.

## Several kernels, one span

The runtime times the event a body returns. A body that submits several kernels returns a `Device::SubmittedEvents`; the recorder hands each event to the runtime with `note_dispatch_event` and returns the last. The dispatch then runs from the first kernel's start to the last kernel's end, and FLOPs over that span is the rate of the whole body. [The cost README](../../modules/foundation/cost/README.md) lists which bodies do this and which do not yet.

## Facts and ceilings

`register_profiler_parts` adds a contributor to a SushiProfiler registry. It states the library's version and settings and, per device, the ceilings from `<device>.ceilings.json`, keyed by the track name the runtime puts the device's spans on. [The ceilings file](../reference/CEILINGS_FORMAT.md) is written by `device_ceilings`, which measures through a graph run observed on the device clock. [The profiling guide](../guides/PROFILING.md) shows the use.

## Tying a span to a layer

A record call adds exactly one rule to the runtime graph. A caller that lowers an IR node reads `graph.node_count()` before and after and maps the range through `Graph::rule_registration`. SushiBLAS also labels nodes: `TaskRecorder::origin(label)` opens a scope, and every node recorded while it lives carries the label as its origin, which the span names. A graph can be named with `set_label` too, so the spans of two graphs stay apart.

## What is compiled in

`SB_ENABLE_PROFILER` and `SB_PROFILE_LEVEL` come from the header `profile_config.hpp`, which the build generates, not from a compile definition: with the option off every macro of the `profile` module is nothing, and the level (default 2, in every build type) is the deepest zone a translation unit records. `profile.hpp` is included from `.cpp` files and from the profiling and benchmark modules only, so a SushiProfiler change recompiles those and no kernel, recorder or lowering unit, and the SYCL device pass never parses it. The zones are `blas.record` and `blas.scratch.lease`, a lock wait on the recording mutex, the counters `blas.scratch.pool_hit`, `pool_miss` and `unpooled`, all written from `engine_profile.cpp`, and one instant per GEMM launch naming the variant that ran, `blas.gemm.local_tiled`, `depth_split`, `joint_matrix`, `no_local` and `packed`, which a kernel notes through `Device::note_kernel_variant` and the engine writes.

# Profiling and logging

**Status:** Open — written, not built; the first build confirms it; every phase is `building` or `planned`.

**About:** What SushiBLAS tells SushiProfiler about its work, and where it logs a decision it took.

**Modules:** modules/foundation/profile, modules/foundation/cost, modules/engine/profiling, applications/benchmarks/device_ceilings

## Problem

SushiBLAS measured nothing. The benchmark harness times record, plan and run together on the host. No operation knew its FLOP count, and the library could not say how close a kernel ran to its device's limits. A profile of SushiAI's inference run could show where time went on the device and not whether the time was well spent.

The logs had the opposite fault. Twenty-eight per-launch `DEBUG` records and fifteen task-body `TRACE` records restated what a profiler measures, while a device that fell back to an assumed cache size, a tile pin that was accepted and a failed cleanup of a partial file were silent.

## Decision

The decisions were taken by the orchestrator on the owner's delegation, and a later review amended them.

- A task states its cost where it is recorded. `TaskRecorder::insert` applies one rule from a table in the `cost` module to the shape facts the call site states, adds the bytes the operands declared, and passes the result to SushiRuntime as `NodeOptions::cost`. The runtime copies it and its device source writes `flops`, `bytes` and `dtype` on the span.
- A task with no cost states none. Absent means the source omits the attributes; zero is a stated zero.
- `roofline` is a generic analysis in SushiProfiler. It reads span attributes and ceiling facts and knows no kernel. SushiBLAS states the costs and the ceilings and implements no analysis.
- A body that submits several kernels hands every event to the runtime, so the span covers the whole body. The last stage alone would make FLOPs over time a wrong rate.
- The ceilings are measured, per dtype, through the path the spans come from, and stated in the record as facts keyed by the runtime's track name.
- The profiling front, the cost table and the registration live in three modules: `profile` and `cost` in the foundation tier, `profiling` in the engine tier. `profiling` does not depend on `engine`.
- Logging follows the rules of the `logging` skill: a fallback is stated once per device, an accepted pin once per process, a record per launch is removed.

## Design

### Cost

`CostShape` is a `std::variant` of one struct per family: GEMM, convolution, elementwise, reduction, window and copy. `cost_of` visits it once. A new family adds a struct and a rule; no call site changes. The conventions, the weights of each row and the audit of the bodies that submit more than once are in [`modules/foundation/cost/README.md`](../../../modules/foundation/cost/README.md).

### From a span to an IR node

A record call adds exactly one runtime rule. SushiAI reads `graph.node_count()` around a lowering and maps the range through `Graph::rule_registration`, so the join to an IR node needs no field on `TaskHandle`. `TaskRecorder::origin` labels the nodes recorded while a scope is open, and spans carry the label.

### Ceilings

`device_ceilings` times an FMA loop per dtype, a streaming copy and an empty kernel with each run at least 10 ms, warm-up runs discarded and the median of five. A CPU device's numbers are marked smoke-only. [`docs/reference/CEILINGS_FORMAT.md`](../../reference/CEILINGS_FORMAT.md) holds the file; the `profiling` module reads it and states the facts.

### Alternatives set aside

A SushiBLAS source that annotates the runtime's spans needs a way for one source to read another's spans, which SushiProfiler does not have. Putting the cost in the eight task parameters leaves no free slot in GEMM. A `roofline` analysis in SushiBLAS cannot reach `sp record show` without `sp` linking every client.

## Phases

| Id | Phase | State |
| --- | --- | --- |
| P1 | The `profile` front and the options that compile it in or out | building |
| P2 | The `cost` module, the rules and their seams | building |
| P3 | The cost, the origin and the multi-event bodies reach the runtime | building |
| P4 | A test observer for the launch-width tests | building |
| P5 | The `profiling` module and `device_ceilings` | building |
| P6 | Zones, counters and the benchmark programs under a session | building |
| P7 | The kernel variant as a span attribute | planned |
| P8 | The logs of the launch probe, the tile pin, the fallback and the cleanup | building |

## Open questions

- The variant a GEMM ran as is an instant per launch. A fact written from inside a task body, joined to the node, needs a design on both sides: SushiProfiler's sources cannot read another source's spans.
- Every other `Unaudited` row needs its body read and a rule written; RW-121 names each.
- Nothing is measured: the ceilings are untested on the GPU and the overhead of the zones is not measured.

# Layered execution in SushiBLAS: report

**Status:** Open — B1, B2, B6 and B11 landed on 2026-10-24; not built yet.

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

## Dispatches

```
dispatch: B1 — sonnet medium, 31 turns, passed, rung 1
dispatch: B2 — sonnet medium, 112 turns, passed, rung 1
dispatch: B6 — sonnet medium, 46 turns, passed, rung 1
dispatch: B11 — sonnet medium, 48 turns, passed, rung 1
```

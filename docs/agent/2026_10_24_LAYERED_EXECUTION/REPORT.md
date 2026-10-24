# Layered execution in SushiBLAS: report

**Status:** Open — B1 and B2 landed on 2026-10-24; not built yet.

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

## Dispatches

```
dispatch: B1 — sonnet medium, 31 turns, passed, rung 1
dispatch: B2 — sonnet medium, 112 turns, passed, rung 1
```

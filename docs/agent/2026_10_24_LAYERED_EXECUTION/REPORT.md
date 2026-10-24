# Layered execution in SushiBLAS: report

**Status:** Open — B1 landed on 2026-10-24; not built yet.

## B1

`Test::RecordingFixture` (`modules/engine/engine_fixture/`) owns a `SushiRuntime::Graph` and a
`TaskRecorder` over its recorder, the engine's scratch pool and mode. `GraphProbe` holds one and
no `RecordedGraph`; its public interface is unchanged, so no test changed. `run()` empties the
graph after a run that succeeded and keeps it after one turned away with `InvalidState`, as
`RecordedGraph::run()` did.

Verification: `clang++ -fsyntax-only` of the probe's test, the fixture's test and three probe
users in other modules; the checkers under `tools/` pass. The other 16 probe users were not
syntax-checked. Not built: the owner ruled that each repository is built once at the end.

## Dispatches

```
dispatch: B1 — sonnet medium, 31 turns, passed, rung 1
```

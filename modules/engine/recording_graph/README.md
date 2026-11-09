# recording_graph

`RecordingGraph`, a runtime graph with a `TaskRecorder` recording on it. The module is internal:
its header is not installed, and it is added when `SB_BUILD_TESTS` or `SB_BUILD_BENCHMARKS` is
on. It includes no test framework, so `engine_fixture` and the benchmark programs of
`benchmark_harness` share it.

It depends on `engine`. Its header is included as
`<SushiBLAS/recording_graph/recording_graph.hpp>`.

| File | Declares |
| --- | --- |
| `recording_graph.hpp` | `RecordingGraph` and `K_DEFAULT_RECORDING_MODE` |
| `graph_host.hpp` | `GraphHost`, the port the graph records into and runs through |
| `runtime_graph_host.hpp` | `RuntimeGraphHost`, the adapter over a SushiRuntime graph, the default |
| `host_graph_host.hpp` | `HostGraphHost`, the adapter over a `HostTaskSink`: no runtime, ForceReference only |

## recording_graph.hpp

`set_label` names the graph in the spans of its runs, as `Graph::set_label` does.

`Engine` keeps no graph to run. `RecordingGraph` holds a `GraphHost` and a `TaskRecorder` over the
host's sink. The constructor that takes a runtime builds a `RuntimeGraphHost`, which owns a
`SushiRuntime::Graph`, the `GraphTaskSink` over its recorder, the `LateCountHold` and the `RunReport`
every run writes into; the constructor that takes a `std::unique_ptr<GraphHost>` takes any host, such
as `HostGraphHost`, which records and runs on the host without a runtime. Both take a scratch pool
and a mode; a caller that holds an engine passes
`engine.scratch_pool()` and `engine.mode()`. `K_DEFAULT_RECORDING_MODE` is the mode when none is
passed.

`recorder()` is what a caller builds its operation families on. `run()` runs the graph, returns
when every task has finished and empties it, so the object records and runs again; it returns
nothing to run for an empty graph and keeps the tasks after a run the runtime turned away for one
in flight. `discard()` drops the recorded tasks and returns how many, `plan()` returns what the
plan orders, and `recorded_count()` and `is_empty()` report what waits. It owns the `LateCountHold` its recorder
records late-count tasks into; each time `run()` or `discard()` empties the graph it releases the
hold, so no storage of a late-count task outlives the graph that held it. Emptying the graph also
destroys its compiled plan, which holds a copy of every task, so a task's operands and scratch are
freed or back in the pool when `run()` returns. `late_counts()` returns
the hold, or null for a host that holds none; `HostGraphHost` takes no late-count node and records
a late-count task as a host body.

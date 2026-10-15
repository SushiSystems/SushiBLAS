# graph_probe

`GraphProbe`, a graph a test owns so it can read what its record calls added. The module is internal and test-only: its headers are not installed, and it is added
only when `SB_BUILD_TESTS` is on.

It depends on `engine` and `test_runner`. Its headers are included as `<SushiBLAS/graph_probe/graph_probe.hpp>`.

| File | Declares |
| --- | --- |
| `graph_probe.hpp` | `GraphProbe` and `RecordedTask` |

## graph_probe.hpp

`Engine` has no accessor for its graph, so a test that asserts on what an operation declared
cannot read it from an engine. `GraphProbe` owns a `ScratchPool`, a `RecordedGraph` and a
`TaskRecorder` over them on the shared test runtime. A test builds a view on
`probe.recorder()` and records. `recorded_count()` returns how many tasks wait on the graph, and
`take_recorded_count()` returns that count and empties the graph with `RecordedGraph::discard()`.
`execute()` is `RecordedGraph::run()`, the call `Engine::execute` makes. The constructor takes the execution
mode the recorder records under.

`recorded_tasks()` is the view a test reads a recorded task through. It calls `Graph::plan()`,
which compiles the graph when no plan is current and runs nothing, and copies each `PlanNode`
into a `RecordedTask`: the task's `name` and `operation`, its `dependency_count` (the node's
`predecessor_count`), and the regions it declared as `reads` and `writes`. It returns a
`SushiRuntime::Result`: the tasks in record order, an empty list when nothing is recorded, or the
error that kept the graph from compiling. `dependency_counts()` returns the counts alone. A test
reads the tasks before `execute()`, which leaves the graph empty. The plan holds no task body, so
a test cannot call one; it runs a task through `execute()`.

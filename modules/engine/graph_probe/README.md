# graph_probe

`GraphProbe` and `HostGraphProbe`, graphs a test owns so it can read what its record calls added. The module is internal and test-only: its headers are not installed, and it is added
only when `SB_BUILD_TESTS` is on.

It depends on `engine`, `engine_fixture` and `test_runner`. Its headers are included as `<SushiBLAS/graph_probe/graph_probe.hpp>`.

| File | Declares |
| --- | --- |
| `graph_probe.hpp` | `GraphProbe` and `RecordedTask` |
| `host_graph_probe.hpp` | `HostGraphProbe` |

## graph_probe.hpp

`RecordedTask` also holds the `cost` the task stated, empty when it stated none, and the `origin` it was recorded under.

`Engine` owns no graph, so a test that asserts on what an operation declared
cannot read it from an engine. `GraphProbe` owns a `ScratchPool` and a `RecordingFixture` over it on the shared test
runtime; the fixture holds the graph and the `TaskRecorder`. A test builds a view on
`probe.recorder()` and records. `recorded_count()` returns how many tasks wait on the graph, and
`take_recorded_count()` returns that count and empties the graph with
`RecordingFixture::discard()`. `execute()` is `RecordingFixture::run()`. The constructor takes the
execution mode the recorder records under.

`recorded_tasks()` is the view a test reads a recorded task through. It calls `Graph::plan()`,
which compiles the graph when no plan is current and runs nothing, and copies each `PlanNode`
into a `RecordedTask`: the task's `name` and `operation`, its `dependency_count` (the node's
`predecessor_count`), and the regions it declared as `reads` and `writes`. It returns a
`SushiRuntime::Result`: the tasks in record order, an empty list when nothing is recorded, or the
error that kept the graph from compiling. `dependency_counts()` returns the counts alone. A test
reads the tasks before `execute()`, which leaves the graph empty. The plan holds no task body, so
a test cannot call one; it runs a task through `execute()`.

## host_graph_probe.hpp

`HostGraphProbe` is the probe for a test that makes no runtime. It owns a `HostScratchSource`, a
`ScratchPool` over it, a `HostTaskSink` and a `TaskRecorder` over both, which records under
`ForceReference`. Its tensors need host storage only, because it reads the regions a task declared
and never the memory behind them. `recorded_tasks()` returns the tasks in record order as
`RecordedTask`s whose `dependency_count` is computed from the regions: a task waits for an earlier
one when a region it reads overlaps a region the earlier one writes, or a region it writes overlaps
a region the earlier one reads or writes (`SushiRuntime::BufferRegion::overlaps`), and an earlier
task counts once. The runtime plan of `GraphProbe` is the reference for that rule; the host probe
reproduces it and has no test that compares the two. `cost` stays empty.

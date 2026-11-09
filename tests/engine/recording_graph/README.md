# Tests of recording_graph

The tests of the `recording_graph` module, built into `sushiblas_recording_graph_test_objects` and linked
into `sushiblas_engine_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks. [`tests/README.md`](../../README.md) holds the reasoning the tests share.

| File | Tests |
| --- | --- |
| `test_recording_graph.cpp` | Tests that a graph over a `HostGraphHost` records, runs, empties, discards, runs again and lets go of the storages of its late-count tasks, on a host engine |
| `test_host_graph_host.cpp` | Tests that `HostGraphHost` counts, runs and drops the tasks of its sink, names the task that failed, has no plan and admits the reference mode only |
| `test_runtime_graph_host.cpp` | Tests, on a runtime, that a graph built from it records under the mode it is given, orders its plan, runs, and releases the hold of its late-count storages |
| `compile_graph_host.cpp` | Checks at compile time that `GraphHost` is an abstract port with a virtual destructor; compiled, never run |
| `recording_graph_tasks.hpp` | Records the two tasks the suites share: one that does nothing and one with a late count |

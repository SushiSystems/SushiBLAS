# Tests of graph_probe

The tests of the `graph_probe` module, built into `sushiblas_graph_probe_test_objects` and linked
into `sushiblas_engine_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_graph_probe.cpp` | Tests that GraphProbe reports the tasks, the dependencies and the counts it recorded |

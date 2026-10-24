# Tests of benchmark_harness

The tests of the `benchmark_harness` module, built into `sushiblas_benchmark_harness_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_benchmark_graph.cpp` | Tests that a benchmark measures a task recorded on the graph it owns |
| `test_benchmark_harness.cpp` | Tests the size parsing, the names, the timing and the guard the benchmarks share |

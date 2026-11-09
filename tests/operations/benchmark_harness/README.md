# Tests of benchmark_harness

The tests of the `benchmark_harness` module, built into `sushiblas_benchmark_harness_test_objects` and linked
into `sushiblas_operations_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/operations/benchmark_harness/README.md` describes the harness.

| File | Tests |
| --- | --- |
| `test_benchmark_harness.cpp` | Tests the size parsing, the refusal of an unbenchmarked dtype, the device support, the timing on a scripted clock, measure over a RecordingGraph, the guard and the result table the benchmarks share |
| `test_environment_value.cpp` | Tests, over an injected lookup, that environment_value returns a set value and returns empty for an unset or empty variable |
| `test_profile_session.cpp` | Tests that a benchmark program records nothing unless `SB_BENCH_PROFILE` names a folder, and that marking no session does nothing |

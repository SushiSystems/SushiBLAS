# Tests of benchmark_harness

The tests of the `benchmark_harness` module, built into `sushiblas_benchmark_harness_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/operations/benchmark_harness/README.md` describes the harness.

| File | Tests |
| --- | --- |
| `test_benchmark_harness.cpp` | Tests the size parsing, the names, the timing, measure over a RecordingGraph and the guard the benchmarks share |
| `test_environment_value.cpp` | Tests that environment_value returns a set value and returns empty for an unset or empty variable |
| `test_profile_session.cpp` | Tests that a benchmark program records nothing unless `SB_BENCH_PROFILE` names a folder |

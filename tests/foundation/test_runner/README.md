# Tests of test_runner

The tests of the `test_runner` module, built into `sushiblas_test_runner_test_objects` and linked
into `sushiblas_foundation_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_cpu_queue.cpp` | Tests that the environment makes one queue exactly when SYCL reports a CPU device; an empty device list is `Small_` |
| `test_captured_record_count.cpp` | Tests that `count_records` counts the records of one level whose message holds a text |
| `test_seeded_schedule.cpp` | Tests that `SeededSchedule` runs every step of every task once, each task's steps in order, and the same order for the same seed |
| `test_scoped_log_level.cpp` | Tests that `ScopedLogLevel` sets the level of SushiBLAS while it lives and puts back the level it found |
| `test_test_environment.cpp` | Tests that TestEnvironment hands out one runtime, its queues and its allocation count |
| `test_run_observer_capture.cpp` | Tests that `RunObserverCapture` keeps one record per dispatch of each run it observed |
| `compile_test_runner_log_category.cpp` | Checks at compile time that the log category names its module and project; compiled, never run |

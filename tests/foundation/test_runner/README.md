# Tests of test_runner

The tests of the `test_runner` module, built into `sushiblas_test_runner_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_test_environment.cpp` | Tests that TestEnvironment hands out one runtime, its queues and its allocation count |

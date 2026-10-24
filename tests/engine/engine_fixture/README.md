# Tests of engine_fixture

The tests of the `engine_fixture` module, built into `sushiblas_engine_fixture_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_sushiblas_test.cpp` | Tests that SushiBLASTest gives each test a row-major engine and initialised tensors |
| `test_recording_fixture.cpp` | Tests that RecordingFixture records on its own graph, runs it and leaves it empty |

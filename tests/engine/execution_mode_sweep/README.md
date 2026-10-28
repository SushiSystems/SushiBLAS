# Tests of execution_mode_sweep

The tests of the `execution_mode_sweep` module, built into `sushiblas_execution_mode_sweep_test_objects` and linked
into `sushiblas_engine_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_execution_mode_sweep.cpp` | Tests that the sweep names each forced execution mode once |

# Tests of substitution

The tests of the `substitution` module, built into `sushiblas_substitution_test_objects` and linked
into `sushiblas_foundation_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_running_solution.cpp` | Tests RunningSolution: where it keeps a solved element and what it rounds |

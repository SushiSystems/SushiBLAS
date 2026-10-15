# Tests of operands

The tests of the `operands` module, built into `sushiblas_operands_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_scalar.cpp` | Tests BLASScalar: what converts to it, what it narrows to and what it reports |

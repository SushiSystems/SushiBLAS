# Tests of core

The tests of the `core` module, built into `sushiblas_core_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_blas_types.cpp` | Tests the numbers and the size of DataType and of the BLAS flags, and the flag queries |
| `test_logger.cpp` | Pins that the logger writes every line before an assertion aborts or a shutdown returns |

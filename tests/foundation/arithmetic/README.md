# Tests of arithmetic

The tests of the `arithmetic` module, built into `sushiblas_arithmetic_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_complex_arithmetic.cpp` | Tests the conjugation, product and quotient bricks on real and complex values |

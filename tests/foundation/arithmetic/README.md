# Tests of arithmetic

The tests of the `arithmetic` module, built into `sushiblas_arithmetic_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_accumulator.cpp` | Tests that Accumulator widens HALF to FLOAT32 and keeps every other type |
| `test_complex_arithmetic.cpp` | Tests the conjugation, product and quotient bricks on real and complex values |
| `test_device_math.cpp` | Tests that the host side of Math names the standard functions with their results |
| `test_safe_math.cpp` | Tests the real and complex safe functions of Dispatch against the standard library |

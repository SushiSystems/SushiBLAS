# Tests of serial_reductions

The tests of the `serial_reductions` module, built into `sushiblas_serial_reductions_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_serial_reductions.cpp` | Tests the serial host reductions a test takes its expected values from |

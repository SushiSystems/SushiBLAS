# Tests of reference

The tests of the `reference` module, built into `sushiblas_reference_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_reference_accumulator.cpp` | Tests the type and the running sum a reference kernel accumulates in |
| `test_serial_1d.cpp` | Tests the serial map, fused update and indexed loop against the launched shapes |
| `test_serial_reduce.cpp` | Tests the serial host folds against the launched folds and accumulators they mirror |

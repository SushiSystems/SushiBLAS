# Tests of reduction

The tests of the `reduction` module, built into `sushiblas_reduction_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_reduce_1d.cpp` | Tests the accumulators, the reduction plan and a reduction on the host device |
| `test_reduce_axis.cpp` | Tests the axis span, the axis plan and both launch shapes of reduce_axis |

# Tests of experimental

The tests of the `experimental` module, built into `sushiblas_experimental_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `lapack/test_linalg.cpp` | Tests lu_factor, lu_solve and gmres in both forced modes against P L U reconstruction and a direct solve, and their refusals |
| `signal/test_transforms.cpp` | Tests fft and inverse_fft in both forced modes against a direct DFT, and their refusals |
| `test_dense_operand.cpp` | Tests the record-time operand checks the transforms and dense solvers share |

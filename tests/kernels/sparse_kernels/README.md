# Tests of sparse_kernels

The tests of the `sparse_kernels` module, built into `sushiblas_sparse_kernels_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `reference/test_structure.cpp` | Tests find_defect on the arrays of a CSR matrix and on the positions of a sparse vector |

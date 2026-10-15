# Tests of sparse_storage

The tests of the `sparse_storage` module, built into `sushiblas_sparse_storage_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_sparse_builder.cpp` | Tests the sparse vectors and CSR matrices SparseBuilder makes from host data |
| `test_sparse_matrix.cpp` | Tests that a SparseMatrix reads its entry count and both dtypes from its tensors |
| `test_sparse_vector.cpp` | Tests that a SparseVector reads its entry count and both dtypes from its tensors |

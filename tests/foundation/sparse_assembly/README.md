# Tests of sparse_assembly

The tests of the `sparse_assembly` module, built into `sushiblas_sparse_assembly_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_sparse_assembly.cpp` | Tests assemble_csr and require_distinct_positions on host triplets and positions |

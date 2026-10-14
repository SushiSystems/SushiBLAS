# Tests of sparse_assembly

The unit tests of the `sparse_assembly` module, built into `sushiblas_sparse_assembly_tests` by the module's `CMakeLists.txt`. The source comments state what a test checks. `tests/unit/README.md` holds the reasoning the tests share, and `tests/common/README.md` describes the fixtures.

| File | Tests |
| --- | --- |
| `test_sparse_assembly.cpp` | Tests assemble_csr and require_distinct_positions on host triplets and positions |

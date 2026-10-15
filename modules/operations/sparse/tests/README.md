# Tests of sparse

The unit tests of the `sparse` module, built into `sushiblas_sparse_tests` by the
module's `CMakeLists.txt`. The source comments state what a test checks; a section below
holds why a test is written the way it is. `tests/unit/README.md` holds the reasoning the
tests share, and `modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_sparse_level1.cpp` | Tests the sparse-dense level 1 routines of SparseOps against values worked out by hand |
| `test_sparse_spmm.cpp` | Tests SparseOps::spmm column by column against host products, on both layouts |
| `test_sparse_spmv.cpp` | Tests SparseOps::spmv and the transposed SYCL kernel against host products |
| `test_sparse_spsymv.cpp` | Tests SparseOps::spsymv, sphemv and their SYCL kernel against mirrored host products |
| `test_sparse_sptrsv.cpp` | Tests SparseOps::sptrsv against triangular systems whose solution is known |
| `test_sparse_validation.cpp` | Tests the record-time operand checks of the sparse operations |

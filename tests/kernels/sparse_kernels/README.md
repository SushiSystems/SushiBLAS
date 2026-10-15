# Tests of sparse_kernels

The tests of the `sparse_kernels` module, built into `sushiblas_sparse_kernels_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `reference/test_level1.cpp` | Tests the reference sparse-dense level 1 kernels against hand-computed results |
| `reference/test_spmm.cpp` | Tests the reference spmm column by column against the reference spmv, in both layouts |
| `reference/test_spmv.cpp` | Tests the reference spmv against a dense product, plain and transposed |
| `reference/test_spsymv.cpp` | Tests the reference spsymv against a dense product with the mirrored triangle |
| `reference/test_sptrsv.cpp` | Tests that the reference sptrsv recovers x from op(A) * x for either triangle |
| `reference/test_structure.cpp` | Tests find_defect on the arrays of a CSR matrix and on the positions of a sparse vector |
| `sycl/test_level1.cpp` | Tests the SYCL sparse-dense level 1 kernels against the reference ones on the host device |
| `sycl/test_row_blocks.cpp` | Tests how many row blocks a scattering product takes, where each starts, and their fold |
| `sycl/test_spmm.cpp` | Tests the SYCL spmm against the reference spmm in both layouts and both forms |
| `sycl/test_spmv.cpp` | Tests the SYCL spmv, plain and scattered transposed, against the reference spmv |
| `sycl/test_spsymv.cpp` | Tests the scattered SYCL spsymv against the reference spsymv on the host device |
| `sycl/test_sptrsv.cpp` | Tests the SYCL sptrsv against the reference sptrsv for every flag on the host device |
| `test_csr_pattern.cpp` | Tests that a CsrPattern reads its rows and columns under either index base |
| `test_index_list.cpp` | Tests that an IndexList reads the dense positions of its entries under either base |
| `test_sparse.cpp` | Tests that sparse.hpp alone reaches the structure checks and both kinds of kernel |

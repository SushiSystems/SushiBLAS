# Tests of blas_kernels

The tests of the `blas_kernels` module, built into `sushiblas_blas_kernels_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `reference/level1/test_level1.cpp` | Tests the reference Level 1 kernels against hand-computed results and strides |
| `reference/level1/test_rotg.cpp` | Tests the reference rotg on the 3-4-5 triangle, a zero second component and a zero first |
| `reference/level1/test_rotm.cpp` | Tests the reference rotm under each of the four flags of its parameter |
| `reference/level1/test_rotmg.cpp` | Tests that the reference rotmg builds a transformation that zeroes the second component |
| `reference/level2/test_gbmv.cpp` | Tests the reference gbmv against a loop over the band, in both layouts and transposes |
| `reference/level2/test_gemv.cpp` | Tests the reference gemv against a dense loop, in both layouts and transposes |
| `reference/level2/test_ger.cpp` | Tests the reference ger, A += alpha * x * y^T, in both layouts and with a conjugated y |
| `reference/level2/test_sbmv.cpp` | Tests the reference sbmv against a dense loop over the mirrored band triangle |
| `reference/level2/test_spmv.cpp` | Tests the reference spmv against a dense loop over the mirrored packed triangle |
| `reference/level2/test_spr.cpp` | Tests the reference spr, which adds alpha * x * x^T to a packed triangle |
| `reference/level2/test_spr2.cpp` | Tests the reference spr2, which adds alpha * (x * y^T + y * x^T) to a packed triangle |
| `reference/level2/test_symmetric_product.cpp` | Tests the reference symmetric product through a packed and a band access |
| `reference/level2/test_symv.cpp` | Tests the reference symv against a dense loop over the mirrored triangle |
| `reference/level2/test_syr.cpp` | Tests the reference syr, which adds alpha * x * x^T to one triangle and nothing else |
| `reference/level2/test_syr2.cpp` | Tests the reference syr2, which adds alpha * (x * y^T + y * x^T) to one triangle |
| `reference/level2/test_tbmv.cpp` | Tests the reference tbmv against a dense product with the band triangle, for every flag |
| `reference/level2/test_tbsv.cpp` | Tests that the reference tbsv recovers x from op(A) * x for a band triangle |
| `reference/level2/test_tpmv.cpp` | Tests the reference tpmv against a dense product with the packed triangle, for every flag |
| `reference/level2/test_tpsv.cpp` | Tests that the reference tpsv recovers x from op(A) * x for a packed triangle |
| `reference/level2/test_triangular_product.cpp` | Tests the reference triangular product through a packed access, for every flag |
| `reference/level2/test_triangular_solve.cpp` | Tests that the reference triangular solve undoes a dense product through a packed access |
| `reference/level2/test_trmv.cpp` | Tests the reference trmv against a dense product with the triangle, for every flag |
| `reference/level3/test_syr2k.cpp` | Tests the reference syr2k, single and batched, against a dense rank-2k update |
| `reference/level3/test_syrk.cpp` | Tests the reference syrk, single and batched, against a dense rank-k update |
| `reference/level3/test_trsm.cpp` | Tests that the reference trsm recovers X from op(A) * X or X * op(A), single and batched |
| `sycl/level1/test_level1.cpp` | Tests the SYCL Level 1 kernels against the reference kernels on the host device |
| `sycl/level1/test_rotg.cpp` | Tests the SYCL rotg against the reference rotg on the host device |
| `sycl/level1/test_rotm.cpp` | Tests the SYCL rotm against the reference rotm under every flag on the host device |
| `sycl/level1/test_rotmg.cpp` | Tests the SYCL rotmg against the reference rotmg on the host device |
| `sycl/level2/test_gbmv.cpp` | Tests the SYCL gbmv against the reference gbmv on the host device |
| `sycl/level2/test_gemv.cpp` | Tests the SYCL gemv against the reference gemv on the host device |
| `sycl/level2/test_ger.cpp` | Tests the SYCL ger against the reference ger on the host device |
| `sycl/level2/test_row_reduction.cpp` | Tests the row width a profile gives and the group fold of whole rows and triangle rows |
| `sycl/level2/test_sbmv.cpp` | Tests the SYCL sbmv against the reference sbmv on the host device |
| `sycl/level2/test_spmv.cpp` | Tests the SYCL spmv against the reference spmv on the host device |
| `sycl/level2/test_spr.cpp` | Tests the SYCL spr against the reference spr on the host device |
| `sycl/level2/test_spr2.cpp` | Tests the SYCL spr2 against the reference spr2 on the host device |
| `sycl/level2/test_symmetric_product.cpp` | Tests the SYCL symmetric product against the reference one through a packed access |
| `sycl/level2/test_symv.cpp` | Tests the SYCL symv against the reference symv on the host device |
| `sycl/level2/test_syr.cpp` | Tests the SYCL syr against the reference syr on the host device |
| `sycl/level2/test_syr2.cpp` | Tests the SYCL syr2 against the reference syr2 on the host device |
| `sycl/level2/test_tbmv.cpp` | Tests the SYCL tbmv against the reference tbmv for every flag on the host device |
| `sycl/level2/test_tbsv.cpp` | Tests the SYCL tbsv against the reference tbsv for every flag on the host device |
| `sycl/level2/test_tpmv.cpp` | Tests the SYCL tpmv against the reference tpmv for every flag on the host device |
| `sycl/level2/test_tpsv.cpp` | Tests the SYCL tpsv against the reference tpsv for every flag on the host device |
| `sycl/level2/test_triangular_product.cpp` | Tests the SYCL triangular product against the reference one through a packed access |
| `sycl/level2/test_triangular_solve.cpp` | Tests the blocked SYCL triangular solve against the reference one through a packed access |
| `sycl/level2/test_trmv.cpp` | Tests the SYCL trmv against the reference trmv for every flag on the host device |
| `sycl/level2/test_vector_snapshot.cpp` | Tests that VectorSnapshot copies a strided vector into a contiguous one |
| `sycl/level3/test_syr2k.cpp` | Tests the SYCL syr2k against the reference syr2k on the host device, batched included |
| `sycl/level3/test_syrk.cpp` | Tests the tiled and the packed SYCL syrk against the reference syrk on the host device |
| `sycl/level3/test_triangular_plan.cpp` | Tests the block plan of a triangular solve on a CPU and a GPU profile |
| `sycl/level3/test_triangular_system.cpp` | Tests the decoding of a TRSM call into the steps and vectors of one substitution |
| `sycl/level3/test_trsm.cpp` | Tests the blocked SYCL trsm against the reference trsm on the host device |
| `test_blas.cpp` | Tests that blas.hpp alone reaches a reference and a SYCL kernel of each BLAS level |

# Tests of blas_kernels

The tests of the `blas_kernels` module, built into `sushiblas_blas_kernels_test_objects` and linked
into `sushiblas_kernels_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
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
| `reference/level2/compile_symmetric_product.cpp` | Checks at compile time that the reference symmetric product keeps the signature of its access form |
| `reference/level2/test_symv.cpp` | Tests the reference symv against a dense loop over the mirrored triangle |
| `reference/level2/test_syr.cpp` | Tests the reference syr, which adds alpha * x * x^T to one triangle and nothing else |
| `reference/level2/test_syr2.cpp` | Tests the reference syr2, which adds alpha * (x * y^T + y * x^T) to one triangle |
| `reference/level2/test_tbmv.cpp` | Tests the reference tbmv against a dense product with the band triangle, for every flag |
| `reference/level2/test_tbsv.cpp` | Tests that the reference tbsv recovers x from op(A) * x for a band triangle |
| `reference/level2/test_tpmv.cpp` | Tests the reference tpmv against a dense product with the packed triangle, for every flag |
| `reference/level2/test_tpsv.cpp` | Tests that the reference tpsv recovers x from op(A) * x for a packed triangle |
| `reference/level2/compile_triangular_product.cpp` | Checks at compile time that the reference triangular product keeps the signature of its access form |
| `reference/level2/compile_triangular_solve.cpp` | Checks at compile time that the reference triangular solve keeps the signature of its access form |
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
| `sycl/level2/test_row_reduction.cpp` | Tests that gemv and trmv fold short rows, long rows and triangle rows to their sums, in a fixed order, and that a row takes a work-group from the staged threshold of 16384 columns |
| `sycl/level2/test_sbmv.cpp` | Tests the SYCL sbmv against the reference sbmv on the host device |
| `sycl/level2/test_spmv.cpp` | Tests the SYCL spmv against the reference spmv on the host device |
| `sycl/level2/test_spr.cpp` | Tests the SYCL spr against the reference spr on the host device |
| `sycl/level2/test_spr2.cpp` | Tests the SYCL spr2 against the reference spr2 on the host device |
| `sycl/level2/test_symv.cpp` | Tests the SYCL symv against the reference symv on the host device |
| `sycl/level2/test_syr.cpp` | Tests the SYCL syr against the reference syr on the host device |
| `sycl/level2/test_syr2.cpp` | Tests the SYCL syr2 against the reference syr2 on the host device |
| `sycl/level2/test_tbmv.cpp` | Tests the SYCL tbmv against the reference tbmv for every flag on the host device |
| `sycl/level2/test_tbsv.cpp` | Tests the SYCL tbsv against the reference tbsv for every flag on the host device |
| `sycl/level2/test_tpmv.cpp` | Tests the SYCL tpmv against the reference tpmv for every flag on the host device |
| `sycl/level2/test_tpsv.cpp` | Tests the SYCL tpsv against the reference tpsv for every flag on the host device |
| `sycl/level2/test_triangular_solve.cpp` | Tests the blocked SYCL tpsv with a plan against the reference tpsv for each block width, triangle and transpose |
| `sycl/level2/test_trmv.cpp` | Tests the SYCL trmv against the reference trmv for every flag on the host device |
| `sycl/level2/compile_symmetric_product.cpp` | Checks at compile time that the SYCL symmetric product exists for a packed access |
| `sycl/level2/compile_triangular_product.cpp` | Checks at compile time that the SYCL triangular product exists for a packed access |
| `sycl/level2/compile_vector_snapshot.cpp` | Checks at compile time that the SYCL VectorSnapshot functor can be copied to a device |
| `sycl/level3/test_syr2k.cpp` | Tests the SYCL syr2k against the reference syr2k on the host device, batched included |
| `sycl/level3/test_syrk.cpp` | Tests the tiled and the packed SYCL syrk against the reference syrk on the host device, a NaN in C under a beta of zero, a scratch below the packed plan, and the tile check |
| `sycl/level3/test_triangular_plan.cpp` | Tests the block plan of a triangular solve on a CPU and a GPU profile, as rows, with no device |
| `sycl/level3/test_triangular_system.cpp` | Tests the decoding of a TRSM call into the steps and vectors of one substitution |
| `sycl/level3/test_trsm.cpp` | Tests the blocked SYCL trsm against the reference trsm on the host device, and a non-unit diagonal against the solution it was built from |
| `test_blas.cpp` | Tests that the level 3 kernels reached through blas.hpp, the reference one and the SYCL one, give the hand value of a small update |
| `compile_blas.cpp` | Checks at compile time that blas.hpp alone reaches a reference and a SYCL kernel of each BLAS level |
| `compile_blas_kernels_log_category.cpp` | Checks at compile time that the log category names its module and project |
| `test_blend_index.cpp` | Tests the beta blend of an updated value with its prior value, real and complex, against hand-derived values |
| `test_matrix_vector_index.cpp` | Tests the extent, row-shape and row-window math of the matrix-vector kernels against hand-derived values |
| `test_rank_update_index.cpp` | Tests the layout offset, work-item split and triangle tests of the rank updates against hand-derived values and an element cover |
| `test_rotm_index.cpp` | Tests the ROTM flag test and pair transformation against hand-derived values |
| `test_triangular_solve_index.cpp` | Tests the block walk, trailing range and stream split of the blocked triangular solves against hand-derived values and a cover check |

## How the tests are written

A kernel that takes flags runs one row per flag combination. The row names come from the label
functions of `blas_test_oracle` (`layout_uplo_label`, `triangular_label`), and a file that keeps
plain tests beside its rows gives the rows a fixture of their own, such as `Small_ReferenceGemvDense`.
The batched forms run one row per matrix of the batch against a dense oracle, so a stride error shows
as the failing matrix. The reference tests are `Small_`, the SYCL ones `Medium_`, and the block plan and
the system decoding of the triangular solve are `Small_` rows because they need no device.

A header whose only public name is a `Detail` function has no test of its own. A `compile_` file
beside the tests checks the signature at compile time, and the promise the function serves is held by
the public kernel that calls it (`spmv` and `sbmv` for the symmetric product, `tpmv` for the triangular
product, `tpsv` for the triangular solve, `trmv` for the vector snapshot).


# Tests of linalg_kernels

The tests of the `linalg_kernels` module, built into `sushiblas_linalg_kernels_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks. [`tests/README.md`](../../README.md) holds the reasoning the tests share.

| File | Tests |
| --- | --- |
| `reference/test_gmres.cpp` | Tests the reference GMRES against a direct solve, its reported residual, its warm start and a zero right-hand side |
| `reference/test_lu.cpp` | Tests the reference LU by P L U reconstruction, its pivot rule, its singular flag and its solve |
| `sycl/test_gmres.cpp` | Tests that the SYCL GMRES equals the reference bit for bit on the host device |
| `sycl/test_lu.cpp` | Tests that the SYCL LU and its solve equal the reference bit for bit |
| `test_chunked_dot.cpp` | Tests that chunked_dot sums each chunk in index order and then the chunk sums in order |
| `test_gmres_workspace.cpp` | Tests that GmresLayout places every array of the workspace after the one before it |
| `test_linalg.cpp` | Tests that linalg.hpp alone reaches the reference and the SYCL kernels |

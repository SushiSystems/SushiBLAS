# Tests of linalg_kernels

The tests of the `linalg_kernels` module, built into `sushiblas_linalg_kernels_test_objects` and linked
into `sushiblas_kernels_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks. [`tests/README.md`](../../README.md) holds the reasoning the tests share.

| File | Tests |
| --- | --- |
| `reference/test_gmres.cpp` | Tests the reference GMRES against a direct solve, its reported residual, its warm start and a zero right-hand side |
| `reference/test_lu.cpp` | Tests the reference LU by P L U reconstruction, its pivot rule, its singular flag and its solve |
| `sycl/test_gmres.cpp` | Tests that the SYCL GMRES equals the reference bit for bit on the host device from a non-zero guess, and that a zero right-hand side overwrites a poisoned guess with zero |
| `sycl/test_lu.cpp` | Tests that the SYCL LU and its solve equal the reference bit for bit |
| `test_chunked_dot.cpp` | Tests that chunked_dot sums each chunk in index order and then the chunk sums in order |
| `test_gmres_workspace.cpp` | Tests that GmresLayout places every array of the workspace after the one before it, and that a default solve reaches the default tolerance |
| `test_lu_index.cpp` | Tests the LU pivot search, its tie rule and singular bound, and the trailing-block and solve coordinates on the host |
| `test_linalg.cpp` | Tests that linalg.hpp alone declares the LU factor and solve in the reference and the SYCL kernels |

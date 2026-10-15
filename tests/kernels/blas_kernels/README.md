# Tests of blas_kernels

The tests of the `blas_kernels` module, built into `sushiblas_blas_kernels_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `sycl/level3/test_triangular_plan.cpp` | Tests the block plan of a triangular solve and the decoding of a TRSM call into one substitution |

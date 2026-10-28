# Tests of random_kernels

The tests of the `random_kernels` module, built into `sushiblas_random_kernels_test_objects` and linked
into `sushiblas_kernels_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `reference/test_orthonormalize.cpp` | Tests that the reference sweep leaves strided vectors orthogonal and of the given norm |
| `sycl/test_orthonormalize.cpp` | Tests the work-group SYCL sweep against the reference sweep on the host device |
| `sycl/test_philox.cpp` | Tests Philox4x32-10 against its known answers and the draws built on it, host and device |
| `sycl/test_poisson.cpp` | Tests the Poisson parameters, both draw methods and their agreement on the host device |
| `sycl/test_truncated_normal.cpp` | Tests the normal distribution function and the truncated draws it inverts |
| `test_random.cpp` | Tests that random.hpp alone reaches the draws and both orthonormalisation sweeps |

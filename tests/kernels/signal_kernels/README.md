# Tests of signal_kernels

The tests of the `signal_kernels` module, built into `sushiblas_signal_kernels_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `reference/test_convolve_1d.cpp` | Tests the reference valid-mode convolution against hand values and a direct loop |
| `sycl/test_convolve_1d.cpp` | Tests the SYCL valid-mode convolution against the reference on the host device |
| `test_signal.cpp` | Tests that signal.hpp alone reaches the reference and the SYCL signal kernels |

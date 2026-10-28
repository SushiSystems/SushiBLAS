# Tests of signal_kernels

The tests of the `signal_kernels` module, built into `sushiblas_signal_kernels_test_objects` and linked
into `sushiblas_kernels_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `reference/test_fft.cpp` | Tests the reference batched FFT against a direct DFT, and its inverse round trip |
| `reference/test_convolve_1d.cpp` | Tests the reference valid-mode convolution against hand values and a direct loop |
| `sycl/test_convolve_1d.cpp` | Tests the SYCL valid-mode convolution against the reference on the host device |
| `sycl/test_fft.cpp` | Tests that the SYCL batched FFT equals the reference bit for bit on the host device |
| `test_fft_plan.cpp` | Tests that FftPlan cooks the radix-2 and Bluestein tables and sizes its scratch |
| `test_signal.cpp` | Tests that signal.hpp alone reaches the reference and the SYCL signal kernels |

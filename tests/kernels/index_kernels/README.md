# Tests of index_kernels

The tests of the `index_kernels` module, built into `sushiblas_index_kernels_test_objects` and linked
into `sushiblas_kernels_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks.
[`tests/README.md`](../../README.md) holds the reasoning the tests share.

| File | Tests |
| --- | --- |
| `reference/test_along.cpp` | Tests the reference take_along and put_along against hand values |
| `reference/test_rows.cpp` | Tests the reference take_rows and add_rows against hand values |
| `sycl/test_along.cpp` | Tests the SYCL take_along and put_along against the reference ones |
| `sycl/test_rows.cpp` | Tests the SYCL take_rows and add_rows against the reference ones |
| `test_spans.cpp` | Tests MatrixSpan offsets and is_in_range against hand values |
| `compile_index.cpp` | Compile check, never run: `index.hpp` stands alone |

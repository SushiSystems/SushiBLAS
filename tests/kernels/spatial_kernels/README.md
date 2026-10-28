# Tests of spatial_kernels

The tests of the `spatial_kernels` module, built into `sushiblas_spatial_kernels_test_objects` and linked
into `sushiblas_kernels_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `reference/test_broadcast.cpp` | Tests the reference channel affine, spatial broadcast and layout conversion |
| `reference/test_channels.cpp` | Tests that the reference channel copy moves a range of channels between offsets |
| `reference/test_col2im.cpp` | Tests that the reference col2im is the adjoint of im2col and counts the overlaps |
| `reference/test_im2col.cpp` | Tests the reference im2col on a one-tap window and on a padded two-by-two window |
| `reference/test_pool_avg.cpp` | Tests the reference average pool against hand values and its backward as its adjoint |
| `reference/test_pool_max.cpp` | Tests the reference max pool, the index it records and the backward that uses it |
| `reference/test_upsample.cpp` | Tests the reference nearest upsample against hand values and its backward as its adjoint |
| `reference/test_window_dot.cpp` | Tests the reference window product against hand values and its backward as its adjoint |
| `reference/test_window_dot_weights.cpp` | Tests that the reference weight gradient pairs with the window product as a bilinear form |
| `sycl/test_broadcast.cpp` | Tests the SYCL affine, broadcast and layout conversion against the reference |
| `sycl/test_channels.cpp` | Tests the SYCL channel copy against the reference channel copy on the host device |
| `sycl/test_col2im.cpp` | Tests the SYCL col2im against the reference col2im on the host device |
| `sycl/test_im2col.cpp` | Tests the SYCL im2col against the reference im2col on the host device |
| `sycl/test_pool_avg.cpp` | Tests the SYCL average pool and its backward against the reference ones |
| `sycl/test_pool_max.cpp` | Tests the SYCL max pool and its backward against the reference ones |
| `sycl/test_upsample.cpp` | Tests the SYCL nearest upsample and its backward against the reference ones |
| `sycl/test_window_dot.cpp` | Tests the SYCL window product and its backward against the reference ones |
| `sycl/test_window_dot_weights.cpp` | Tests the SYCL weight gradient in both launch shapes against the reference |
| `test_spatial.cpp` | Tests that spatial.hpp alone reaches the reference and the SYCL spatial kernels |

## Helpers

The headers below are shared by tests of this module only, so they sit in this folder.

| Header | What it does |
| --- | --- |
| `nhwc_span.hpp` | Gives the window span of a dense NHWC input and output, and the geometry the device tests share. |

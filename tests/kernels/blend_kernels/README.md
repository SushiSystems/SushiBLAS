# Tests of blend_kernels

The tests of the `blend_kernels` module, built into `sushiblas_blend_kernels_test_objects` and linked
into `sushiblas_kernels_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `reference/test_blend_transforms.cpp` | Tests the reference blend of transforms against hand values and a direct loop |
| `sycl/test_blend_transforms.cpp` | Tests the SYCL blend of transforms against the reference on the host device |
| `test_blend.cpp` | Tests that blend.hpp alone reaches the reference and the SYCL blend kernels |

# Tests of blend_kernels

The tests of the `blend_kernels` module, built into `sushiblas_blend_kernels_test_objects` and linked
into `sushiblas_kernels_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `compile_blend.cpp` | Compile check, never run: `blend.hpp` stands alone and declares the reference and the SYCL blend; the values are tested through the two `test_blend_transforms.cpp` files |
| `test_blend_index.cpp` | Tests the blend's offsets against the packed buffers, the live-slot test, the implied last row and the homogeneous w |
| `reference/test_blend_transforms.cpp` | Tests the reference blend of transforms against hand values and a direct loop |
| `sycl/test_blend_transforms.cpp` | Tests the SYCL blend of transforms against the reference on the host device |

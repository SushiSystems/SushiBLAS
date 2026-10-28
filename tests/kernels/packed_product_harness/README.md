# Tests of packed_product_harness

The tests of the `packed_product_harness` module, built into `sushiblas_packed_product_harness_test_objects` and linked
into `sushiblas_kernels_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_packed_product_harness.cpp` | Tests the harness helpers: the patterned values, the blockings and the problem it builds |

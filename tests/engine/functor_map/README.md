# Tests of functor_map

The tests of the `functor_map` module, built into `sushiblas_functor_map_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_read_binding.cpp` | Tests bind_reads, the bundle that declares the tensors an element functor reads |

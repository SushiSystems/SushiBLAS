# Tests of functor_map

The tests of the `functor_map` module, built into `sushiblas_functor_map_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_index_task.cpp` | Tests that an index task gives one result on the reference and the SYCL path, and that the host body of a late count reads its width when it runs |
| `test_map_task.cpp` | Tests that a map task and a zip task give one result on the reference and the SYCL path |
| `test_read_binding/test_read_binding.cpp` | Tests bind_reads, the bundle that declares the tensors an element functor reads |
| `test_read_binding/test_include_order.cpp` | Tests that a seam defined before `write_binding.hpp` is included still checks a write bundle's reads |
| `test_scalar.cpp` | Tests that ElementScalar holds one parameter in every element type, integers saturated |
| `test_scalar_binding.cpp` | Tests that bound scalars reach the functor after its elements, narrowed to their type |
| `test_write_binding.cpp` | Tests bind_writes, the bundle that declares the tensors an element functor writes |

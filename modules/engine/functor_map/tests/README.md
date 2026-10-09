# Tests of functor_map

The unit tests of the `functor_map` module, built into `sushiblas_functor_map_tests` by the
module's `CMakeLists.txt`. The source comments state what a test checks; a section below
holds why a test is written the way it is. `tests/unit/README.md` holds the reasoning the
tests share, and `tests/common/README.md` describes the fixtures.

| File | Tests |
| --- | --- |
| `test_read_binding.cpp` | Tests bind_reads, the bundle that declares the tensors an element functor reads |

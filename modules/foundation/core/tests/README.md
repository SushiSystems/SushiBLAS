# Tests of core

The unit tests of the `core` module, built into `sushiblas_core_tests` by the
module's `CMakeLists.txt`. The source comments state what a test checks; a section below
holds why a test is written the way it is. `tests/unit/README.md` holds the reasoning the
tests share, and `modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_blas_types.cpp` | Tests the numbers and the size of DataType and of the BLAS flags, and the flag queries |

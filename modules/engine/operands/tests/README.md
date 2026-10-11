# Tests of operands

The unit tests of the `operands` module, built into `sushiblas_operands_tests` by the
module's `CMakeLists.txt`. The source comments state what a test checks; a section below
holds why a test is written the way it is. `tests/unit/README.md` holds the reasoning the
tests share, and `tests/common/README.md` describes the fixtures.

| File | Tests |
| --- | --- |
| `test_blas_scalar.cpp` | Tests BLASScalar: what converts to it, what it narrows to and what it reports |

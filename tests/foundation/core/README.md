# Tests of core

The tests of the `core` module, built into `sushiblas_core_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_blas_types.cpp` | Tests the numbers and the size of the BLAS flags, and the flag queries |
| `test_common.cpp` | Tests the names, the element sizes and the kind predicates of DataType, and Layout |
| `test_errors.cpp` | Tests that UnsupportedExecution is an invalid_argument that keeps its message |
| `test_export.cpp` | Tests that SUSHIBLAS_API names the symbol visibility of the platform |
| `test_format.cpp` | Tests Core::format: placeholders, escaped braces, width, fill, precision and bases |
| `test_logger.cpp` | Pins that the logger writes every line before an assertion aborts or a shutdown returns |
| `test_sparse_types.cpp` | Tests the numbers, the validity, the first index and the names of IndexBase |
| `test_traits.cpp` | Tests that is_complex holds for std::complex and for nothing else |
| `test_version.cpp` | Tests that the version macros agree with each other and with their encoding |

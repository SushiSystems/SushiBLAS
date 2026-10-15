# Tests of typed_suite_naming

The tests of the `typed_suite_naming` module, built into `sushiblas_typed_suite_naming_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_default_type_names.cpp` | Tests that DefaultTypeNames names a type by its index and that the suite macro uses it |

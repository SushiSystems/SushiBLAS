# Tests of sparse_fixtures

The tests of the `sparse_fixtures` module, built into `sushiblas_sparse_fixtures_test_objects` and linked
into `sushiblas_foundation_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_integer_csr.cpp` | Tests that integer_csr builds a valid CSR matrix of small integers and its dense twin |
| `test_sparse_samples.cpp` | Tests the sample sparse matrices, the index storage and the host product they give |

# Tests of operands

The tests of the `operands` module, built into `sushiblas_operands_test_objects` and linked
into `sushiblas_engine_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_argument_check.cpp` | Tests the record-time checks on a flag, a scalar, a dtype and a refused transpose |
| `test_blas_metadata.cpp` | Tests that BLASMetadata names the task and fills its slots in the order given |
| `test_matrix_operand.cpp` | Tests how a tensor is read as a matrix operand and the checks between two of them |
| `test_scalar.cpp` | Tests BLASScalar: what converts to it, what it narrows to and what it reports |
| `test_scalar_result.cpp` | Tests the real dtype of a scalar result and the check on the tensor that receives it |
| `test_scratch_lease.cpp` | Tests when a BLAS call leases scratch, how much, and that the lease is declared written |
| `test_vector_operand.cpp` | Tests how a tensor is read as a vector operand, strided ones included, and its checks |

# Tests of tensor_expectations

The tests of the `tensor_expectations` module, built into `sushiblas_tensor_expectations_test_objects` and linked
into `sushiblas_data_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_scalar_dtype.cpp` | Tests that sushiblas_dtype names the DataType of each element type a test uses |
| `test_tensor_comparison.cpp` | Tests that the tensor comparisons accept a match and report a mismatch, NaN included |
| `test_tensor_fill.cpp` | Tests that fill_tensor converts its values to each dtype and that fill_random repeats |
| `test_tensor_readback.cpp` | Tests that read_tensor copies a tensor's elements out and refuses another element type |

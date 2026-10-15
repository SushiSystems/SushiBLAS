# Tests of tensor

The tests of the `tensor` module, built into `sushiblas_tensor_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_host_data_type.cpp` | Tests that HostDataType gives each host element type its DataType |
| `test_storage.cpp` | Tests that Storage rounds its size up, aligns its allocation and refuses a bad request |
| `test_tensor.cpp` | Tests the strides, views, checks and regions of a Tensor over one storage |
| `test_tensor_factory.cpp` | Tests the layout, the byte size and the tensors a TensorFactory creates |

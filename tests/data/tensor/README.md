# Tests of tensor

The tests of the `tensor` module, built into `sushiblas_tensor_test_objects` and linked
into `sushiblas_data_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_block_maker.cpp` | Tests that the host and runtime adapters of BlockMaker size, place and report their blocks |
| `test_host_block.cpp` | Tests that HostBlock owns aligned host memory and hands it over on a move |
| `test_host_data_type.cpp` | Tests that HostDataType gives each host element type its DataType |
| `test_runtime_block.cpp` | Tests that RuntimeBlock allocates or borrows a buffer and copies through its device queue |
| `test_storage_block.cpp` | Tests the defaults and helpers of the StorageBlock port through an in-memory adapter |
| `test_storage/test_storage.cpp` | Tests that Storage rounds its size up, aligns its allocation, refuses a bad request, views a buffer it borrows and forwards to its block |
| `test_storage/test_storage_block_contract.cpp` | Tests that the host and runtime adapters of StorageBlock keep the StorageBlock contract |
| `test_tensor/test_tensor.cpp` | Tests the strides, views, checks and regions of a Tensor over one storage |
| `test_tensor/test_shape_only_tensor.cpp` | Tests that a shape-only Tensor carries dims, strides, dtype and layout and no storage |
| `test_tensor_factory/test_tensor_factory.cpp` | Tests the layout, the byte size and the tensors a TensorFactory creates |
| `test_tensor_factory/test_tensor_factory_contract.cpp` | Tests that the host and runtime adapters of TensorFactory keep the TensorFactory contract, and that a host factory has no runtime |

## Contract suites

`StorageBlockContract` and `TensorFactoryContract` are typed suites: the checks live in
`tests/support/storage_block_contract.hpp` and `tests/support/tensor_factory_contract.hpp`, and the
adapters, `HostBlockSubject`, `RuntimeBlockSubject`, `HostTensorSubject` and `RuntimeTensorSubject`,
in `tests/support/tensor_subjects.hpp`.
Each suite is declared twice, as `Small_` on the host adapter and `Medium_` on the runtime one, since
the size is read from the suite name.

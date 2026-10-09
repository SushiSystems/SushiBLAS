# tensor

The dense containers: `Storage`, the reference-counted USM allocation; `Tensor`, the typed and
strided view over it; and `TensorFactory`, which allocates tensors from one runtime context.

It depends on `core`. Its headers are included as `<SushiBLAS/tensor/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `host_data_type.hpp` | HostDataType, which maps a host element type to its DataType |
| `storage.hpp` | Storage, the reference-counted USM allocation a tensor's elements live in |
| `tensor.hpp` | Tensor, a typed, strided view over a Storage |
| `tensor_factory.hpp` | TensorFactory, which allocates dense tensors from one runtime context |

# tensor

The dense containers: `Storage`, the reference-counted USM allocation; `Tensor`, the typed and
strided view over it; and `TensorFactory`, which allocates tensors from one runtime.

It depends on `core`. Its headers are included as `<SushiBLAS/tensor/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `host_data_type.hpp` | HostDataType, which maps a host element type to its DataType |
| `storage.hpp` | Storage, the reference-counted USM allocation a tensor's elements live in |
| `tensor.hpp` | Tensor, a typed, strided view over a Storage |
| `tensor_factory.hpp` | TensorFactory, which allocates dense tensors from one runtime |

## Borrowing memory

`make_sushi<Storage>(Storage::Borrowed{}, runtime, std::move(buffer))` builds a `Storage` over a
`SushiRuntime::Buffer<std::byte>` whose memory another owner keeps alive, such as the result of
`Interop::borrow`. It allocates nothing on the device and copies nothing; the cost of a call is the
one reference-counted `Storage` object. The device, the residency and the size are the buffer's,
and the size is not rounded. The address must be aligned to `alignof(std::max_align_t)`, not to
`K_STORAGE_ALIGNMENT`. An empty buffer, a misaligned one and a device the runtime does not drive
throw `std::runtime_error`.

An `OperandSet` hands the owner's observer of each borrowed storage to the runtime, so a run that
names a tensor after the owner let go is refused with `DanglingResource`. The memory must be
reachable by the device a task runs on.

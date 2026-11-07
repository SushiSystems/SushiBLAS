# tensor

The dense containers: `Storage`, the reference-counted allocation; `Tensor`, the typed and
strided view over it; and `TensorFactory`, which allocates tensors from one runtime or, in its host
form, from host memory.

It depends on `core`. Its headers are included as `<SushiBLAS/tensor/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `host_data_type.hpp` | HostDataType, which maps a host element type to its DataType |
| `host_block.hpp` | HostBlock, aligned host memory that no runtime owns |
| `storage.hpp` | Storage, the reference-counted allocation a tensor's elements live in |
| `tensor.hpp` | Tensor, a typed, strided view over a Storage, or over none for validation |
| `tensor_factory.hpp` | TensorFactory, which allocates dense tensors from one runtime or from host memory |

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

## Host storage and the host factory

`make_sushi<Storage>(Storage::Host{}, n_bytes)` builds a `Storage` over a `HostBlock`: aligned host
memory that no `SushiRuntime::Runtime` owns, freed with the storage. The size is rounded up to
`K_STORAGE_ALIGNMENT` and the base is aligned to it, as for a runtime storage. A tensor's `data()` and
`allocation_base()` work, `residency()` reports `Residency::Host`, `device_index()` is 0 and `is_host()`
is true. `queue()` throws `UnsupportedExecution`, because no device owns the memory. A zero size throws
`std::runtime_error`.

`TensorFactory(TensorFactory::Host{}, layout)` needs no runtime and makes host storages. `create`
ignores the residency it is given, accepts only device index 0, and returns a tensor without storage
for a shape with a zero dimension, as the runtime factory does. `runtime()` throws
`UnsupportedExecution`; `is_host()` tells the two factories apart. The constructor that takes a
runtime, and its `create`, are unchanged.

## Shape-only tensors

`Tensor::make_shape_only(dims, dtype, layout)`, and the overload that takes strides, build a tensor with
dims, strides, dtype and layout and no storage; its `shape_only` field is true, and `transpose`,
`reshape` and `slice` keep it. A check that reads only shape and dtype runs on it without a runtime.
`data()` and `is_aligned()` throw `std::runtime_error`, `device()` throws, and
`Dispatch::require_storage` refuses it with `std::invalid_argument` naming it as shape-only when it holds
elements. It is not a placeholder for an allocation that has not happened: nothing can run on it.

## Tests

`StorageContract` and `TensorFactoryContract` are typed suites in `tests/data/tensor/`, `Small_` on the
host adapter and `Medium_` on the runtime one; the shared checks are under `tests/support/`.

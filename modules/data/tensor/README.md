# tensor

The dense containers: `Storage`, the reference-counted allocation; `Tensor`, the typed and
strided view over it; and `TensorFactory`, which allocates tensors from one runtime or, in its host
form, from host memory. `Storage` reaches its bytes through the `StorageBlock` port, which `HostBlock`,
`RuntimeBlock` and `PlannedBlock` implement.

It depends on `core`. Its headers are included as `<SushiBLAS/tensor/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `host_data_type.hpp` | HostDataType, which maps a host element type to its DataType |
| `storage_block.hpp` | StorageBlock, the port a Storage reads its bytes through, and `K_STORAGE_ALIGNMENT` |
| `block_maker.hpp` | BlockMaker, the port a TensorFactory gets its blocks through |
| `host_block.hpp` | HostBlock and HostBlockMaker, aligned host memory that no runtime owns, and `make_host_storage` |
| `host_handle_space.hpp` | `host_handle_space()`, the one process-wide LocalHandleSpace that mints the handle of every HostBlock, and `K_HOST_HANDLE_SPACE_ID` |
| `planned_block.hpp` | PlannedBlock, over a `Residency::Planned` SushiRuntime buffer, which has a handle and no host address |
| `runtime_block.hpp` | RuntimeBlock and RuntimeBlockMaker over a SushiRuntime buffer, `make_runtime_storage` and `borrow_runtime_storage`; the one header here that includes `SushiRuntime/sushiruntime.hpp` |
| `storage.hpp` | Storage, the reference-counted allocation a tensor's elements live in |
| `tensor.hpp` | Tensor, a typed, strided view over a Storage, or over none for validation |
| `tensor_factory.hpp` | TensorFactory, which allocates dense tensors from one runtime or from host memory |

## The block contract

Every `StorageBlock` answers `data()`, `size_bytes()`, `residency()`, the two copies,
`owner_liveness()` and `global_handle()`. A block answers `device_index()` and `queue()` only when one
device owns its memory: `device_index()` is a `std::optional<std::size_t>` and `queue()` is null
otherwise. `HostBlock` names device 0, which works on host memory, and has no queue. `PlannedBlock` names
no device and has no queue, because the runtime places the bytes. `Storage::device_index()` forwards the
optional; a caller that needs a device calls `Storage::require_device_index(operation)`, which throws
`UnsupportedExecution` naming the operation when the allocation is planned. Tensor packs, late counts,
reduction scratch and scratch leases call it, and so refuse a planned operand with a message.

`copy_to_host` and `copy_from_host` of every block return with the copy done and throw
`std::runtime_error` for a range outside the block. `RuntimeBlock` copies through the buffer's
`read_range` and `write_range`, as `PlannedBlock` does, so the runtime sees each copy and a
`Residency::Device` buffer, which a runtime configured to guard device memory page-protects against
host access, is opened for it. Host code never submits a raw `queue()->memcpy` on a runtime buffer outside a
recorded task. A read stages the bytes in a `std::vector` before the copy to the caller.

## Handle

`Tensor::global_handle()` returns the `SushiRuntime::GlobalHandle` of the allocation the tensor views,
through `Storage::global_handle()` and `StorageBlock::global_handle()`. `RuntimeBlock` returns the
handle of the buffer it holds, for an allocated and for a borrowed block. `HostBlock` returns a handle
of the host handle space (below), and a tensor with no storage returns the null handle. Every view of
one storage reports one handle; `region()` names the bytes of the view within it.

### The host handle space

`host_handle_space.hpp` declares `host_handle_space()`, one `SushiRuntime::LocalHandleSpace` for the
whole process: a function-local static, so every translation unit that includes the header shares
it. Its space id is `K_HOST_HANDLE_SPACE_ID` (1), carried in the machine field of each handle; the
host double of `graph_primitives` takes 2, and any other SushiBLAS space needs an id of its own.
Each `HostBlock` that owns memory mints one handle of kind `Buffer` in its constructor, before it
allocates, and the handle stays with the block through a move. A default or moved-from block owns
nothing and returns the null handle. Minting takes no lock and the slots count in mint order, so the
same sequence of allocations gives the same handles. The constructor throws `std::runtime_error`
after 2^32 mints.

The `region()` of a host tensor is therefore valid: it overlaps the regions of its own storage that
share bytes with it and nothing else, so a host sink orders the tasks that name the same host
storage. The handles carry the local universe, which no runtime serves, so a runtime refuses them
as a foreign universe and none is ever given to a runtime graph.

## Borrowing memory

`borrow_runtime_storage(runtime, std::move(buffer))` builds a `Storage` over a
`SushiRuntime::Buffer<std::byte>` whose memory another owner keeps alive, such as the result of
`Interop::borrow` for foreign memory, or `Buffer::as_bytes()` for a buffer a runtime allocated, which
keeps that buffer's handle. It allocates nothing on the device and copies nothing; the cost of a call is the
one reference-counted `Storage` object. The device, the residency and the size are the buffer's,
and the size is not rounded. The address must be aligned to `alignof(std::max_align_t)`, not to
`K_STORAGE_ALIGNMENT`. An empty buffer, a misaligned one, a buffer that states no device (a planned
one, `runtime_block.hpp`) and a device the runtime does not drive throw `std::runtime_error`.

An `OperandSet` hands the owner's observer of each borrowed storage to the runtime, so a run that
names a tensor after the owner let go is refused with `DanglingResource`. The memory must be
reachable by the device a task runs on.

## Host storage and the host factory

`make_host_storage(n_bytes)` builds a `Storage` over a `HostBlock`: aligned host
memory that no `SushiRuntime::Runtime` owns, freed with the storage. The size is rounded up to
`K_STORAGE_ALIGNMENT` and the base is aligned to it, as for a runtime storage. A tensor's `data()` and
`allocation_base()` work, `residency()` reports `Residency::Host`, `device_index()` holds 0 and its block has
no queue (`block().queue()` is null). `Storage::queue()` throws `UnsupportedExecution`, because no device owns the memory. A zero size throws
`std::runtime_error`.

`TensorFactory(TensorFactory::Host{}, layout)` needs no runtime and makes host storages. `create`
ignores the residency it is given, accepts only device index 0, and returns a tensor without storage
for a shape with a zero dimension, as the runtime factory does, but refuses `Residency::Planned`,
`planned` and `from_file`. The factory owns a `BlockMaker`, `HostBlockMaker` or `RuntimeBlockMaker`, that its
constructors build. The constructor that takes a runtime, and its `create`, are unchanged.

## Planned storage

`TensorFactory::planned(dims, dtype)` makes a tensor over a `PlannedBlock`, a `Residency::Planned`
buffer of `Runtime::buffer(count, Residency::Planned)`; `create` with `Residency::Planned` makes the same.
`TensorFactory::from_file(path, offset, dims, dtype)` makes one over `Runtime::buffer_from_file`, whose
bytes are the file range; its size is the tensor's bytes, not rounded. Both refuse a shape with no
element. The runtime places the bytes and may move them, so `data()` of the block is null,
`Tensor::data()` throws, `device_index()` is empty and the block has no queue. `global_handle()` is the
buffer's handle. `copy_to_host` and `copy_from_host` go through the buffer's `read_range` and
`write_range`.

An operation that records the address of each operand calls `Dispatch::require_storage`, which
refuses a planned operand with `std::invalid_argument` naming the operation, before any address is
read. The operations that record by handle (GEMM and COPY of BLAS, the elementwise executors and
the broadcast executor, see [the engine](../../engine/engine/README.md)) call `Dispatch::require_bindable_storage` and take
a planned tensor.

## Shape-only tensors

`Tensor::make_shape_only(dims, dtype, layout)`, and the overload that takes strides, build a tensor with
dims, strides, dtype and layout and no storage; its `shape_only` field is true, and `transpose`,
`reshape` and `slice` keep it. A check that reads only shape and dtype runs on it without a runtime.
`data()` and `is_aligned()` throw `std::runtime_error`, `device()` throws, and
`Dispatch::require_storage` refuses it with `std::invalid_argument` naming it as shape-only when it holds
elements. It is not a placeholder for an allocation that has not happened: nothing can run on it.

## Tests

`StorageBlockContract` and `TensorFactoryContract` are typed suites in `tests/data/tensor/`, `Small_` on the
host adapter and `Medium_` on the runtime one; `PlannedBlock` passes the queries every block answers in
`Medium_PlannedStorageBlockContract`; the shared checks are under `tests/support/`.

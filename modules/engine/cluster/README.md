# cluster

The tensor side of SushiRuntime's distributed layer: tensors packed into one shared buffer, the
buffer shared with a cluster, and a registered task over packs recorded into the caller's
graph. The design is in `docs/design/LAYERED_EXECUTION/README.md`, section "The cluster module".

The module is added only when `SB_ENABLE_DISTRIBUTED` is on, and needs SushiRuntime built with
`SR_ENABLE_DISTRIBUTED`, which is when the `SushiRuntime::distributed` target exists.

It depends on `core`, `engine` and `tensor`, and on `SushiRuntime::distributed`. Its headers are included
as `<SushiBLAS/cluster/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `bound_pack.hpp` | `BoundPack` and `PackSpan`, the typed segments and tensors of a pack a kernel received |
| `cluster_binding.hpp` | `ClusterBinding`, the cluster and the kernel registry one engine works with |
| `pack_layout.hpp` | `PackLayout`, `PackEntry` and `PackSegment`, where each tensor of a pack lies |
| `remote_task.hpp` | `record_remote()`, `bind_views()`, `remote_operation()` and `remote_operation_name()` |
| `tensor_pack.hpp` | `TensorPack`, the buffer of one layout, tensor views over it and the gather and scatter tasks |

## pack_layout.hpp

`PackLayout::of()` takes an ordered list of tensors and reads only their type, layout and shape.
It gives each data type one segment, in the order of `Core::DataType` (`Half`, `Float32`,
`Float64`, `Complex32`, `Complex64`, `Int32`, `Int64`, `UInt32`), and places the tensors of a type
inside its segment in the order given, with no gap. A segment that holds anything starts on a multiple
of `K_PACK_SEGMENT_ALIGNMENT`, 128 bytes; an empty segment takes no bytes. `size_bytes()` ends at
the end of the last segment. A list that is empty or holds no element is refused.

### The fingerprint

`fingerprint()` is the 64-bit FNV-1a hash of this byte sequence, every field little-endian
whatever the host:

| Field | Bytes |
| --- | --- |
| `K_PACK_FINGERPRINT_VERSION` (2) | 4 |
| Entry count | 8 |
| Per entry: data type | 1 |
| Per entry: size of the type in bytes | 8 |
| Per entry: layout (`RowMajor` 0, `ColumnMajor` 1) | 1 |
| Per entry: rank | 4 |
| Per entry: each extent | 8 each |
| Per entry: offset and size in bytes | 8 and 8 |
| Per segment, all eight: offset and size in bytes | 8 and 8 |
| Pack size in bytes | 8 |

Two builds that disagree on any of these, a type size among them, get different fingerprints. A
change to the sequence raises `K_PACK_FINGERPRINT_VERSION`.

### Byte order

A pack carries elements as raw bytes. `pack_layout.cpp` refuses to compile on a host whose
`__BYTE_ORDER__` is not `__ORDER_LITTLE_ENDIAN__`, so a big-endian build never exists; a compiler
that does not define the macro fails the same check. Every toolchain the stack builds with is
clang-based and defines it.

## tensor_pack.hpp

`TensorPack` allocates a `Storage` of the layout's size as `Residency::Shared` on device 0, so
`Cluster::share()`, which refuses device-resident memory, accepts it. Copies of a pack name the
same buffer.

`TensorPack::tensor(index)` returns a `Tensor` over the place of one entry, with the entry's type,
layout and shape, and `TensorPack::segment_tensor(dtype)` a one-dimensional `Tensor` over the whole
segment of a type, every entry of that type one after the other. A type no entry has gives an empty
tensor. Both are views of the pack's own `Storage`: nothing is copied or allocated beyond the
`Tensor`, and the view holds the storage, so it outlives the pack. `tensor()` throws
`std::out_of_range` for an index past the entry count. The two names follow
`BoundPack::tensor()` and `BoundPack::segment()`; `view()` already names the `BufferView` of the
buffer. `BoundPack` offers no tensor views: it holds a raw address, not a `Storage`, and the
worker side has no use for them yet. A caller can sum the same segment of several packs with
`ElementwiseOps::ordered_sum()` without scattering any of them first.

`record_gather()` records through a `TaskRecorder` one task that copies each tensor into its
place, and `record_scatter()` one that copies each place into its tensor. The pack enters the
task's `OperandSet` as a `Float32` tensor over the whole buffer, whose size is a multiple of 128. Each takes one tensor per entry in
the layout's order, refuses a tensor whose type, layout or shape differs from its entry, and
refuses one that is not contiguous. The task reads the tensors and writes the pack, or the
reverse, and keeps every storage it touches alive. The copies run one after the other on the
task's queue, under every execution mode: a `memcpy` between shared allocations has no separate
reference path.

## cluster_binding.hpp

`ClusterBinding` holds a `Cluster` handle and the `KernelRegistry` the engine's remote tasks run.
It exposes `share()`, `is_shared()`, `has_worker_for()` and `kernels()`, never the cluster
itself. `share()` registers the pack's buffer with the master; the buffer leaves the registry when
the last copy of the pack frees it, through the allocator's release observer.

## remote_task.hpp

`record_remote()` is the tensor form of `record_registered_task()`. It takes the caller's
`SushiRuntime::Recorder`, never an engine graph: a caller with a `TaskRecorder` passes the
runtime recorder it built that `TaskRecorder` on. Each pack becomes one `BufferView`, inputs
first. It refuses a null pack, a pack the binding has not shared, an operation the registry has
no kernel for and a recording the runtime refuses. The node does not keep the packs alive, so
they must outlive the run.

`bind_views()` runs on the machine that runs the kernel. It takes the views the kernel received
and one layout per view, refuses a different count and a view whose size is not its layout's,
and returns one `BoundPack` per view. `BoundPack::segment()` and `BoundPack::tensor()` return a
`PackSpan`, whose `as<T>()` refuses an element type other than the span's.

`remote_operation_name(name, fingerprint)` returns `<name>:<16 lower-case hex digits>` and
`remote_operation()` its `OperationID`. A worker registers its kernel under it, so a worker whose
layout differs advertises another operation and the master never sends it the task; the task
then runs on the master. `bind_views()` checks sizes only, so the operation name is what refuses
a layout of the same size.

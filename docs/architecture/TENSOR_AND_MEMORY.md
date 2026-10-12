# 4. `Tensor`, `Storage` and memory

A `Tensor` (`tensor/tensor.hpp`) is metadata: `shape`, `strides`, `rank`, `dtype`, `layout`,
`num_elements`, a `storage` pointer and a `storage_offset`. `transpose()`, `reshape()` and
`slice()` return a new `Tensor` over the same `SushiRuntime::sushi_ptr<Storage>` and move no
data. `K_MAX_TENSOR_RANK` is 6.

`Core::DataType` is `Half`, `Float32`, `Float64`, `Complex32`, `Complex64`, `Int32`, `Int64`, with
the values 0 to 6 written out in `core/common.hpp`. The values are a file-format contract: none
changes, and a new type is appended. `Core::bytes_per_element()` is the one mapping from a dtype
to its width.

**The integer types are index and mask types.** `iamax`, `argmax`, `argmin` and
`count_nonfinite` answer with an `INT64` because an index or a count stored in a float is exact
only up to the mantissa. Beyond those results the integer dtypes are served by `copy`,
`constant`, the comparison and logic family, `io`, the moves that do no arithmetic (`gthr`,
`gthrz`, `sctr`, `concat_channels`, `slice_channels`), and the elementwise operations that have an
integer meaning (`add`, `sub`, `mul`, `neg`, `abs`, `min`, `max`, `clamp`), whose arithmetic
wraps. Every other operation rejects them by name at record time, through `Dispatch::Domain`
([section 6](DISPATCH_SEAMS.md)).

**Layout and the batch axis.** `Core::Layout` (`RowMajor`, `ColumnMajor`) is fixed at
construction and describes the order inside one matrix. `Tensor::layout_strides()` is the single
definition: the constructor lays a tensor out with it and `is_contiguous()` compares against it.
For rank 3 and above the trailing two axes are the matrix and take the layout's order, and every
leading batch axis is outermost in either layout, so a rank-3 `{b, m, n}` tensor has batch stride
`m * n` under both. That matches how `Dispatch::batch_stride` takes the element count of one
matrix as the distance between matrices.

**Offsets count elements.** `storage_offset` is in elements, and `data()` scales it by the
element width, so a view is aligned for its own element type. A view that is not contiguous, such
as a slice along an inner axis, keeps its parent's strides and is built through the
stride-explicit constructor. `Tensor::element_span()` gives the distance from its first element
to its last, `1 + sum((shape[i] - 1) * strides[i])`, which is what the capacity check and
`region()` measure.

An operation addresses a tensor through three accessors and never through the allocation base,
which is private to `Storage`:

| Accessor | Gives | Used for |
| --- | --- | --- |
| `data()` / `data_as<T>()` | The view's first element | Every kernel pointer |
| `data_or_null()` | The same, or `nullptr` without storage | A source that tolerates an empty operand |
| `region()` | The view's byte interval, keyed on the allocation base | Every declared read and write |

`region()` is what lets a memory planner pack an arena. Two disjoint views of one allocation give
regions that do not overlap, so the tracker does not order them, and two overlapping views are
ordered.

## Storage and alignment

`Storage` (`tensor/storage.hpp`) owns one reference-counted USM allocation with a
`SushiRuntime::Residency` (`Shared`, `Device` or `Host`) on one device queue of a
`SushiRuntime::Runtime`. The allocation is a `SushiRuntime::Buffer<std::byte>` from
`Runtime::buffer`, which takes the alignment as an argument. Every allocation base is aligned to
`K_STORAGE_ALIGNMENT`, 128 bytes, and the allocated size is the requested size rounded up to
that. An address off that boundary fails an `SB_ASSERT`. The buffer shares ownership of the runtime's state, so a `Storage` may outlive the `Runtime`
object it was allocated from. The allocation is freed when the last reference to the `Storage`
goes. A failed allocation throws `std::runtime_error` with the runtime's message, and so do a
size of zero and a device index the runtime does not drive. `Storage::queue()` returns the
`sycl::queue` of the device that owns the allocation.

`TensorFactory` (`tensor/tensor_factory.hpp`) allocates dense tensors of one default layout from
one `SushiRuntime::Runtime`, which must outlive it. `create(dims, dtype, residency, device_index)` checks the rank, the sign of each
dimension, the byte count for overflow and the device index, and returns a tensor without storage
when a dimension is 0. `Engine::create_tensor` forwards to it.

## Retention

A recorded task keeps its operands' storage alive. `OperandSet` takes one reference to the
`Storage` of every tensor declared on it, in an `Internal::StorageKeepAlive` (six references
inline, the rest on the heap), and `record_pair` copies that list into the task. The storage of a
tensor the caller drops between the record call and `execute()` is therefore still referenced when
the task runs. A pointer hidden inside a bare functor is not a declared operand: nothing retains it,
orders against it, or checks it under `ForceReference`.

## The scratch pool

An operation that needs device scratch (the staged partials of a fold, the panels of a packed
product, the wide solution of a substitution, the partial sums of a sparse scatter) leases it at
record time and the task holds the lease by value.

- `ScratchPool` (`engine/scratch_pool.hpp`) owns a reference-counted `Detail::ScratchShelf`
  (`engine/scratch_shelf.hpp`) of idle device blocks per device queue. The engine owns one pool
  and the recorder hands it out through `scratch_pool()`.
- `ScratchBuffer` (`engine/scratch.hpp`) is the lease. The block goes back to the shelf when the
  last copy of the handle is destroyed, which for a recorded task is when the graph drops the
  task.
- Blocks are pooled in six size classes from 128 to 4096 bytes, at most 64 idle blocks per class
  and device. A larger request is allocated for the lease and freed when it ends.
  `ScratchPool::trim()` frees the idle blocks.
- `OperandSet::scratch(buffer)` declares the lease as a written region of the task and retains
  its storage.

Under `ForceReference` the reduction seams and `Internal::allocate_blas_scratch` lease nothing:
a host fold stages no partials.

# functor_map

The per-element seam: the map and zip task bodies, and the bindings that declare the tensors and
narrow the scalars a functor reads. `elementwise`, `nonlinear` and `logic` record their operations
through it.

It depends on `core`, `support`, `tensor` and `engine`. Its headers are included as
`<SushiBLAS/functor_map/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `index_task.hpp` | The task body of the per-index seam, with its serial host path and its SYCL path |
| `map_task.hpp` | The task bodies of the per-element seams, each with its serial host path and its SYCL path |
| `read_binding.hpp` | The binding that declares the tensors an element functor reads and hands them to it as typed pointers |
| `scalar.hpp` | ElementScalar, a double operation parameter narrowed once to every element type |
| `scalar_binding.hpp` | The binding that narrows an operation's scalar parameters on the host before its functor reaches a kernel |
| `write_binding.hpp` | The binding that declares the tensors an element functor writes and hands them to it as typed pointers |

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to.

## The per-element seam

The executors in `elementwise/dispatch.hpp`, `nonlinear/dispatch.hpp` and `logic/dispatch.hpp`
share one task body. Each checks its operands, declares them in an `OperandSet` and records a
`Detail::MapTask` (one output from some inputs) or a `Detail::ZipTask` (N tensors advanced
together) through `TaskRecorder::record_pair`. The functor the caller hands over is one of three
things:

- a bare callable over elements;
- a `bind_scalars(fn, s0, s1, ...)` bundle, whose scalars the seam narrows to the element type on
  the host and passes after the elements;
- a `bind_reads(fn, read_as_element(t0), read_as<U>(t1), ...)` bundle, whose tensors the seam
  checks, declares on the task and passes as typed pointers after the elements. `fn` may itself
  be a `bind_scalars` bundle, and its scalars then come last;
- a `bind_writes(bind_reads(fn, ...), write_as_element(t0), write_as<U>(t1), ...)` bundle, whose
  tensors the seam checks, declares as written on the task and passes as writable typed pointers
  after the read pointers: `fn(elements..., reads..., writes..., scalars...)`.

### `functor_map/map_task.hpp`

`MapTask` and `ZipTask` hold the dtype, the base addresses, the element count and the functor. Under
`ForceReference` the body runs `Serial::map` or `Serial::zip_inplace` on the host; under any other
mode it submits one kernel through `Launch::map` or `Launch::zip_inplace`, with the launch shape
planned from the queue's device. Both paths call the same functor, so the host path is not a second
implementation of the operation. `make_map_task` builds a `MapTask` from the seam's arguments.

### `functor_map/scalar_binding.hpp`

`bind_scalars` returns a `ScalarBound`, which holds each scalar as an `ElementScalar` and stays on
the host. Inside the `with_element_type` arm the seam calls `narrow_for(tag, op)`, which returns a
`NarrowedOp<Fn, T, N>` holding the scalars in the element type `T`; that is what the kernel
captures. A HALF or FLOAT32 kernel therefore carries no double. A plain functor passes through
`narrow_for` unchanged.

### `functor_map/read_binding.hpp`

A functor that dereferences a tensor the seam does not know about, such as a per-channel table,
reads memory the task never declared, and the graph is free to run the task that writes it at the
same time. `bind_reads` closes that: `read_as_element(t)` names a tensor read as an array of the
operation's own element type and `read_as<U>(t)` one read as an array of `U`. The seam then

- checks each read with `ReadBound::require`: it must hold elements, have storage, hold the dtype
  it was declared with and be contiguous, and its device must be able to compute in that dtype;
- declares each read with `ReadBound::declare_into`, which refuses a read that overlaps a region
  the task writes;
- keeps only the base addresses in the task (`AddressedOp`) and hands the kernel a `ReadingOp`
  typed for `T`.

Declaring a read also retains its storage until the task is destroyed, and a device-only read is
refused under `ForceReference` with the operands. A bare functor declares nothing:
`require_declared_reads` and `declare_reads` are no-ops for it.

### `functor_map/write_binding.hpp`

`bind_writes` is the counterpart of `bind_reads` for the tensors a functor writes through a
pointer: `write_as_element(t)` names a tensor written as an array of the operation's own element
type and `write_as<U>(t)` one written as an array of `U`. Both take a tensor that outlives the
call, never a temporary. `bind_writes` wraps a bare functor, a `bind_scalars` bundle or a
`bind_reads` bundle; reads bind inside, so the call order is
`bind_writes(bind_reads(fn, reads...), writes...)`. The seam then

- checks each write with `WriteBound::require`, which applies the checks of a read under the
  operand name `write k`;
- declares each write with `WriteBound::declare_into`, after the reads were declared with
  `declare_reads`;
- keeps only the base addresses in the task (`WriteAddressedOp`, held as the functor of the
  `AddressedOp` when the bundle also reads) and hands the kernel a `WritingOp` typed for `T`.

A tensor named in a read and in a write of one task is one region read and written: the read stays
declared once by `declare_reads`, and `declare_into` adds the write once and skips a write whose
region the task already holds, so a tensor named in two writes is declared once too. A write that
overlaps a region the task reads without being that region is refused, because the seam would
declare one tensor under two views. The seam cannot see which element an index writes: whole
tensors are declared as written, and indices of one call that write one element are ordered by the
caller.

The seam calls `require_declared_reads` and `require_declared_writes`, then `declare_reads` and
`declare_writes` in that order, and records `task_functor` with the kernel built by `kernel_op`.
A bundle that declares writes does not suit the elementwise, nonlinear and logic seams, which
write their own output.

`element_dtype_of` returns the dtype of the first `read_as_element` or `write_as_element` tensor of a
bundle, reads before writes, for a seam that has no tensor of its own to take the element type from.
`IndexOps` is the one caller.

The kernel functor is narrowed by `narrow_for`, which calls the `narrow<T>()` member of any op that
has one, as `ScalarBound` and `WriteAddressedOp` do.

## `functor_map/index_task.hpp`

`IndexTask` runs `op(i)` for each `i` below a fixed count, in order on the host under
`ForceReference` and as one kernel through `Launch::indexed` otherwise. The functor is the one
`kernel_op` builds, so a bundle with reads and writes is called as `fn(i, reads..., writes...)`.

A late count does not go through `IndexTask`. `record_index_task` takes the dtype branch at record
time and hands `TaskRecorder::record_late_pair` two things built from the same functor: the kernel
the runtime's dynamic node runs on the device, and `LateIndexHostRun`, the `ForceReference` body.
The host body reads the width from the count tensor when it runs, which is after the node that
wrote it, and refuses a width below 0 or above the capacity, as the runtime fails a run for the
same. `IndexOps` in [`index`](../../operations/index/README.md) records both.

## `functor_map/scalar.hpp`

### `ElementScalar`

A scalar parameter of an elementwise or non-linear op (`pow`'s exponent, `clamp`'s bounds, the
`leaky_relu` and `elu` alpha) is a `double` in the public signature. `ElementScalar` narrows it on
the host, once, to `sycl::half`, `float`, `double`, `int32_t` and `int64_t`, and the kernel reads
the member for its element type through `as<T>()`. A HALF or FLOAT32 kernel therefore does no
double arithmetic. The HALF member is the `double` rounded to binary32, then to binary16. Narrowing to an integer type saturates to the type's range, truncates toward
zero and maps NaN to 0. The task metadata records the `double` as given.

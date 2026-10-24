# index

`IndexOps`, the per-index seam: one task that calls a functor once for each index below a count,
on the device or on the host by the execution mode.

It depends on `core`, `tensor`, `engine` and `functor_map`. Its headers are included as
`<SushiBLAS/index/<name>.hpp>`. The task body that runs the functor is
`index_task.hpp` in [`functor_map`](../../engine/functor_map/README.md); the bindings that declare
the tensors are `read_binding.hpp` and `write_binding.hpp` there.

| File | Declares |
| --- | --- |
| `index_ops.hpp` | IndexOps, whose `for_each` records a per-index task with a fixed or a late count |

`source/` holds one translation unit: the checks and the operation id `IndexOps` shares across
its templates.

## `index_ops.hpp`

```cpp
SushiBLAS::IndexOps index(recorder);
index.for_each(n,
    SushiBLAS::Dispatch::bind_writes(
        SushiBLAS::Dispatch::bind_reads(projection, read_as<Constraint>(constraints)),
        write_as_element(positions)),
    SushiBLAS::OperationOptions{"pgs_project", SushiRuntime::Determinism::Bitwise});
```

The functor is called as `functor(i, reads..., writes...)`. The call takes a `bind_writes` bundle,
because a task that declares no write is not ordered against anything. The operation's element
type is the dtype of the first `read_as_element` or `write_as_element` tensor, reads before
writes. A call with typed bindings only has no element type and runs under `Float32` for the default
`Domain::Real`. The `Dom` template argument names the families a generic functor is well-formed
for, as it does for the elementwise seams; the default is `Domain::Real`.

**Disjoint writes.** The seam declares whole tensors as written and does not see which element an
index writes. Two indices of one call write the same element only if the caller orders them. A
colouring is one `for_each` per colour, recorded in order; the runtime orders the calls because
each declares the same tensor as written. Under `ForceReference` the indices run in order on the
host.

**A late count.** The second form takes a capacity and a count tensor: a contiguous `UINT32` tensor, or an
`INT32` one, with storage in `Residency::Host`, whose first element a previous node wrote. The task reads
the count tensor, so it waits for the node that wrote it. A bundle that writes the count tensor is
refused at record time, and so is a count tensor of another dtype or residency. The width is not
read by `IndexOps`: under `ForceSYCL` and `Auto` the task is the runtime's dynamic node
(`Recorder::dynamic_kernel` with `sized_from_device`), which reads the word when the node runs and
calls the functor for the indices below it. Under `ForceReference` the host body reads the count
after its producer ran and runs the indices in order. A width above the capacity, or a negative one,
fails the run on both paths; a width of 0 runs nothing.

`INT64` counts are refused instead of narrowed. The runtime names its width as a
`Buffer<std::uint32_t>` and reads one 32-bit word; a narrowing would be a copy of the count into
another tensor, a second path, or a silent loss of the high word. A caller that holds an `INT64`
count writes a `UINT32` one. `INT32` stays accepted because the runtime reads the same 32-bit
word; a negative one reads as a width above the capacity and fails the run.

A device-mode late-count task cannot hold the storage of its tensors itself, because the runtime's
kernel capture must be device copyable. The recorder holds it, and the runtime watches that hold, so
a run after the recorder is destroyed is refused instead of reading freed memory. The engine README's
section on `TaskRecorder::record_late_pair` states how long the recorder holds it.

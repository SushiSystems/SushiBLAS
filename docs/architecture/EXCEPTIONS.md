# 11. Exceptions and statuses

SushiBLAS reports a failure through one of four channels, and which one depends on when the
failure is found:

| When | Channel | What reports through it |
| --- | --- | --- |
| Record time and creation | An exception | A wrong operand, flag or scalar; a mode or a device that cannot run the operation; a tensor, storage or scratch block that cannot be made; a recording the runtime refuses |
| Run time | `SushiRuntime::Result<void>` | `Engine::execute()`: a graph that does not compile, a task that fails, a device fault, a run refused because another is in flight |
| A graph primitive | `SushiRuntime::Result<void>` | `reduce`, `segmented_reduce`, `prefix_scan`, `counting_sort`: a broken recording precondition as `ErrorCode::InvalidGraph`, a scratch block that cannot be allocated as `AllocationFailed`, or the recorder's first refusal since its `clear()` |
| A file operation | `IOStatus` | `IO::save`, `load`, `save_bin`, `load_bin`, `save_npy`, `load_npy`: a file defect, a mismatch, or `RunFailed` |

No exception leaves `Engine::execute()`, a graph primitive or the SushiRuntime calls below them.
`SB_ASSERT` is none of the four: it logs and aborts in a build without `NDEBUG`, and is for a
defect inside the library, such as an allocation the runtime returned off the storage alignment.

## Record time throws

A record call throws before any task reaches the graph, so the failure names the line that asked
for the operation. Three types cover the library:

| Type | Base | Thrown when |
| --- | --- | --- |
| `std::invalid_argument` | `std::logic_error` | An operand, flag, scalar or parameter is wrong: missing storage, a dtype, rank, shape, layout, stride or length mismatch, a dtype outside the operation's domain, a flag outside its enumeration, a complex scalar for a real tensor, a bad window geometry |
| `SushiBLAS::UnsupportedExecution` | `std::invalid_argument` | The operands are right and the operation cannot run under this mode or on this device: a device-only operand under `ForceReference`, FLOAT64 or COMPLEX64 storage on a device without `fp64`, HALF storage on a device without `fp16`, `poisson` on a device without `fp64` |
| `std::runtime_error` | `std::exception` | Creation: a `Tensor` or `Storage` rule (rank above 6, a negative dimension, an overflowing size, a failed allocation, a view past its storage), `TensorFactory::create`, `SparseBuilder` and `Sparse::assemble_csr`, a scratch allocation. A recording the runtime's recorder refuses. A failed run in `IO::read`, `write` and `to_string` |

`UnsupportedExecution` (`core/errors.hpp`) derives from `std::invalid_argument`, so a handler
written for the base still catches it, and a caller that wants the two apart lists it first:

```cpp
template<typename Call>
[[nodiscard]] Outcome outcome_of(const Call& call)
{
    try
    {
        call();
    }
    catch (const UnsupportedExecution&)
    {
        return Outcome::UnsupportedExecution;
    }
    catch (const std::invalid_argument&)
    {
        return Outcome::InvalidArgument;
    }

    return Outcome::Returned;
}
```

Source: `modules/engine/engine/tests/test_unsupported_execution.cpp`, where `Outcome` is an
enumeration of the test. The first handler takes a refusal by mode or device, the second a wrong
operand, flag or scalar.

Where each type comes from:

- The operand checks of `engine/operand_check.hpp` (`require_storage`, `require_dtype`,
  `require_same_shape`, `require_same_layout`, `require_same_element_count`,
  `require_min_elements`, `require_contiguous`, `require_rank`, `require_scalar`,
  `require_extent`, `require_operand`) throw `std::invalid_argument` with a message of the form
  `<op>: operand '<name>' <rule>`. `require_device_dtype` throws `UnsupportedExecution`.
- `Dispatch::require_domain` and `set_task_params` throw `std::invalid_argument`.
- `TaskRecorder::require_host_reachable`, which `record_pair` calls, throws
  `UnsupportedExecution`.
- `TaskRecorder::record_pair` throws `std::runtime_error` when the runtime's recorder refuses the
  task, with the message `<name>: the runtime refused the recording: <the runtime's message>`.
  The refusal stays in the recorder, so the next `execute()` returns it unless
  `discard_recorded()` is called first.
- `Storage` throws `std::runtime_error` when `Runtime::buffer` returns an error, with the message
  `Allocation failed for <n> bytes: <the runtime's message>`.
- `SB_THROW_IF` (`core/logger.hpp`) throws `std::runtime_error`. It ends every dtype switch, so a
  dtype that slipped past the record-time check cannot fall through to nothing.
- `IO` throws `std::invalid_argument` for an operand rule (a strided tensor, a dtype or byte-count
  mismatch). A file does not throw: the six file operations return an `IOStatus`. `IO::read`,
  `write` and `to_string` run the recorded graph first and have no status to return, so a failed
  run leaves them as `std::runtime_error` with the runtime's message.

Every helper logs at error level before it throws. A record call that throws leaves the tasks
recorded before it on the graph; `Engine::discard_recorded()` drops them.

## A run returns a `Result`

`Engine::execute()` returns `SushiRuntime::Result<void>` and is `[[nodiscard]]`: success, or a
`SushiRuntime::Error` with an `ErrorCode` and a message. A graph the compiler rejects, a task that
throws, a scheduler failure and a device error all arrive as that value. The graph is empty after
the call except after `ErrorCode::InvalidState` for a run in flight on the runtime, which started
no task and keeps what was recorded. A caller with no way to go on writes
`engine.execute().value();`, which stops the process on an error.

## A graph primitive returns a `Result`

The four primitives of `graph_primitives` record nodes on a `SushiRuntime::Recorder` and return
`SushiRuntime::Result<void>`. An argument that breaks a recording precondition returns
`ErrorCode::InvalidGraph` and records nothing, and a scratch block the recorder cannot allocate
returns `ErrorCode::AllocationFailed`. Otherwise the primitive returns what the recorder holds
once its nodes are recorded: success, or the first recording the recorder refused since its
`clear()`, which may be older than the call.

## File operations return a status

`IO::save`, `load`, `save_bin`, `load_bin`, `save_npy` and `load_npy` are `[[nodiscard]]` and
return an `IOStatus` (`io/io_status.hpp`). A missing or damaged file is a failure outside the
program, so the caller is handed a value to branch on. The reason is logged once, at error
level, where the failure is found.

| Status | Returned when |
| --- | --- |
| `Ok` | The file was written or loaded whole |
| `FileUnreadable` | The file could not be opened, measured or read; after a failed read the tensor holds a partial load |
| `FileUnwritable` | The file could not be opened, written or closed; the partial file is removed |
| `MalformedFile` | The magic, the format version or the header breaks the format |
| `Truncated` | The file ends before the bytes its header or the tensor calls for; the tensor is unchanged |
| `ShapeMismatch` | The file's rank or a dimension differs from the tensor's |
| `DtypeMismatch` | The file's element type differs from the tensor's |
| `LayoutMismatch` | The file's memory order differs from the tensor's layout |
| `RunFailed` | The tasks recorded on the engine failed in the run every file operation starts with; no file was opened and the tensor holds what the tasks left |

The tiled SYRK and SYR2K are refused at record time, with `UnsupportedExecution`, on a device that
cannot host their local tile.

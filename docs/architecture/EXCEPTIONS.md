# 11. Exceptions and statuses

A record call throws before any task reaches the graph, so the failure names the line that asked
for the operation. Four types cover the library:

| Type | Base | Thrown when |
| --- | --- | --- |
| `std::invalid_argument` | `std::logic_error` | An operand, flag, scalar or parameter is wrong: missing storage, a dtype, rank, shape, layout, stride or length mismatch, a dtype outside the operation's domain, a flag outside its enumeration, a complex scalar for a real tensor, a bad window geometry |
| `SushiBLAS::UnsupportedExecution` | `std::invalid_argument` | The operands are right and the operation cannot run under this mode or on this device: a device-only operand under `ForceReference`, FLOAT64 or COMPLEX64 storage on a device without `fp64`, HALF storage on a device without `fp16`, `poisson` on a device without `fp64` |
| `std::runtime_error` | `std::exception` | Creation: a `Tensor` or `Storage` rule (rank above 6, a negative dimension, an overflowing size, a failed allocation, a view past its storage), `TensorFactory::create`, `SparseBuilder` and `Sparse::assemble_csr`, a scratch allocation |
| `SushiRuntime::SushiException` | `std::runtime_error` | The runtime refused a graph or a run |

`UnsupportedExecution` (`core/errors.hpp`) derives from `std::invalid_argument`, so a handler
written for the base still catches it, and a caller that wants the two apart lists it first:

```cpp
try
{
    SushiBLAS::BLASOps(engine).gemm(A, B, C);
}
catch (const SushiBLAS::UnsupportedExecution& refusal)
{
    // The operands are right; this mode or this device cannot run the operation.
    std::printf("%s\n", refusal.what());
}
catch (const std::invalid_argument& rejected)
{
    // An operand, a flag or a scalar is wrong.
    std::printf("%s\n", rejected.what());
}
```

Where each type comes from:

- The operand checks of `engine/operand_check.hpp` (`require_storage`, `require_dtype`,
  `require_same_shape`, `require_same_layout`, `require_same_element_count`,
  `require_min_elements`, `require_contiguous`, `require_rank`, `require_scalar`,
  `require_extent`, `require_operand`) throw `std::invalid_argument` with a message of the form
  `<op>: operand '<name>' <rule>`. `require_device_dtype` throws `UnsupportedExecution`.
- `Dispatch::require_domain` and `set_task_params` throw `std::invalid_argument`.
- `TaskRecorder::require_host_reachable`, which `record_pair` calls, throws
  `UnsupportedExecution`.
- `SB_THROW_IF` (`core/logger.hpp`) throws `std::runtime_error`. It ends every dtype switch, so a
  dtype that slipped past the record-time check cannot fall through to nothing.
- `IO` throws `std::invalid_argument` for an operand rule (a strided tensor, a dtype or byte-count
  mismatch). A file does not throw: the six file operations return an `IOStatus`.

Every helper logs at error level before it throws.

## File operations return a status

`IO::save`, `load`, `save_bin`, `load_bin`, `save_npy` and `load_npy` are `[[nodiscard]]` and
return an `IOStatus` (`io_status.hpp`). A missing or damaged file is a failure outside the
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

`Engine::execute` can throw what the graph compiler throws (`std::runtime_error` for a cycle),
whatever the first failing task threw, a `SushiException` from the scheduler, and a
`sycl::exception` for a device error. The tiled SYRK and SYR2K are refused at record time, with
`UnsupportedExecution`, on a device that cannot host their local tile. An `IO` call that
reads or writes a tensor runs `execute()` first and can throw any of these. `SB_ASSERT` is not an
exception: it logs and aborts in a build without `NDEBUG`.

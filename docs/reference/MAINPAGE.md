# SushiBLAS {#mainpage}

A SYCL tensor and BLAS library built on [SushiRuntime](https://github.com/SushiSystems/SushiRuntime).
An operation call records a task on a graph and returns a `SushiBLAS::TaskHandle`; the runtime
orders the tasks by the byte ranges they declare, and `SushiBLAS::Engine::execute` runs the graph.
The order of every floating-point sum is fixed by the device and the extents: the floating-point
folds are hand-written and use no `sycl::reduction`.

This is the generated API reference. It documents what the headers say. The interface is below
1.0.0 and not stable: a MINOR release may change any name in it, and the Stability section of
`README.md` says how to depend on it. The reasoning behind the design is in the prose documents,
which are the better place to start:

- `README.md`: what the library is and how to build it.
- `docs/architecture/README.md`: the layers, the recorder, the execution modes, the seams
  and the kernels of each family.
- `docs/guides/INTEGRATION.md`: consuming SushiBLAS from your own CMake project.
- `cli/README.md`: every `sb` command.
- `docs/CONTRIBUTING.md`: how a change is built, tested and documented.

## Where to start reading

| If you want to | Read |
|---|---|
| Hold data | `SushiBLAS::Tensor`, `SushiBLAS::Storage`, `SushiBLAS::TensorFactory`: `tensor/tensor.hpp`, `tensor/storage.hpp`, `tensor/tensor_factory.hpp` |
| Own a graph and run it | `SushiBLAS::Engine`, built on a `SushiRuntime::Runtime`: `engine/engine.hpp`. `execute()` returns a `SushiRuntime::Result<void>` |
| Call an operation | The view of its family, built from an `Engine` or a `TaskRecorder`: `SushiBLAS::BLASOps` in `blas/blas.hpp`, and `SparseOps`, `ElementwiseOps`, `NonLinearOps`, `LogicOps`, `ReductionOps`, `RandomOps`, `SpatialOps`, each in `<family>/<family>.hpp` |
| Pass a BLAS flag or scalar | `SushiBLAS::Core::Transpose`, `Uplo`, `Diag`, `Side` in `core/blas_types.hpp`; `SushiBLAS::BLASScalar` in `operands/scalar.hpp` |
| Hold a sparse operand | `SushiBLAS::SparseMatrix`, `SparseVector`, `SparseBuilder`: `sparse_storage/sparse_matrix.hpp`, `sparse_storage/sparse_vector.hpp`, `sparse_storage/sparse_builder.hpp` |
| Move data to and from the host | `SushiBLAS::IO`: `io/io.hpp`; the file operations return `SushiBLAS::IOStatus` |
| Record on a graph of your own | `SushiBLAS::TaskRecorder`, `TaskDescription`, `OperandSet`, `TaskHandle`: `engine/task_recorder.hpp`, `engine/task_description.hpp`, `engine/operand_set.hpp`, `engine/task_handle.hpp` |
| Reduce, scan or sort on a SushiRuntime recorder | `SushiBLAS::GraphPrimitives::reduce`, `segmented_reduce`, `prefix_scan`, `counting_sort`: the headers of `graph_primitives/` |
| Choose the path a task runs | `SushiBLAS::ExecutionMode`: `engine/execution_mode.hpp` |
| Fuse several operations into one kernel | `SushiBLAS::Dispatch`: `engine/dispatch.hpp` and the `dispatch.hpp` of the elementwise, non-linear, logic, reduction and random families; `bind_scalars` and `bind_reads` in `functor_map/` |
| Attach an epilogue to a GEMM | `SushiBLAS::Dispatch::execute_gemm` and `gemm_epilogue`: `blas/dispatch.hpp` |
| Write a fold with a fixed order | `SushiBLAS::Reduce`: `reduction/reduce_1d.hpp`, `reduction/reduce_axis.hpp` |
| Tell a refusal from a wrong operand | `SushiBLAS::UnsupportedExecution`: `core/errors.hpp` |

No pass fuses recorded tasks automatically. The `Dispatch` seams are where a caller hands the
library one functor that does the work of several operations.

## Not in this reference

`modules/operations/experimental/include/SushiBLAS/experimental/` is excluded. Decision D1 quarantined the FFT and linear-algebra
headers: they are outside the umbrella header and have no implementation, so listing them here
would advertise an API that does not link.

## A caution about the reference

Doxygen lists every public symbol, which flattens a deliberate hierarchy. Three headers have
warnings that matter more than their signatures do, written in the module README each cites:

- `engine/operand_set.hpp` and `blas/dispatch.hpp`: why a task's operands, and an epilogue's,
  are declared as a `Tensor` and never as a `SushiRuntime::BufferRegion` built by hand. A region
  built from a bare pointer names the whole allocation behind that pointer, so two views of one
  arena become unrelated resources and no dependency edge is emitted between them.
- `functor_map/read_binding.hpp`: a tensor a functor reads at execution time is
  declared through `bind_reads`. A pointer captured inside a bare functor is not ordered, not kept
  alive and not checked under `ForceReference`.
- `reductions/reductions.hpp`: why a count or an index is an `INT64` and not a value in the
  input's own dtype.

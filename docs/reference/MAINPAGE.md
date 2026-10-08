# SushiBLAS {#mainpage}

A SYCL tensor and BLAS library built on [SushiRuntime](https://github.com/SushiSystems/SushiRuntime).
An operation call records a task on a graph and returns a `SushiBLAS::TaskHandle`; the runtime
orders the tasks by the byte ranges they declare, and `SushiBLAS::Engine::execute` runs the graph.
The order of every floating-point sum is fixed by the device and the extents: the floating-point
folds are hand-written and use no `sycl::reduction`.

This is the generated API reference. It documents what the headers say. The v0.1.0 interface it
describes has been written and not yet built or run as a whole. The reasoning behind the design
is in the prose documents, which are the better place to start:

- `README.md`: what the library is and how to build it.
- `docs/architecture/ARCHITECTURE.md`: the layers, the recorder, the execution modes, the seams
  and the kernels of each family.
- `docs/guides/INTEGRATION.md`: consuming SushiBLAS from your own CMake project.
- `cli/README.md`: every `sb` command.
- `docs/CONTRIBUTING.md`: how a change is built, tested and documented.

## Where to start reading

| If you want to | Read |
|---|---|
| Hold data | `SushiBLAS::Tensor`, `SushiBLAS::Storage`, `SushiBLAS::TensorFactory`: `tensor.hpp`, `storage.hpp`, `tensor_factory.hpp` |
| Own a graph and run it | `SushiBLAS::Engine`: `engine.hpp` |
| Call an operation | The view of its family, built from an `Engine` or a `TaskRecorder`: `SushiBLAS::BLASOps`, `SparseOps`, `ElementwiseOps`, `NonLinearOps`, `LogicOps`, `ReductionOps`, `RandomOps`, `SpatialOps` under `engine/` |
| Pass a BLAS flag or scalar | `SushiBLAS::Core::Transpose`, `Uplo`, `Diag`, `Side` in `core/blas_types.hpp`; `SushiBLAS::BlasScalar` in `engine/blas/scalar.hpp` |
| Hold a sparse operand | `SushiBLAS::SparseMatrix`, `SparseVector`, `SparseBuilder`: `sparse_matrix.hpp`, `sparse_vector.hpp`, `sparse_builder.hpp` |
| Move data to and from the host | `SushiBLAS::IO`: `io.hpp` |
| Record on a graph of your own | `SushiBLAS::TaskRecorder`, `OperandSet`, `TaskHandle`: `engine/task_recorder.hpp`, `engine/operand_set.hpp`, `engine/task_handle.hpp` |
| Choose the path a task runs | `SushiBLAS::ExecutionMode`: `engine/execution_mode.hpp` |
| Fuse several operations into one kernel | `SushiBLAS::Dispatch`: `engine/dispatch.hpp` and the `dispatch.hpp` of the elementwise, non-linear, logic, reduction and random families; `bind_scalars` and `bind_reads` in `engine/math/elementwise/` |
| Attach an epilogue to a GEMM | `SushiBLAS::Dispatch::execute_gemm` and `gemm_epilogue`: `engine/blas/dispatch.hpp` |
| Write a fold with a fixed order | `SushiBLAS::Reduce`: `support/reduce_1d.hpp`, `support/reduce_axis.hpp` |
| Tell a refusal from a wrong operand | `SushiBLAS::UnsupportedExecution`: `core/errors.hpp` |

No pass fuses recorded tasks automatically. The `Dispatch` seams are where a caller hands the
library one functor that does the work of several operations.

## Not in this reference

`include/SushiBLAS/experimental/` is excluded. Decision D1 quarantined the FFT and linear-algebra
headers: they are outside the umbrella header and have no implementation, so listing them here
would advertise an API that does not link.

## A caution about the reference

Doxygen lists every public symbol, which flattens a deliberate hierarchy. Three headers carry
warnings that matter more than their signatures do:

- `engine/operand_set.hpp` and `engine/blas/dispatch.hpp`: why a task's operands, and an
  epilogue's, are declared as a `Tensor` and never as a `ResourceRegion`. A region converts
  implicitly from a bare pointer, and a bare pointer names the whole allocation, so two views of
  one arena become unrelated resources and **no dependency edge is emitted**. That defect has
  cost this stack two silently wrong training runs.
- `engine/math/elementwise/read_binding.hpp`: a tensor a functor reads at execution time is
  declared through `bind_reads`. A pointer captured inside a bare functor is not ordered, not kept
  alive and not checked under `FORCE_REFERENCE`.
- `engine/math/reductions.hpp`: why a count or an index is an `INT64` and not a value in the
  input's own dtype.

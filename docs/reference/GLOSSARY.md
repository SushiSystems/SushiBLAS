# Glossary

The words this repository's documents use. Symbols are spelled as the code spells them.

| Word | Meaning |
| --- | --- |
| Engine | The composition root a caller holds. It owns one task graph with its scratch pool, task recorder, tensor factory and random stream, sets the execution mode and runs the graph; it names no operation family (`include/SushiBLAS/engine.hpp`) |
| View | A class that holds one operation family, such as `BLASOps`, `SparseOps` or `ReductionOps`. A caller builds it from an `Engine` or a `TaskRecorder` and calls the operations on it |
| Task recorder | `TaskRecorder`, the one caller of the graph's `add_task`. It records onto the task graph it was given, under the execution mode it holds, and returns a task handle (`engine/task_recorder.hpp`) |
| Task handle | `TaskHandle`, the value a record call returns: it identifies the task by its recorder and registration index, or names no task (`engine/task_handle.hpp`) |
| Operand set | `OperandSet`, the tensors one task reads and writes. It gives the graph their regions and keeps their storage allocated for as long as the recorded task lives (`engine/operand_set.hpp`) |
| Tensor factory | `TensorFactory`, which allocates the storage of a new tensor; `Engine::create_tensor` forwards to it (`tensor_factory.hpp`) |
| Scratch pool | `ScratchPool`, the engine's pool of small device blocks in six size classes up to 4 KiB; a `ScratchBuffer` leases one for a task (`engine/scratch_pool.hpp`) |
| Tensor | Shape, stride, rank, dtype and layout metadata over a `Storage`. `transpose()`, `reshape()` and `slice()` return new views over the same storage |
| Storage | A ref-counted SushiRuntime USM allocation, aligned to 128 bytes, that one or more tensors view |
| Task graph | SushiRuntime's `TaskGraph`. An operation call records one task onto it through the task recorder; `engine.execute()` compiles and runs the engine's own |
| Record time | The moment an operation call adds its task to the graph, before anything runs. Input checks throw here |
| Dtype | A member of `Core::DataType`: `Half`, `Float32`, `Float64`, `Complex32`, `Complex64`, `Int32`, `Int64`, numbered 0 to 6. The numbers are a file-format contract and a new type is appended |
| Domain | `Dispatch::Domain`, the dtype families an operation admits: `Real`, `Complex`, `Integral` and their unions |
| Layout | `Core::Layout`, `RowMajor` or `ColumnMajor`; how a matrix operand is addressed |
| Reference kernel | The plain C++ implementation of an operation, under `kernels/reference/`. The BLAS, sparse and spatial families have one per operation |
| Reference path | What runs on the host under `ForceReference`: the reference kernel where an operation has one, and for the elementwise, nonlinear, logic, reduction and random families a serial loop over the same functor, fold or Philox counters as the SYCL kernel |
| SYCL kernel | The device implementation of an operation, under `kernels/sycl/` |
| Execution mode | `ExecutionMode`: `Auto` and `ForceSycl` run the SYCL kernel, `ForceReference` runs the reference path on the host. A task keeps the mode its recorder held when it was recorded |
| Dispatch helper | `Internal::execute_typed_task` in `engine/task.hpp` records a task through the recorder and switches on the dtype; `Internal::KernelPairDispatcher` runs the reference or the SYCL kernel of a pair by the recorded mode |
| Dispatch seam | The public `SushiBLAS::Dispatch` functions that take a recorder and a caller-written functor and record it as one task, such as `execute_binary` and `execute_gemm` |
| Declared read | A tensor a caller's functor reads at execution time, named through `Dispatch::bind_reads` so the graph orders the task after its writers and keeps the storage (`engine/math/elementwise/read_binding.hpp`) |
| Epilogue | A functor GEMM applies to each output element just before the store; it receives the value, the row and the column |
| Launch shape | How a kernel maps work onto the device: per element (`launch_1d.hpp`), a whole-tensor fold (`reduce_1d.hpp`) or a segmented fold (`reduce_axis.hpp`) |
| Segment | The run of elements an axis reduction folds into one output element |
| Accumulation policy | A type such as `Reduce::CompensatedSum` or `Reduce::NaNPropagatingMax` that states how a fold combines values |
| Conjugation policy | `Kernels::Unconjugated` or `Kernels::Conjugated`, the template argument that makes one kernel serve a symmetric routine and its Hermitian sibling (`kernels/blas/conjugation.hpp`) |
| Operand access | `Kernels::OperandAccess`: the row step, the column step and the conjugation with which a kernel reads element (row, col) of op(X), resolved on the host from the layout and the `Transpose` flag |
| BLAS flag | `Transpose`, `Uplo`, `Diag` or `Side`, the enumerations a BLAS call takes in place of a bool; they carry the CBLAS numbers, so a value-initialised flag is invalid (`core/blas_types.hpp`) |
| `BlasScalar` | The type of alpha and beta in a BLAS call: a real or a complex value, built from `double` or `std::complex<double>` |
| Band storage | The CBLAS layout of a band matrix: only the `kl` sub-diagonals and `ku` super-diagonals, in a rank-2 tensor whose leading dimension is at least `kl + ku + 1`; `gbmv`, `sbmv`, `hbmv`, `tbmv` and `tbsv` read it |
| Packed storage | One triangle of a matrix of order n in a rank-1 tensor of `n * (n + 1) / 2` elements; `spmv`, `hpmv`, `tpmv`, `tpsv` and the `spr` and `hpr` updates read it. Not the packed panels below |
| Packed product | The CPU path of GEMM, SYRK and SYMM: it blocks over the three dimensions, copies the A and B panels into contiguous scratch and runs a fixed register-block micro-kernel |
| CSR | Compressed sparse row: row offsets, column indices and values, the three tensors a `SparseMatrix` holds |
| Windowed product | `window_dot`: for each output position, the sum over a window of input times weight, one group per channel over NHWC; a depthwise convolution |
| Device profile | `Device::Profile`, the facts about a device that a kernel dispatches on, probed once per device (`support/device_policy.hpp`) |
| Register block | The square of accumulators one GEMM work-item keeps in registers; 8x8 for 4-byte and narrower element types, 4x4 for wider ones |
| Tile | The block of an operand a GEMM work-group stages in local memory |
| AOT target | A device target compiled ahead of time through `SB_SYCL_TARGETS`, as opposed to SPIR-V compiled by the runtime |
| Quarantine | The headers under `include/SushiBLAS/experimental/`: declared, outside the umbrella header, and not all defined |
| Functional suite | The CTest tests labelled unit, integration or regression; the default of `sb test` |
| Package suite | `sb test --suite package`: installs the package and builds and runs `tests/package/` against it |
| Sibling checkout | A SushiRuntime checkout in the folder beside this repository, `../sushiruntime` |
| Workspace | A SushiStack folder that holds several module checkouts and one `dependencies/` tree |
| `hub` | SushiStack's CLI; it provisions the toolchains and libraries of a workspace |
| `sb` | This repository's CLI, also installed as `sushiblas`; see `cli/README.md` |
| Dependency fragment | `cli/sushistack.deps.toml`, where this module declares what `sb setup` and `hub install` provision for it |
| D1 | The decision that moved `LinalgOps` and `TransformsOps` out of the `Engine` facade into the quarantine. It is recorded in SushiAI's `docs/design/ROADMAP.md` |
| D3 | The three-step resolution of the SushiRuntime dependency in `cmake/Runtime.cmake`: a superproject target, then `find_package(SushiRuntime)`, then a sibling `add_subdirectory` |
| SB-0 to SB-12 | Numbered slices of the SushiBLAS build-out, used in commit subjects and in older documents. SB-6 is the packaging slice. The numbering comes from SushiAI's `docs/design/ROADMAP.md` |

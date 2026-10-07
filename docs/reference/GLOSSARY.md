# Glossary

The words this repository's documents use. Symbols are spelled as the code spells them.

| Word | Meaning |
| --- | --- |
| Engine | The facade a caller holds. It creates tensors, sets the execution mode, owns the task graph and hands out one view object per operation family (`include/SushiBLAS/engine.hpp`) |
| Tensor | Shape, stride, rank, dtype and layout metadata over a `Storage`. `transpose()`, `reshape()` and `slice()` return new views over the same storage |
| Storage | A ref-counted SushiRuntime USM allocation that one or more tensors view |
| Task graph | SushiRuntime's `TaskGraph`. An operation call adds one task to it; `engine.execute()` compiles and runs it |
| Record time | The moment an operation call adds its task to the graph, before anything runs. Input checks throw here |
| Dtype | A member of `Core::DataType`: `HALF`, `FLOAT32`, `FLOAT64`, `COMPLEX32`, `COMPLEX64`, `INT32`, `INT64` |
| Domain | `Dispatch::Domain`, the dtype families an operation admits: `REAL`, `COMPLEX`, `INTEGRAL` and their unions |
| Layout | `Core::Layout`, `ROW_MAJOR` or `COLUMN_MAJOR`; how a matrix operand is addressed |
| Reference kernel | The plain C++ implementation of an operation, under `kernels/reference/`. Correctness is judged against it |
| SYCL kernel | The device implementation of an operation, under `kernels/sycl/` |
| Execution mode | `ExecutionMode`: `AUTO` and `FORCE_SYCL` run the SYCL kernel, `FORCE_REFERENCE` runs the reference kernel on the host, `VERIFY` runs the SYCL kernel and then the reference kernel over the same output without comparing them |
| Dispatch helper | `Internal::execute_typed_task` and `Internal::dispatch_with_mode` in `engine/task.hpp`: the first records a task and switches on the dtype, the second applies the execution mode |
| Dispatch seam | The public `SushiBLAS::Dispatch` functions that take a caller-written functor and run it as one SYCL kernel, such as `execute_binary` and `execute_gemm` |
| Epilogue | A functor GEMM applies to each output element just before the store; it receives the value, the row and the column |
| Launch shape | How a kernel maps work onto the device: per element (`launch_1d.hpp`), a whole-tensor fold (`reduce_1d.hpp`) or a segmented fold (`reduce_axis.hpp`) |
| Segment | The run of elements an axis reduction folds into one output element |
| Accumulation policy | A type such as `Reduce::CompensatedSum` or `Reduce::RunningMax` that states how a fold combines values |
| Device profile | `Device::Profile`, the facts about a device that a kernel dispatches on, probed once per device (`support/device_policy.hpp`) |
| Register block | The square of accumulators one GEMM work-item keeps in registers; 8x8 for 4-byte and narrower element types, 4x4 for wider ones |
| Tile | The block of an operand a GEMM work-group stages in local memory |
| AOT target | A device target compiled ahead of time through `SB_SYCL_TARGETS`, as opposed to SPIR-V compiled by the runtime |
| Quarantine | The headers under `include/SushiBLAS/experimental/`: declared, outside the umbrella header and the `Engine` facade, and not all defined |
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

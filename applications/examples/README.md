# Examples

Eight programs that link the library the way a user would. `applications/examples/CMakeLists.txt`
builds them, one target each. Every program compares what it computed with a value computed on
the host, prints the outcome and returns 0, or 1 when the comparison fails.
The numbers are the same on every machine, with or without a GPU.

| Source | Target | Subject |
| --- | --- | --- |
| `adapter/` | `adapter_demo` | A host program that reaches the library through one file, in its own types and its own status |
| `errors_demo.cpp` | `errors_demo` | What a wrong record call throws, what `execute()` returns, and what a file operation returns |
| `run_failure_demo.cpp` | `run_failure_demo` | A run that fails: the error `execute()` returns, and the same engine running again |
| `recording_demo.cpp` | `recording_demo` | Several operations recorded, one `execute()`, tensors of two residencies, results read back |
| `operation_families_demo.cpp` | `operation_families_demo` | One checked call of each operation family |
| `graph_primitives_demo.cpp` | `graph_primitives_demo` | The four graph primitives on a SushiRuntime graph, without an engine |
| `quickstart_demo.cpp` | `quickstart_demo` | One 256 x 256 GEMM on shared-memory tensors with a column-major engine, the program the root README quotes |
| `io_demo.cpp` | `io_demo` | A tensor saved to a `.npy` file, loaded back and printed, with every `IOStatus` tested |

`sb run` starts `quickstart_demo` by default; see `cli/README.md`.

## `adapter/`: one file between a program and the library

Four files built into `adapter_demo`. Read this one first if you are about to depend on
SushiBLAS, because [the stability section](../../README.md#stability) asks every consumer for
this shape.

| File | Holds |
| --- | --- |
| `adapter/host_program.hpp` | What the host program knows: `HostProgram::Matrix`, `Status` with a `StatusCode`, a `Backend` it only declares, and the two operations it needs, `create_backend` and `multiply_add`. It includes the standard library and nothing else |
| `adapter/sushiblas_adapter.cpp` | The operations. It defines `Backend` as a `SushiRuntime::Runtime` and a `SushiBLAS::Engine`, and implements `create_backend` and `multiply_add` |
| `adapter/status_mapping.hpp` | The failures. `Detail::guarded` and `Detail::status_of_error` turn every exception and every runtime error into a `Status`; `sushiblas_adapter.cpp` alone includes it |
| `adapter/adapter_demo.cpp` | `main`. It includes `host_program.hpp` alone |

The two adapter files, `sushiblas_adapter.cpp` and `status_mapping.hpp`, are the only ones that
include `<SushiBLAS/SushiBLAS.h>`.

`multiply_add` computes `a * b + offset` and the sum of the result with `BLASOps::gemm`,
`ElementwiseOps::add` and `ReductionOps::sum`, recorded on one engine and run by one
`execute()`. Two functions of `status_mapping.hpp` turn a failure into a `Status`. `guarded` runs the
work inside one `try` and catches `UnsupportedExecution`, `std::invalid_argument`,
`std::bad_alloc` and `std::exception`, most derived first. `status_of_error` is a `switch` over
the ten values of `SushiRuntime::ErrorCode` with no `default`, so a compiler that warns on an
unhandled enumerator names the code a runtime release adds. After a failure `multiply_add`
calls `Engine::discard_recorded()`, because a record call that throws leaves the tasks
recorded before it on the engine.

`main` multiplies a 2 x 3 matrix by a 3 x 2 one, adds 0.5 and compares the four elements and
their sum with three loops on the host. It then asks for `a * a`, which has no meaning for a
2 x 3 matrix, and prints the status the adapter made of the library's exception. The program
prints `a * b + offset = [58.5, 64.5; 139.5, 154.5], sum = 417`, then `a * a: InvalidRequest: `
followed by the library's sentence, then `adapter_demo: ok`.

When a release of SushiBLAS or SushiRuntime breaks the interface, the two adapter files are
the ones that change. `host_program.hpp` and `adapter_demo.cpp` name no library type, so
they compile as they are.

## `errors_demo.cpp`: the failure channels

The program provokes one failure on each channel a caller of an engine meets, and
[the architecture, section 11](../../docs/architecture/EXCEPTIONS.md) lists what arrives on
each.

| Step | What the program does | What comes back |
| --- | --- | --- |
| A wrong operand | `axpy` on a vector of 4 elements and one of 3 | `std::invalid_argument` |
| A recorded task dropped | `Engine::discard_recorded()` after that throw, then a read of `y` | `y` is unchanged: the `axpy` recorded before the throw did not run |
| A mode that cannot run the call | `axpy` on a `Residency::Device` tensor under `ExecutionMode::ForceReference` | `SushiBLAS::UnsupportedExecution` |
| A tensor that cannot be made | `create_tensor({-1})` | `std::runtime_error` |
| A run | `execute()`, with its `SushiRuntime::Result<void>` tested and the `Error` printed on failure | Success, and `y` equals `2 * x` |
| A file | `IO::load_npy` of a file that does not exist | `IOStatus::FileUnreadable` |

`outcome_of` holds the three `catch` clauses, `UnsupportedExecution` first because it derives
from `std::invalid_argument`. The program prints one line per step with the type and the
library's message, and `errors_demo: ok` when every step gave the outcome in the table. The
library also logs each refusal at error level before it throws, so its log lines appear
between the program's own.

The run in this program succeeds, so the branch that prints `ran.error().code` and
`ran.error().message` is not taken here. `run_failure_demo.cpp` takes it.

## `run_failure_demo.cpp`: a run that fails

No operation of the views fails a run on every machine, so the program records a task of
its own whose body throws. `TaskRecorder::record_pair`, reached through
`Engine::recorder()`, takes a `TaskDescription`, an `OperandSet` and a body; every view
method records through it. The body is host code that the scheduler calls to submit the
task's kernels, so it runs on the host under every `ExecutionMode` and on a machine with or
without a GPU, and the program sets no mode. Recording succeeds, because the body has not
run yet.

`execute()` then returns an error and throws nothing. The program tests the `Result`, prints
`ErrorCode::KernelFailed` and the message, which names the task and ends with the sentence
the body threw, and goes on with the same engine: a second `execute()` succeeds because the
failed run emptied the graph, and an `axpy` recorded after it gives `y = 2 * x`. The output
is `execute: KernelFailed: node 'run_failure_demo.failing' (registration index 0) failed:
the consumer's kernel refused to run`, then `execute after the failure: ok, y = 2 * x` and
`run_failure_demo: ok`. The registration index in the message is the runtime's and may
differ.

The body's first parameter is the library's kernel-pair dispatcher, a type of its `Internal`
namespace. The lambda takes it as `const auto&` and does not use it.
`modules/engine/engine/tests/test_run_failure.cpp` pins the same behaviour.

## `recording_demo.cpp`: record, then run

An operation of a view records a task and runs nothing. The program shows that with a tensor
the host can read directly: it fills `result` with -1, records `scal`, `add` and `sum`, and
reads `result[0]` through `Tensor::data_as` before and after `Engine::execute()`. The first
read is still -1 and the second is `2 * a[0] + b[0]`, which is 12.

`a` and `result` are `Residency::Shared`, the default, and `b` is `Residency::Device`. The
host cannot dereference device memory, so `b` is filled and read through `IO::from_vector`
and `IO::to_vector`, which work for every residency. One `add` reads both.

The program also prints the bytes behind `result`: 6 FLOAT32 elements need 24 bytes and
`Storage::size_bytes()` reports 128, because every allocation is a multiple of
`K_STORAGE_ALIGNMENT`. A second `execute()` with nothing recorded returns success. The last
lines are `total = 252 (expected 252)` and `recording_demo: ok`.

## `operation_families_demo.cpp`: one call of each family

Every family is a view built from the engine. The program records one operation of each,
runs them with one `execute()` and compares each result with numbers written in the source.

| View | Call | Expected |
| --- | --- | --- |
| `BLASOps` | `dot` of (1, 2, 3, 4) and (4, 3, 2, 1) | 20 |
| `BLASOps` | `gemv` of [1 2; 3 4] and (1, 1) | (3, 7) |
| `BLASOps` | `gemm` of [1 2; 3 4] with itself | [7 10; 15 22] |
| `ElementwiseOps` | `mul` of the two vectors | (4, 6, 6, 4) |
| `NonLinearOps` | `relu` of (-1, 2, -3, 4), in place | (0, 2, 0, 4) |
| `LogicOps` | `greater` of the two vectors | (0, 0, 1, 1) |
| `ReductionOps` | `max` of (1, 2, 3, 4) | 4 |
| `SparseOps` | `spmv` of [1 0 2; 0 3 0], built by `SparseBuilder::matrix_from_triplets`, and (1, 2, 3) | (7, 6) |
| `SpatialOps` | `upsample_nearest` of a 2 x 2 image by 2 in each direction | Each pixel repeated over a 2 x 2 block |
| `RandomOps` | `set_seed(42)` and `uniform`, twice, on 64 elements each | The two draws are equal and every value is in [0, 1) |

The inputs are small integers, so every expected value is exact in FLOAT32 and the
comparison is `==`. The random draw is checked for equality between two draws of one seed
and for its range; the program prints no drawn number. A spatial tensor is rank 4 with the
image index first and the channel last. The program prints one line per row, `ok` or
`MISMATCH`, then `operation_families_demo: ok`.

## `graph_primitives_demo.cpp`: the primitives on a runtime graph

For a caller who uses SushiRuntime directly. The four primitives of
[`graph_primitives`](../../modules/foundation/graph_primitives/README.md) take a
`SushiRuntime::Recorder&` and `SushiRuntime::Buffer<T>` operands; no `Engine` and no `Tensor`
is involved. The umbrella header does not include them, so this program includes
`<SushiRuntime/sushiruntime.hpp>` and the headers under `<SushiBLAS/graph_primitives/>`.

The program records all four on one `Graph` and runs it once:

| Primitive | Input | Expected |
| --- | --- | --- |
| `reduce` with `Sum` | 1, 2, ..., 1000 | 500500 |
| `segmented_reduce` with `Sum` | The same values with offsets 0, 10, 10, 1000 | 55, 0 and 500445; the empty segment gives the identity |
| `prefix_scan` | Counts 3, 0, 2, 5 | Offsets 0, 3, 3, 5, 10 and a total of 10 |
| `counting_sort` | Keys 2, 0, 2, 1, 0, 2 over a key range of 3 | Order 1, 4, 3, 0, 2, 5 and bucket starts 0, 2, 3, 6 |

Each primitive returns a `SushiRuntime::Result<void>`, and the program stops on an error
with `value()`. Before the run it shows the other branch: `segmented_reduce` asked for five
segments over four offsets returns `ErrorCode::InvalidGraph` and records nothing, and the
program prints `refused: InvalidGraph: ` followed by the message. It then prints one line per
check and `graph_primitives_demo: ok`. A count is a `std::size_t` or a
`SushiRuntime::LateCount`; `LateCount::fixed` gives one that is known when the call is
recorded.

## `io_demo.cpp`: save, load and print

The program writes a 2 x 3 tensor with `IO::save_npy`, reads it into a second tensor with
`IO::load_npy` and compares the six elements with the host's. Both calls return an `IOStatus`,
and the program tests each one and returns 1 with the status name when it is not `Ok`. It then
loads the same file into a 3 x 2 tensor, which the library refuses with
`IOStatus::ShapeMismatch`.

`IO::to_string` and `IO::print` render a tensor as text, a real one at two decimals and a
COMPLEX32 one at the default precision. The text is printed and not compared; the complex
tensor is checked through `IO::to_vector`.

The file is `sushiblas_io_demo.npy` in `std::filesystem::temp_directory_path()`. A
`TemporaryFile` in the source removes it when `main` returns, on the paths that return 1 too.
The last lines are `load_npy into a 3 x 2 tensor: ShapeMismatch`, the complex tensor and
`io_demo: ok`; the refused load also leaves one error line of the library's log.

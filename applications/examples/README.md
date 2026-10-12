# Examples

Three programs that link the library the way a user would. `applications/examples/CMakeLists.txt`
builds them, one target each. Every program but `io_example` compares what it computed with a
value computed on the host, prints the outcome and returns 0, or 1 when the comparison fails.
The numbers are the same on every machine, with or without a GPU.

| Source | Target | Subject |
| --- | --- | --- |
| `adapter/` | `adapter_demo` | A host program that reaches the library through one file, in its own types and its own status |
| `example.cpp` | `sushiblas_example` | One 256 x 256 GEMM on shared-memory tensors with a column-major engine, the program the root README quotes |
| `io_example.cpp` | `io_example` | `to_string`, `save_npy` and `load_npy` on a 2 x 3 tensor, then a complex tensor printed |

`sb run` starts `sushiblas_example` by default; see `cli/README.md`.

## `adapter/`: one file between a program and the library

Three files built into `adapter_demo`. Read this one first if you are about to depend on
SushiBLAS, because [the stability section](../../README.md#stability) asks every consumer for
this shape.

| File | Holds |
| --- | --- |
| `adapter/host_program.hpp` | What the host program knows: `HostProgram::Matrix`, `Status` with a `StatusCode`, a `Backend` it only declares, and the two operations it needs, `create_backend` and `multiply_add`. It includes the standard library and nothing else |
| `adapter/sushiblas_adapter.cpp` | The only file that includes `<SushiBLAS/SushiBLAS.h>`. It defines `Backend` as a `SushiRuntime::Runtime` and a `SushiBLAS::Engine`, and implements the two operations |
| `adapter/adapter_demo.cpp` | `main`. It includes `host_program.hpp` alone |

`multiply_add` computes `a * b + offset` and the sum of the result with `BLASOps::gemm`,
`ElementwiseOps::add` and `ReductionOps::sum`, recorded on one engine and run by one
`execute()`. Two functions of the adapter turn a failure into a `Status`. `guarded` runs the
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

When a release of SushiBLAS or SushiRuntime breaks the interface, `sushiblas_adapter.cpp` is
the file that changes. `host_program.hpp` and `adapter_demo.cpp` name no library type, so
they compile as they are.

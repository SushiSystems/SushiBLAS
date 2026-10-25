# Device compilation only where a unit instantiates a kernel: plan

**Status:** Open — written 2026-10-26; S1 to S3 run inline in one session, S4 waits on the owner's build.

## How it runs

One session runs the tasks in order, inline, without workers, and commits each with its pages and
its changelog line. The owner ruled on 2026-10-26 that this work may edit `CMakeLists.txt` and
`cmake/*.cmake` and may not run CMake, Ninja, CTest or `sb`. Each task is checked by reading and
by single-file `clang++ -fsyntax-only` runs with the flags of `build/compile_commands.json`, one at
a time. The design and the measurements are in `SPEC.md`.

## Tasks

| Id | Task | Waits on | Acceptance | State |
| --- | --- | --- | --- | --- |
| S1 | List every unit that instantiates a kernel under a `DEVICE_` keyword: add `DEVICE_SOURCES` to `sushiblas_add_module()` and `sushiblas_add_application_test_program()`, `DEVICE_TESTS` and `DEVICE_BENCHMARKS` to `sushiblas_add_test_program()`, and move the 406 files `SPEC.md` section 2 measured. Every unit still compiles for the device | nothing | The owner's configure gives the same compile command for every unit as before S1 | written |
| S2 | The policy in `cmake/SyclTarget.cmake`: `sushiblas_sycl_headers`, `sushiblas_device_code`, `sushiblas_apply_sycl(DEVICE_SOURCES)`, `sushiblas_publish_sycl()`; `cmake/SushiBLASSyclHeaders.cmake`; the library publishes link options only; the export, the Config file and the package test carry the same shape; the examples name their device units; the manual pages that describe the old policy | S1 | The owner's build compiles exactly the units under a `DEVICE_` keyword with `-fsycl`, and `sb test` and `sb test --package` pass | planned |
| S3 | Drop `<sycl/sycl.hpp>` from the five public headers that name nothing from SYCL | nothing | Each header and one includer of it pass `clang++ -fsyntax-only` with and without the device options | planned |
| S4 | `REPORT.md`: what landed, the owner's build numbers, the changes SushiAI and SushiEngine need; close the folder | S2, S3, the owner's build | The report carries the before and after build times and the count of `-fsycl` units | planned |

## Proposals for the owner

Each changes a public header or signature, so none is carried out here.

| Proposal | Cost today | Gain |
| --- | --- | --- |
| `Tensor::device()` returns `SushiRuntime::DeviceIndex`, or moves out of `tensor.hpp` | Breaks every caller of `device()` | Nothing alone: `storage.hpp` still brings SYCL; pairs with the next row |
| `tensor/storage.hpp`, `tensor/tensor.hpp` and `engine/task_recorder.hpp` include the SushiRuntime headers they use instead of `<SushiRuntime/sushiruntime.hpp>`, and `Storage::queue()` leaves the public header | A consumer that relied on the umbrella arriving through them adds its own include | The runtime's graph and fluent API leave the parse of every includer; SYCL still arrives through SushiRuntime's own headers |
| SushiRuntime keeps `<sycl/sycl.hpp>` out of the headers a host unit needs (`Runtime`, `Buffer`, `Result`) | A SushiRuntime change | The only change that takes the SYCL parse out of a host unit |
| The task bodies of `engine/task_recorder.hpp` take a SushiRuntime queue handle instead of `sycl::queue&` and `sycl::event` | Breaks every task body in the stack | SYCL leaves the recorder's interface |

## SOLID breaches and unneeded dependencies met

| Where | What | Cost | State |
| --- | --- | --- | --- |
| `cmake/SyclTarget.cmake`, `sushiblas_apply_sycl()` | Device options published as `PUBLIC` compile options of a library | Two device passes on every unit of every consumer | Removed by S2 |
| `blas/sparse.hpp`, `elementwise/dispatch.hpp`, `io/io.hpp`, `logic/dispatch.hpp`, `nonlinear/dispatch.hpp` | Include `<sycl/sycl.hpp>` and name nothing from it | A dependency the header does not use | Removed by S3 |
| `tensor/storage.hpp`, `tensor/tensor.hpp`, `engine/task_recorder.hpp` | Include the SushiRuntime umbrella for a few types | Every includer parses the runtime's whole API | Kept; proposal |
| 52 test units under `tests/` | Include `<sycl/sycl.hpp>` and mention SYCL on no other line | A dependency the unit does not use | Seen; not in this work's scope |
| `applications/examples/CMakeLists.txt` | Eight copies of one `POST_BUILD` copy command | Each new example copies it again | Seen |
| `sushiengine/cmake/SyclTarget.cmake`, `sushiai/cmake/SushiAISyclTarget.cmake` | Each derives the toolchain's include directory and reads the device targets itself | Three copies of one lookup in the stack | Seen; listed for those repositories in `REPORT.md` |

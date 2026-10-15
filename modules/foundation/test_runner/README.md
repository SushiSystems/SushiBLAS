# test_runner

The GoogleTest environment that owns the one `SushiRuntime::Runtime` of a test program, and the `main` of each test program size. The module is internal and test-only: its headers are not installed, and it is added
only when `SB_BUILD_TESTS` is on.

It depends on `core`, SushiRuntime and GoogleTest. Its headers are included as `<SushiBLAS/test_runner/test_environment.hpp>`.

| File | Declares |
| --- | --- |
| `test_environment.hpp` | `TestEnvironment`, the runtime the tests share |

`source/test_main.cpp` holds the `main` of a program with a `Medium_` or `Large_` suite:
synchronous logging, the environment, then `RUN_ALL_TESTS`. `source/small_test_main.cpp` holds
the `main` of a program whose suites are all `Small_`: synchronous logging and `RUN_ALL_TESTS`,
with no environment, so the program opens no device and starts no worker thread. Neither is a
source of the module. `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake` compiles the
one that fits the program's suites into it, so neither reaches the library.

## test_environment.hpp

`TestEnvironment` creates the runtime in `SetUp()` with `SushiRuntime::Runtime::create()` and
hands it out through `runtime()`. `queue(device_index)` and `device_count()` forward to the
runtime's `advanced()` view.

`live_allocation_count()` returns how many USM allocations of the runtime are live; the lifetime
tests count with it. It forwards to `AdvancedControls::live_allocation_count()`.

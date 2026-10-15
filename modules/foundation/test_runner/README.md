# test_runner

The GoogleTest environment that owns the one `SushiRuntime::Runtime` of a test program, and the `main` every test program runs. The module is internal and test-only: its headers are not installed, and it is added
only when `SB_BUILD_TESTS` is on.

It depends on `core`, SushiRuntime and GoogleTest. Its headers are included as `<SushiBLAS/test_runner/test_environment.hpp>`.

| File | Declares |
| --- | --- |
| `test_environment.hpp` | `TestEnvironment`, the runtime the tests share |

`source/test_main.cpp` holds `main`: synchronous logging, the environment, then `RUN_ALL_TESTS`.
It is not a source of the module. `sushiblas_add_test_program()` in
`cmake/SushiBLASTests.cmake` compiles it into each test program, so it never reaches the library.

## test_environment.hpp

`TestEnvironment` creates the runtime in `SetUp()` with `SushiRuntime::Runtime::create()` and
hands it out through `runtime()`. `queue(device_index)` and `device_count()` forward to the
runtime's `advanced()` view.

`live_allocation_count()` returns how many USM allocations of the runtime are live; the lifetime
tests count with it. It forwards to `AdvancedControls::live_allocation_count()`.

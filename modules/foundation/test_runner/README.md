# test_runner

The GoogleTest environment that owns the one `SushiRuntime::Runtime` of a test program, and the `main` of each test program size. The module is internal and test-only: its headers are not installed, and it is added
only when `SB_BUILD_TESTS` is on.

It depends on `core`, SushiRuntime and GoogleTest. Its headers are included as `<SushiBLAS/test_runner/test_environment.hpp>`.

| File | Declares |
| --- | --- |
| `test_environment.hpp` | `TestEnvironment`, the runtime the tests share |
| `run_observer_capture.hpp` | `RunObserverCapture`, which keeps the dispatch records of the runs of a runtime |
| `log_capture.hpp` | `LogCapture`, which keeps the records SushiBLAS writes while a test runs |

`source/test_main.cpp` holds the `main` of each tier program and of an application program with a
`Medium_` or `Large_` suite: synchronous logging, the environment, then `RUN_ALL_TESTS`. It leaves
the environment out when `--gtest_filter` selects only `Small_` suites, so a `Small_` test opens no
device and starts no worker thread. `source/small_test_main.cpp` holds the `main` of an
application program whose suites are all `Small_`: synchronous logging and `RUN_ALL_TESTS`, with no
environment. Neither is a source of the module. `cmake/SushiBLASTests.cmake` compiles the one
that fits into each program, so neither reaches the library.

## test_environment.hpp

`TestEnvironment` creates the runtime in `SetUp()` with `SushiRuntime::Runtime::create()` and
hands it out through `runtime()`. `queue(device_index)` and `device_count()` forward to the
runtime's `advanced()` view.

`live_allocation_count()` returns how many USM allocations of the runtime are live; the lifetime
tests count with it. It forwards to `AdvancedControls::live_allocation_count()`.

`RunObserverCapture` adds a `RunObserver` to a runtime and turns device timing on. A test that asserts the widths a dispatch launched over reads `records()` after the run; `clear()` forgets the runs so far. `LogCapture` switches SushiLog to synchronous delivery, keeps every record at every level and restores stderr and the default level when it ends. It stands in for the shipped capture sink SushiLog's backlog describes.

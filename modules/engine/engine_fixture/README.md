# engine_fixture

`SushiBLASTest`, the GoogleTest fixture that gives each test a fresh `Engine` on the shared
runtime. The module is internal and test-only: its headers are not
installed, and it is added only when `SB_BUILD_TESTS` is on.

It depends on `core`, `tensor`, `engine`, `test_runner`, `tensor_expectations` and GoogleTest. Its headers are included as `<SushiBLAS/engine_fixture/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `sushiblas_test.hpp` | `SushiBLASTest` and its `create_initialized_tensor` |

## sushiblas_test.hpp

`SetUp` builds a row-major `Engine` on `TestEnvironment::runtime()` and restarts the
generator of `fill_random`, so a test sees the same inputs whether it runs alone or with its
suite. `reinit_engine` rebuilds the engine in another layout. `create_initialized_tensor<T>`
creates a tensor of `T`'s dtype and copies a host vector into it.

The fixture includes no operation family. A test includes the headers of the families it
records, or `<SushiBLAS/SushiBLAS.hpp>`.

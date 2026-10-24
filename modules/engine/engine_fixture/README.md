# engine_fixture

`SushiBLASTest`, the GoogleTest fixture that gives each test a fresh `Engine` on the shared
runtime, and `RecordingFixture`, the `RecordingGraph` a test records on with its GoogleTest helper. The module is internal and test-only: its headers are not
installed, and it is added only when `SB_BUILD_TESTS` is on.

It depends on `core`, `tensor`, `engine`, `recording_graph`, `test_runner`, `tensor_expectations` and GoogleTest. Its headers are included as `<SushiBLAS/engine_fixture/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `sushiblas_test.hpp` | `SushiBLASTest` and its `create_initialized_tensor` |
| `recording_fixture.hpp` | `RecordingFixture` |

## sushiblas_test.hpp

`SetUp` builds a row-major `Engine` on `TestEnvironment::runtime()`, a `RecordingFixture` over
it, reachable as `recording`, and restarts the generator of `fill_random`, so a test sees the
same inputs whether it runs alone or with its suite. A test builds its operation views on
`recorder()` and runs them with `recording->run()`. `set_mode` sets the execution mode of the
engine and of that recorder. `reinit_engine` rebuilds the engine in another layout, and the
graph with it. `create_initialized_tensor<T>`
creates a tensor of `T`'s dtype and copies a host vector into it.

## recording_fixture.hpp

`RecordingFixture` holds a `RecordingGraph` (`modules/engine/recording_graph/README.md`), which
owns the graph, the `TaskRecorder`, `run()`, `discard()` and `plan()`, and forwards its
`recorder()`, `run()`, `plan()`, `discard()`, `recorded_count()` and `is_empty()`. It adds
`run_succeeds()`, `run()` as a GoogleTest assertion that prints the runtime's message on failure.
It holds the graph rather than deriving from it, so `RecordingGraph` keeps its non-virtual
destructor and a fixture exposes only the calls a test makes. The constructors mirror those of
`RecordingGraph`.

The fixture includes no operation family. A test includes the headers of the families it
records, or `<SushiBLAS/SushiBLAS.hpp>`.

# engine_fixture

`SushiBLASTest`, the GoogleTest fixture that gives each test a fresh `Engine` on the shared
runtime, and `RecordingFixture`, the graph and recorder a test records on. The module is internal and test-only: its headers are not
installed, and it is added only when `SB_BUILD_TESTS` is on.

It depends on `core`, `tensor`, `engine`, `test_runner`, `tensor_expectations` and GoogleTest. Its headers are included as `<SushiBLAS/engine_fixture/<name>.hpp>`.

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

`Engine` keeps no graph for a test to run. `RecordingFixture` is the four lines a caller that
owns a graph writes: it holds a `SushiRuntime::Graph` and a `TaskRecorder` over that graph's
recorder. The constructor that takes an `Engine&` borrows the engine's scratch pool and
execution mode; the one that takes a runtime takes a pool and a mode of its own.
`recorder()` is what a test builds its operation families on.

`run()` runs the graph, returns when every task has finished and empties it, so the fixture
records and runs again; it returns the run's `SushiRuntime::Result<void>`, also when nothing was
recorded. `run_succeeds()` is `run()` as a GoogleTest assertion that prints the runtime's
message on failure. `discard()` drops the recorded tasks and returns how many, `plan()` returns
what the plan orders and `recorded_count()` and `is_empty()` report what waits.

The fixture includes no operation family. A test includes the headers of the families it
records, or `<SushiBLAS/SushiBLAS.hpp>`.

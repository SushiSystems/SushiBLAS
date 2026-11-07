# engine_fixture

`SushiBLASTest`, the GoogleTest fixture that gives each test a fresh `Engine` on the shared
runtime, and `RecordingFixture`, the `RecordingGraph` a test records on with its GoogleTest helper.
`HostEngineTest` and `HostRecordingFixture` are their host forms: a `Small_` test of an operation uses
them and makes no queue. The module is internal and test-only: its headers are not
installed, and it is added only when `SB_BUILD_TESTS` is on.

It depends on `core`, `tensor`, `engine`, `recording_graph`, `test_runner`, `tensor_expectations` and GoogleTest. Its headers are included as `<SushiBLAS/engine_fixture/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `sushiblas_test.hpp` | `SushiBLASTest` and its `create_initialized_tensor` |
| `recording_fixture.hpp` | `RecordingFixture` |
| `host_engine_test.hpp` | `HostEngineTest` and its `create_initialized_tensor` |
| `host_recording_fixture.hpp` | `HostRecordingFixture` |

## sushiblas_test.hpp

`SetUp` builds a row-major `Engine` on `TestEnvironment::runtime()`, a `RecordingFixture` over
it, reachable as `recording`, and restarts the generator of `fill_random`, so a test sees the
same inputs whether it runs alone or with its suite. A test builds its operation views on
`recorder()` and runs them with `recording->run()`. `set_mode` sets the execution mode of the
engine and of that recorder. `reinit_engine` rebuilds the engine in another layout, and the
graph with it. `create_initialized_tensor<T>`
creates a tensor of `T`'s dtype and copies a host vector into it.

## host_engine_test.hpp

`HostEngineTest` is the fixture of a `Small_` test of an operation. `SetUp` builds a row-major host
`Engine` (`Engine::Host{}`), a `HostRecordingFixture` over it, reachable as `recording`, and restarts the
generator of `fill_random`. `recorder()`, `reinit_engine` and `create_initialized_tensor<T>` work as in
`SushiBLASTest`, and `create_initialized_tensor` copies by `memcpy`. There is no `set_mode`: a host
engine runs `ForceReference` only. The fixture makes no queue and reaches no `TestEnvironment`;
`fill_tensor`, `fill_random`, `read_tensor`, `expect_tensor` and `expect_tensor_equal` read and write a
host tensor by `memcpy`.

A host engine accepts a recorded body of the shape `(const KernelPairDispatcher&, const
std::vector<sycl::event>&)`, the one shape every operation body has, so every operation runs on it.

## host_recording_fixture.hpp

`HostRecordingFixture` holds a `TaskRecorder` over the `HostTaskSink` of a host engine, with the engine's
scratch pool and mode. It forwards `recorder()`, `recorded_count()`, `is_empty()` and `discard()`. `run()`
runs the bodies in record order and returns the first `HostTaskFailure`, or nothing; `run_succeeds()` is
`run()` as a GoogleTest assertion that names the failed task. It builds no graph, so it has no `plan()`.

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

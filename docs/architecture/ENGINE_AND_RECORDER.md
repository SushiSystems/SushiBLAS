# 5. `Engine`, the recorder and the views

`Engine` (`engine/engine.hpp`) is the composition root a program constructs once on a
`SushiRuntime::Runtime`, with the layout of the tensors it creates. It owns, in this order, the
`ScratchPool`, the `RecordedGraph`, the `TaskRecorder` bound to both, the `TensorFactory` and the
`RNGStream`. Its interface is short:

- `create_tensor(dims, dtype, residency, device_index)`, from a brace list or a span.
- `set_mode` and `mode`, which forward to the recorder, and `layout`.
- `recorder()`, `tensor_factory()`, `rng()`, `scratch_pool()` and `runtime()`.
- `execute()`, which runs the recorded graph on the runtime and returns when every task has
  finished. It returns a `SushiRuntime::Result<void>`: success, also when nothing is recorded, or
  the `SushiRuntime::Error` that ended the run. No exception leaves it. The graph is empty when
  it returns, except after `ErrorCode::InvalidState` for another run in flight on the runtime,
  which keeps what was recorded. No thread may be recording on the engine while it runs.
- `discard_recorded()`, which drops every recorded task that has not run, with the operand
  storage and the scratch those tasks hold.

No accessor reaches the graph. `RecordedGraph` (`engine/recorded_graph.hpp`) owns the
`SushiRuntime::Graph` and has the two verbs that change it, `run()` and `discard()`; `execute()`
and `discard_recorded()` call them. After a run or a discard it replaces the graph with an empty
one from the runtime, which is what destroys the recorded tasks and the plan. The engine is not copyable or movable, and the `Runtime`
object must outlive it.

## `TaskRecorder`, `OperandSet` and `TaskHandle`

`TaskRecorder` (`engine/task_recorder.hpp`) records tasks on one `SushiRuntime::Recorder` under
one mode, and is the only caller of `SushiRuntime::Recorder::task` in SushiBLAS. It has one
recording call:

`record_pair(description, operands, family, body)` checks host reachability and adds one task
whose reads and writes are the operand set's regions, and whose name, operation and parameters
are those of the `TaskDescription`, handed to the runtime as `SushiRuntime::NodeOptions`. A
recording the runtime refuses throws `std::runtime_error` with the runtime's message. The body
is a copyable callable `(const KernelPairDispatcher&, sycl::queue&, const
std::vector<sycl::event>&) -> sycl::event` that must order its work behind the events it is
given. On the SYCL side that means every command group calls `h.depends_on(deps)`;
`check_engine_seams.py` reports a command group without it.

One mutex in the recorder serialises record calls, so several threads may record on one engine
at once; the tasks reach the graph in the order the threads took the lock. `execute()` and
`set_mode` are outside that lock and must not overlap a record call. `run_exclusive(work)` runs a
callable under the same mutex, and `Engine::discard_recorded()` discards through it.

`OperandSet` (`engine/operand_set.hpp`) is how a task says what it touches, and it takes tensors,
never raw regions:

| Call | Declares |
| --- | --- |
| `reads(t)` | `t.region()` as a read |
| `writes(t)` | `t.region()` as a write |
| `updates(t)` | Both, for an operand changed in place |
| `scratch(buffer)` | A leased scratch block as a write |
| `merge(other)` | Every declaration of another set, such as an epilogue's |

`reads`, `writes` and `updates` also retain the tensor's storage and note a device-only residency.
`scratch` retains the block's storage, and `merge` carries over what the other set retained and
noted. A tensor without storage adds nothing. Taking the region from `Tensor::region()` in one
place is deliberate: a `SushiRuntime::BufferRegion` built from a bare pointer is keyed on that
pointer and names a whole allocation, so two views of one arena declared by their data pointers
look unrelated to the tracker, which emits no edge between them.

`TaskHandle` (`engine/task_handle.hpp`) is what every record call returns: the identity of the
recorder and the sequence number the recorder gave the task. A call that records nothing, such as an
elementwise operation on an empty tensor, returns the default handle, for which `is_valid()` is
false. The handle is a name for the task and gives no way to wait on it; `execute()` is the only
way to run the graph.

## Views

A view is a small class of record calls bound to a recorder. Each has two constructors, one
from an `Engine` and one from a `TaskRecorder` the caller owns:

```cpp
ScratchPool pool(TestEnvironment::runtime());
SushiRuntime::Graph first_graph = TestEnvironment::runtime().graph();
TaskRecorder first_recorder(first_graph.recorder(), pool);

// ...

ReductionOps(first_recorder).sum(input, first_result);
```

Source: `tests/operations/reductions/test_reductions/test_scratch_buffer.cpp`

The test builds two such recorders. `TestEnvironment::runtime()` returns the `SushiRuntime::Runtime&` of the test program. The
recorder's third constructor argument is the `ExecutionMode`, `Auto` by default.

`ReductionOps(engine)` is the same view over `engine.recorder()`. A caller that builds its own
recorder owns the graph and runs it with the run verbs of `SushiRuntime::Graph`. `RandomOps` also
takes the `RNGStream` it draws from, and `IO` (`io/io.hpp`) takes an `Engine`, because every host
transfer runs the recorded graph first.

| View | Header | Family |
| --- | --- | --- |
| `BLASOps` | `blas/blas.hpp` | Dense BLAS: `Level1`, `Level2` and `Level3` in one class |
| `SparseOps` | `sparse/sparse.hpp` | Sparse BLAS |
| `ElementwiseOps` | `elementwise/elementwise.hpp` | Arithmetic and elementary functions, and `convert` |
| `NonLinearOps` | `nonlinear/nonlinear.hpp` | Activations and their backward forms |
| `LogicOps` | `logic/logic.hpp` | Comparisons, logical operations, `where`, `all`, `any` |
| `ReductionOps` | `reductions/reductions.hpp` | Whole-tensor and axis reductions |
| `RandomOps` | `random/random.hpp` | Distributions, initialisers, `shuffle`, `set_seed` |
| `SpatialOps` | `spatial/spatial.hpp` | Windowed and strided movement over NHWC tensors |
| `IO` | `io/io.hpp` | Host transfer, `.sushi`, `.npy` and raw files, printing |

A view holds a reference to its recorder, `RandomOps` one more to its stream, and no other
state. It can be copied and cannot be re-bound.

## The random stream

`RNGStream` (`engine/rng_stream.hpp`) is a seed and an offset counter under one mutex.
`reserve(count)` returns the seed with the current offset and advances the counter, so two
random tasks never share a Philox counter range, even when recorded from two threads. The stream
and the recorder lock one after the other, so two random tasks recorded from two threads may
reach the graph in the other order than their offsets.
`reseed(seed)` sets the seed and rewinds the offset, so one seed followed by one sequence of calls
draws the same numbers. `RandomOps::set_seed` calls it. The reservation is made at record time and
captured by the task.

# 5. `Engine`, the recorder and the views

`Engine` (`engine.hpp`) is the composition root a program constructs once on a
`RuntimeContext`, with the layout of the tensors it creates. It owns, in this order, the
`ScratchPool`, the `SushiRuntime::Internal::Graph::TaskGraph`, the `TaskRecorder` bound to both, the
`TensorFactory` and the `RNGStream`. Its interface is short:

- `create_tensor(dims, dtype, residency, device_index)`, from a brace list or a span.
- `set_mode` and `get_mode`, which forward to the recorder, and `get_layout`.
- `get_recorder()`, `get_tensor_factory()`, `get_rng()`, `get_scratch_pool()` and `get_context()`.
- `execute()`, which compiles the recorded graph and runs it through the context's execution
  engine, returning when every task has finished. It returns nothing, and a graph with no root
  returns at once. No thread may be recording on the engine while it runs.
- `discard_recorded()`, which drops every recorded task that has not run, with the operand
  storage and the scratch those tasks hold.

No accessor reaches the graph. The engine is not copyable or movable, and the context must
outlive it and every tensor it creates.

## `TaskRecorder`, `OperandSet` and `TaskHandle`

`TaskRecorder` (`engine/task_recorder.hpp`) records tasks on one graph under one mode. It has one
recording call:

`record_pair(meta, operands, family, body)` checks host reachability, draws a registration index
from the graph, and adds one task whose reads and writes are the operand set's regions. The body
is a copyable callable `(const Internal::KernelPairDispatcher&, sycl::queue&, const
std::vector<sycl::event>&) -> sycl::event` that must order its work behind the events it is
given. On the SYCL side that means every command group calls `h.depends_on(deps)`;
`check_layering.py` reports a command group without it.

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
noted. A tensor without storage adds nothing. Taking the region from `Tensor::region()` in one place is deliberate: a
`ResourceRegion` converts from a bare pointer, a bare pointer names the whole allocation, and two
views of one arena then look unrelated to the tracker, which emits no edge between them.

`TaskHandle` (`engine/task_handle.hpp`) is what every record call returns: the identity of the
recorder and the registration index of the task. A call that records nothing, such as an
elementwise operation on an empty tensor, returns the default handle, for which `is_valid()` is
false. The handle is a name for the task and gives no way to wait on it; `execute()` is the only
way to run the graph.

## Views

A view is a small class of record calls bound to a recorder. Each has two constructors, one
from an `Engine` and one from a `TaskRecorder` the caller owns:

```cpp
SushiRuntime::Internal::Graph::TaskGraph graph(ctx);
ScratchPool pool(ctx);
TaskRecorder recorder(graph, pool, ExecutionMode::ForceReference);

const TaskHandle recorded = BLASOps(recorder).axpy(2.0, x, y);
```

`BLASOps(engine)` is the same view over `engine.get_recorder()`. A caller that builds its own
recorder owns the graph and compiles and runs it as `Engine::execute` does. `RandomOps` also
takes the `RNGStream` it draws from, and `IO` (`io.hpp`) takes an `Engine`, because every host
transfer runs the recorded graph first.

| View | Header | Family |
| --- | --- | --- |
| `BLASOps` | `engine/blas.hpp` | Dense BLAS: `Level1`, `Level2` and `Level3` in one class |
| `SparseOps` | `engine/sparse.hpp` | Sparse BLAS |
| `ElementwiseOps` | `engine/math/elementwise.hpp` | Arithmetic and elementary functions, and `convert` |
| `NonLinearOps` | `engine/math/nonlinear.hpp` | Activations and their backward forms |
| `LogicOps` | `engine/logic/logic.hpp` | Comparisons, logical operations, `where`, `all`, `any` |
| `ReductionOps` | `engine/math/reductions.hpp` | Whole-tensor and axis reductions |
| `RandomOps` | `engine/math/random.hpp` | Distributions, initialisers, `shuffle`, `set_seed` |
| `SpatialOps` | `engine/spatial.hpp` | Windowed and strided movement over NHWC tensors |
| `IO` | `io.hpp` | Host transfer, `.sushi`, `.npy` and raw files, printing |

A view holds a reference to its recorder, `RandomOps` one more to its stream, and no other
state. It can be copied and cannot be re-bound.

## The random stream

`RNGStream` (`engine/rng_stream.hpp`) is a seed and an offset counter under one mutex.
`reserve(count)` returns the seed with the current offset and advances the counter, so two
random tasks never share a Philox counter range, even when recorded from two threads.
`reseed(seed)` sets the seed and rewinds the offset, so one seed followed by one sequence of calls
draws the same numbers. `RandomOps::set_seed` calls it. The reservation is made at record time and
captured by the task.

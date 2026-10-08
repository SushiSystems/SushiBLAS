# SushiBLAS Architecture

This page describes how SushiBLAS is built, layer by layer, for a contributor who has to change
it. Every class and file it names exists under `include/SushiBLAS/`, `src/` or `kernels/`. To use
the library, the [root README](../../README.md) is enough.

The page describes the tree as it is written at v0.1.0. That tree has not been built or run since
the interface break began, so nothing here is a statement that an operation works or that one
kernel is faster than another. No figure for speed appears on this page, because none has been
measured on this code.

---

## 1. What SushiBLAS is

SushiBLAS is a C++17 tensor and BLAS library on
[SushiRuntime](https://github.com/SushiSystems/SushiRuntime)'s task graph. The scheduler, the
execution engine and the USM allocator are SushiRuntime's. SushiBLAS adds a `Tensor`, an `Engine`
that owns one graph, and eight operation families: dense BLAS, sparse BLAS, elementwise
arithmetic, non-linear activations, logic and comparison, reductions, random generation and
spatial (windowed) operations.

Two commitments shape every operation.

1. **An operation is a recorded task.** `SushiBLAS::BLASOps(engine).gemm(A, B, C)` adds one task
   to the engine's graph, with `A` and `B` declared as reads and `C` as a write, and returns a
   `TaskHandle`. Nothing runs. SushiRuntime's dependency tracker orders the task against every
   other recorded task from those declarations. `engine.execute()` compiles the graph, runs it and
   returns when every task has finished.
2. **No vendor math library.** Every kernel is plain C++ or SushiBLAS's own SYCL. The local-memory
   tiling of GEMM and SYRK derives from portBLAS (Apache-2.0, attributed in
   [`NOTICE.md`](../../NOTICE.md)); the rest is written here, the Philox4x32-10 random generator
   included.

The library is built at C++17, the standard SushiRuntime is built to. `SushiRuntime::span`
(`core/span.hpp`) is `std::span` at C++20 and SushiRuntime's own type below it, chosen on
`__cplusplus`, and that choice reaches SushiRuntime's compiled ABI. Every span-taking entry point
in the SushiBLAS headers is inline, so the choice is made again in the consumer's translation
unit and never has to agree with a symbol compiled elsewhere. That is what a consumer at C++20
relies on. `tests/package/` is that consumer, set to C++20, and it has not been built against this
tree. The
pieces C++20 would supply are stand-ins: `core/format.hpp` for `std::format`,
`Core::Detail::SourceLocation` for `std::source_location`, and the value tag `Dispatch::ElementTag`
where a templated lambda would go.

---

## 2. The layered architecture

```
   ┌────────────────────────────────────────────────────────────────┐
   │  Engine                 composition root: owns the scratch     │  engine.hpp
   │                         pool, the graph, the recorder, the     │
   │                         tensor factory and the random stream   │
   ├────────────────────────────────────────────────────────────────┤
   │  Operation views        BLASOps · SparseOps · ElementwiseOps   │  engine/*.hpp
   │                         NonLinearOps · LogicOps · ReductionOps │  built from an Engine
   │                         RandomOps · SpatialOps · IO            │  or a TaskRecorder
   ├────────────────────────────────────────────────────────────────┤
   │  Seams                  Dispatch::execute_* and                │  engine/**/dispatch.hpp
   │                         Internal::execute_typed_task: take a   │  engine/task.hpp
   │                         TaskRecorder and an OperandSet, return │
   │                         a TaskHandle                           │
   ├────────────────────────────────────────────────────────────────┤
   │  TaskRecorder           the one caller of TaskGraph::add_task; │  engine/task_recorder.hpp
   │                         stamps the execution mode on the task  │
   ├───────────────────────┬────────────────────────────────────────┤
   │  Reference kernels    │  SYCL kernels                          │  kernels/,
   │  plain C++ on the     │  take the queue and the dependency     │  include/SushiBLAS/kernels/
   │  host                 │  events, return a sycl::event          │
   ├───────────────────────┴────────────────────────────────────────┤
   │  Support                launch and fold shapes, serial loops,  │  include/SushiBLAS/support/
   │                         accumulators, the device policy        │
   ├────────────────────────────────────────────────────────────────┤
   │  Tensor / Storage       shape, stride and dtype over a         │  tensor.hpp, storage.hpp,
   │                         ref-counted USM allocation             │  tensor_factory.hpp
   ├────────────────────────────────────────────────────────────────┤
   │  SushiRuntime           TaskGraph · RuntimeContext ·           │  not SushiBLAS's code
   │                         Residency · execution engine           │
   └────────────────────────────────────────────────────────────────┘
```

A layer depends on the layers below it and on none above. Four rules state what each layer may
know, and `tools/layering/check_layering.py` checks them from the include lines and the call
sites:

- **`Engine` knows no operation.** `engine.hpp` includes the four bricks it composes
  (`engine/rng_stream.hpp`, `engine/scratch_pool.hpp`, `engine/task_recorder.hpp`,
  `engine/execution_mode.hpp`) and no view, seam or experimental header. It has no `blas()` or
  `elementwise()` accessor.
- **A view knows the engine only as a source of bricks.** A view's constructor takes an `Engine&`
  and calls `get_recorder()` (and `get_rng()` for `RandomOps`), or takes the `TaskRecorder&`
  directly. An operation source under `src/engine/` never includes `engine.hpp`.
- **Only `TaskRecorder` inserts into the graph.** No seam, view or operation source calls
  `TaskGraph::add_task`.
- **A kernel knows neither `Engine` nor `Tensor`.** A header under `kernels/` or
  `include/SushiBLAS/kernels/` includes no `engine/` header, no `tensor.hpp` and no sparse
  operand header. It receives pointers, extents and strides. A support header obeys the same rule
  and also includes no kernel.

---

## 3. Execution modes

`ExecutionMode` (`engine/execution_mode.hpp`) has three enumerators, and `Engine::set_mode` sets
the mode of every operation recorded from then on.

| Mode | What a task recorded under it runs |
| --- | --- |
| `Auto` | The library's choice, which is the SYCL kernel. The default. |
| `ForceSycl` | The SYCL kernel. |
| `ForceReference` | The reference path, on the host thread that executes the task. |

There is no `VERIFY` mode. `set_mode` is `noexcept` and accepts all three values.

**The mode belongs to the task.** `TaskRecorder::record_pair` builds an
`Internal::KernelPairDispatcher` from the recorder's mode at the moment of the call and stores it
in the task. Changing the mode afterwards changes later record calls only, so one graph can hold
tasks of different modes:

```cpp
auto from_reference = engine.create_tensor({64, 64});
auto from_sycl = engine.create_tensor({64, 64});

BLASOps blas(engine);
engine.set_mode(ExecutionMode::ForceReference);
blas.gemm(A, B, from_reference);
engine.set_mode(ExecutionMode::ForceSycl);
blas.gemm(A, B, from_sycl);
engine.execute();
```

This is how the tests compare the two paths of an operation: the same call under each mode into
two tensors, then a comparison of the tensors.

`KernelPairDispatcher::run(q, deps, reference_fn, sycl_fn)` is the single mode switch. Under
`ForceReference` it waits on the dependency events, calls `reference_fn` on the calling thread
and returns an empty event. Otherwise it returns `sycl_fn(q, deps)`.

**Every family has a reference path**, and what that path is differs by family:

| Family | The reference path | Shared with the SYCL path |
| --- | --- | --- |
| Dense BLAS, sparse, spatial | A reference kernel under `kernels/reference/`, written separately from the SYCL kernel | The record-time checks |
| Elementwise, non-linear, logic | `Serial::map` or `Serial::zip_inplace` (`support/serial_1d.hpp`): one loop in index order over the operation's functor | The functor, its narrowed scalars, its declared read pointers, the record-time checks |
| Reductions | `Serial::reduce`, `reduce_indexed` or `reduce_axis` (`support/serial_reduce.hpp`): one fold in index order | The transform and finalize functors, and the accumulator class of every fold that is not a sum |
| Random | The same `draw_block` functor in a loop over the blocks, through `Dispatch::RandomFill` | Philox, the mapping from words to draws, the `{seed, offset}` reservation |

For the last three rows the two paths differ in the loop and, where the functor calls a math
builtin, in the math library. A disagreement between the modes there points at a launch, indexing
or chunking error, a race between work-items, a dropped dependency, or two math libraries rounding
differently. It does not point at the formula: if the functor of `silu_backward` is wrong, both
paths compute the same wrong number and agree. For those families the reference path is a second
execution of one formula, and only the kernel pairs of the first row are two derivations.

A sum is the one fold whose accumulator also differs: the host folds it in
`Kernels::ReferenceSum` (`double`, compensated for FLOAT64), the device in `Reduce::CompensatedSum`
over the element's accumulation type. The two therefore add in different orders and types and are
not expected to agree to the bit.

**`ForceReference` needs host-reachable operands.** A reference path dereferences its operands
on the host, which `Residency::Device` storage does not allow. `OperandSet` notes a device-only
operand as it is declared, and `TaskRecorder::require_host_reachable` throws
`UnsupportedExecution` at record time, before any task or scratch exists. A tensor an epilogue or
a functor declares through `reads` is part of the same operand set and is refused the same way.

---

## 4. `Tensor`, `Storage` and memory

A `Tensor` (`tensor.hpp`) is metadata: `shape`, `strides`, `rank`, `dtype`, `layout`,
`num_elements`, a `storage` pointer and a `storage_offset`. `transpose()`, `reshape()` and
`slice()` return a new `Tensor` over the same `SushiRuntime::sushi_ptr<Storage>` and move no
data. `K_MAX_TENSOR_RANK` is 6.

`Core::DataType` is `Half`, `Float32`, `Float64`, `Complex32`, `Complex64`, `Int32`, `Int64`, with
the values 0 to 6 written out in `core/common.hpp`. The values are a file-format contract: none
changes, and a new type is appended. `Core::bytes_per_element()` is the one mapping from a dtype
to its width.

**The integer types are index and mask types.** `iamax`, `argmax`, `argmin` and
`count_nonfinite` answer with an `INT64` because an index or a count stored in a float is exact
only up to the mantissa. Beyond those results the integer dtypes are served by `copy`,
`constant`, the comparison and logic family, `io`, and the elementwise operations that have an
integer meaning (`add`, `sub`, `mul`, `neg`, `abs`, `min`, `max`, `clamp`), whose arithmetic
wraps. Every other operation rejects them by name at record time, through `Dispatch::Domain`
([section 6](#6-the-dispatch-seams-and-the-fusion-hooks)).

**Layout and the batch axis.** `Core::Layout` (`RowMajor`, `ColumnMajor`) is fixed at
construction and describes the order inside one matrix. `Tensor::layout_strides()` is the single
definition: the constructor lays a tensor out with it and `is_contiguous()` compares against it.
For rank 3 and above the trailing two axes are the matrix and take the layout's order, and every
leading batch axis is outermost in either layout, so a rank-3 `{b, m, n}` tensor has batch stride
`m * n` under both. That matches how the batched BLAS drivers read `A.strides[rank - 3]` as the
distance between matrices.

**Offsets count elements.** `storage_offset` is in elements, and `data()` scales it by the
element width, so a view is aligned for its own element type. A view that is not contiguous, such
as a slice along an inner axis, keeps its parent's strides and is built through the
stride-explicit constructor. `Tensor::element_span()` gives the distance from its first element
to its last, `1 + sum((shape[i] - 1) * strides[i])`, which is what the capacity check and
`region()` measure.

An operation addresses a tensor through three accessors and never through the allocation base,
which is private to `Storage`:

| Accessor | Gives | Used for |
| --- | --- | --- |
| `data()` / `data_as<T>()` | The view's first element | Every kernel pointer |
| `data_or_null()` | The same, or `nullptr` without storage | A source that tolerates an empty operand |
| `region()` | The view's byte interval, keyed on the allocation base | Every declared read and write |

`region()` is what lets a memory planner pack an arena. Two disjoint views of one allocation give
regions that do not overlap, so the tracker does not order them, and two overlapping views are
ordered.

### Storage and alignment

`Storage` (`storage.hpp`) owns one reference-counted USM allocation with a
`SushiRuntime::API::Residency` (`Shared`, `Device` or `Host`) on one device queue of the
`RuntimeContext`. Every allocation base is aligned to `K_STORAGE_ALIGNMENT`, 128 bytes, which is
`SushiRuntime::Core::DEFAULT_ALIGNMENT` (a `static_assert` holds the two equal), and the allocated
size is the requested size rounded up to that. The context is a non-owning pointer and must
outlive every `Storage`. The allocation is freed when the last reference to the `Storage` goes.

`TensorFactory` (`tensor_factory.hpp`) allocates dense tensors of one default layout from one
context. `create(dims, dtype, residency, device_index)` checks the rank, the sign of each
dimension, the byte count for overflow and the device index, and returns a tensor without storage
when a dimension is 0. `Engine::create_tensor` forwards to it.

### Retention

A recorded task keeps its operands' storage alive. `OperandSet` takes one reference to the
`Storage` of every tensor declared on it, in an `Internal::StorageKeepAlive` (six references
inline, the rest on the heap), and `record_pair` copies that list into the task. The storage of a
tensor the caller drops between the record call and `execute()` is therefore still referenced when
the task runs. A pointer hidden inside a bare functor is not a declared operand: nothing retains it,
orders against it, or checks it under `ForceReference`.

### The scratch pool

An operation that needs device scratch (the staged partials of a fold, the panels of a packed
product, the wide solution of a substitution, the partial sums of a sparse scatter) leases it at
record time and the task holds the lease by value.

- `ScratchPool` (`engine/scratch_pool.hpp`) owns a reference-counted `Detail::ScratchShelf`
  (`engine/scratch_shelf.hpp`) of idle device blocks per device queue. The engine owns one pool
  and the recorder hands it out through `get_scratch_pool()`.
- `ScratchBuffer` (`engine/scratch.hpp`) is the lease. The block goes back to the shelf when the
  last copy of the handle is destroyed, which for a recorded task is when the graph drops the
  task.
- Blocks are pooled in six size classes from 128 to 4096 bytes, at most 64 idle blocks per class
  and device. A larger request is allocated for the lease and freed when it ends.
  `ScratchPool::trim()` frees the idle blocks.
- `OperandSet::scratch(buffer)` declares the lease as a written region of the task and retains
  its storage.

Under `ForceReference` the reduction seams and `Internal::allocate_blas_scratch` lease nothing:
a host fold stages no partials.

---

## 5. `Engine`, the recorder and the views

`Engine` (`engine.hpp`) is the composition root a program constructs once on a
`RuntimeContext`, with the layout of the tensors it creates. It owns, in this order, the
`ScratchPool`, the `SushiRuntime::Graph::TaskGraph`, the `TaskRecorder` bound to both, the
`TensorFactory` and the `RngStream`. Its interface is short:

- `create_tensor(dims, dtype, residency, device_index)`, from a brace list or a span.
- `set_mode` and `get_mode`, which forward to the recorder, and `get_layout`.
- `get_recorder()`, `get_tensor_factory()`, `get_rng()`, `get_scratch_pool()` and `get_context()`.
- `execute()`, which compiles the recorded graph and runs it through the context's execution
  engine, returning when every task has finished. It returns nothing, and a graph with no root
  returns at once. No thread may be recording on the engine while it runs.

No accessor reaches the graph. The engine is not copyable or movable, and the context must
outlive it and every tensor it creates.

### `TaskRecorder`, `OperandSet` and `TaskHandle`

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

### Views

A view is a small class of record calls bound to a recorder. Each has two constructors, one
from an `Engine` and one from a `TaskRecorder` the caller owns:

```cpp
SushiRuntime::Graph::TaskGraph graph(ctx);
ScratchPool pool(ctx);
TaskRecorder recorder(graph, pool, ExecutionMode::ForceReference);

const TaskHandle recorded = BLASOps(recorder).axpy(2.0, x, y);
```

`BLASOps(engine)` is the same view over `engine.get_recorder()`. A caller that builds its own
recorder owns the graph and compiles and runs it as `Engine::execute` does. `RandomOps` also
takes the `RngStream` it draws from, and `IO` (`io.hpp`) takes an `Engine`, because every host
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

### The random stream

`RngStream` (`engine/rng_stream.hpp`) is a seed and an offset counter under one mutex.
`reserve(count)` returns the seed with the current offset and advances the counter, so two
random tasks never share a Philox counter range, even when recorded from two threads.
`reseed(seed)` sets the seed and rewinds the offset, so one seed followed by one sequence of calls
draws the same numbers. `RandomOps::set_seed` calls it. The reservation is made at record time and
captured by the task.

---

## 6. The dispatch seams and the fusion hooks

A seam is the function a view method calls to record its task. Seams take a `TaskRecorder&` and
return a `TaskHandle`. There are two kinds.

**The kernel-pair seam** serves the dense BLAS, sparse and spatial families.
`Internal::execute_typed_task<Dom>(recorder, meta, operands, dtype, family, op_func)`
(`engine/task.hpp`) checks the dtype against the domain and records one task whose body switches
on the runtime dtype and calls `op_func(T{}, dispatcher, q, deps)` with a value of the matching
element type. `op_func` calls `dispatcher.run` with its reference kernel and its SYCL kernel.
`Internal::execute_blas_task` (`engine/blas/utils.hpp`) is the same call with the family string
`"BLAS operation"`. Section 5 of [CONTRIBUTING.md](../CONTRIBUTING.md) shows a whole operation
written on it.

**The functor seams** are the public `SushiBLAS::Dispatch::execute_*` functions. Each takes a
caller-written functor, inlines it into one SYCL kernel, and runs the same functor in a serial
host loop under `ForceReference`:

| Header | Executors |
| --- | --- |
| `engine/math/elementwise/dispatch.hpp` | `execute_unary_inplace`, `execute_binary`, `execute_zip_inplace` |
| `engine/math/nonlinear/dispatch.hpp` | `execute_nonlinear_forward`, `execute_nonlinear_backward` |
| `engine/logic/dispatch.hpp` | `execute_logic_unary`, `execute_logic_binary`, `execute_logic_ternary` |
| `engine/math/reductions/dispatch.hpp` | `execute_reduction`, `execute_index_reduction`, `execute_counting_reduction`, `execute_folding_reduction`, `execute_folding_span_reduction`, `execute_folding_axis_reduction` |
| `engine/math/random/dispatch.hpp` | `execute_random` |
| `engine/blas/dispatch.hpp` | `execute_gemm` |

A functor that reaches a device is device-copyable and trivially copyable, captures by value,
dereferences no host pointer, allocates nothing and throws nothing. The compiler does not
diagnose all of these.

**`Dispatch::Domain`** (`engine/dispatch.hpp`) is a bit set over element-type families: `Real`,
`Complex`, `Integral` and the unions `RealAndComplex`, `RealAndIntegral` and `All`. It says
which element types a functor may be instantiated for. `require_domain<Dom>` rejects a dtype
outside the domain at record time with `std::invalid_argument`. `with_element_type<Dom>` is the
dtype switch of the functor seams: an arm outside the domain is never instantiated, which is what
lets a functor that is ill-formed for `std::complex` be dispatched on a runtime dtype. An arm is
written as a generic lambda over `Dispatch::ElementTag<T>`, or, at C++20, as a templated lambda.
`execute_typed_task` spells the same seven arms out in a named class, `Detail::TypedTaskBody`,
because its body takes a value and returns an event.

### The fusion hooks

**No automatic fusion pass exists.** SushiBLAS does not inspect the graph, match a chain of
operations or merge two tasks. Two recorded tasks are two launches. Fusion is something the
caller writes: one functor that does the work of several operations, handed to one seam call.
The library supplies the places to hand it to.

*Composing a functor.* A consumer that wants `x * scale + shift` as one kernel writes it as one
functor and calls `execute_unary_inplace` once. `execute_zip_inplace` does the same over N
same-shaped tensors, each passed to the functor as an lvalue, so an optimiser step that reads and
updates five tensors is one task. All N are declared as both read and written, since the seam
cannot see which the functor touched.

*`bind_scalars`* (`engine/math/elementwise/scalar_binding.hpp`) attaches scalar parameters to a
functor. The scalars are `Dispatch::ElementScalar` values, narrowed on the host once per element
type, and the functor receives them after its elements in that type. A HALF or FLOAT32 kernel
then does no `double` arithmetic for a parameter the public signature takes as `double`.

*`bind_reads`* (`engine/math/elementwise/read_binding.hpp`) declares the tensors a functor reads
at execution time beyond the seam's own operands: a table, a gain, a mask. `read_as_element(t)`
hands the functor a `const T*` in the operation's element type and `read_as<U>(t)` a `const U*`.
The seam checks each declared read (storage, at least one element, the declared dtype,
contiguous, no overlap with a tensor the operation writes), adds it to the operand set so the
graph orders against it and retains it, and passes the functor only the pointer. The two bindings
nest: `bind_reads(bind_scalars(fn, ...), ...)`, pointers before scalars.

```cpp
// Two scalars, narrowed on the host to the element type of each dtype arm.
Dispatch::execute_unary_inplace(recorder, t, "nn.scale_shift", "nn.scale_shift"_op,
    Dispatch::bind_scalars([](auto x, auto scale, auto shift) { return x * scale + shift; },
                           Dispatch::ElementScalar(0.5), Dispatch::ElementScalar(1.0)));

// A tensor the functor reads at execution time, declared so the graph orders against it.
Dispatch::execute_binary(recorder, A, B, C, "nn.gated_add", "nn.gated_add"_op,
    Dispatch::bind_reads([](auto a, auto b, const auto* g) { return a + b * g[0]; },
                         Dispatch::read_as_element(gain)));
```

The `"..."_op` literal comes from `SushiRuntime::Graph::Literals`.

*The GEMM epilogue.* `Dispatch::execute_gemm` records
`C = epilogue(alpha * op(A) * op(B) + beta * C, row, col)` as one task. The epilogue is a functor
`T operator()(T value, int64_t row, int64_t col) const`, applied to each finished element just
before the store. The coordinates make a bias broadcast expressible, so a product, a bias-add and
an activation are one task. `Level3::gemm` is this function with
`Kernels::GemmIdentityEpilogue`, and the reference kernel takes the same parameter, so
`ForceReference` runs a fused GEMM too.

```cpp
struct BiasRelu
{
    const float* bias;   // one value per column of C

    template<typename T>
    T operator()(T value, int64_t, int64_t col) const
    {
        const T y = value + static_cast<T>(bias[col]);
        return y > T(0) ? y : T(0);
    }
};

// One task: the product, the bias broadcast and the ReLU.
Dispatch::execute_gemm(recorder, X, W, Y,
                       Core::Transpose::NoTrans, Core::Transpose::NoTrans, 1.0, 0.0,
                       "nn.linear_relu", "nn.linear_relu"_op,
                       Dispatch::gemm_epilogue(BiasRelu{bias.data_as<float>()}).reads(bias));
```

`Dispatch::gemm_epilogue(functor)` bundles the functor with the tensors it touches, and
`.reads(t)` and `.writes(t)` declare them. `execute_gemm` merges those declarations into the
task's operand set, so the bias is ordered against whatever writes it, retained until the task is
dropped, and refused under `ForceReference` when it is device-only. A bare functor is accepted
and declares nothing. `execute_gemm` defaults to `Domain::Real`; an epilogue that is well-formed
for `std::complex` passes `Domain::RealAndComplex`.

Two warnings travel with the epilogue. An activation fused into the forward pass of training
gives a wrong gradient unless the epilogue also writes its pre-activation value to a buffer the
backward pass reads, declared through `.writes`; nothing throws when this is missing. And under
`ForceReference` the epilogue runs on the host, so a pointer it carries must be host-readable.

---

## 7. Kernels: reference and SYCL

A kernel is a function template on the element type that receives pointers, extents, strides and
flags. The dense BLAS, sparse and spatial families have a kernel pair per operation:

- **`kernels/reference/<family>/...`** is plain C++ with no dependency beyond the support
  headers. It handles both `Core::Layout` values and states the result the operation is defined
  to have.
- **`kernels/sycl/<family>/...`** takes the `sycl::queue&` and a
  `const std::vector<sycl::event>& deps`, makes every command group depend on `deps`, and returns
  the event of its last submission.

`kernels/` is private and never installed. A kernel a public seam instantiates in a consumer's
translation unit cannot be private, so those live under `include/SushiBLAS/kernels/` and are
installed: GEMM's reference and SYCL kernels with the packed product they share with SYRK, the
epilogue headers, the access policies (`operand_access.hpp`, `symmetric_access.hpp`,
`band_access.hpp`, `packed_access.hpp`, `conjugation.hpp`) and the two Givens headers.

The elementwise, non-linear, logic, reduction and random families have no kernel pair. Their SYCL
kernel is a launch primitive of the support layer around the operation's functor, and their
reference path is the serial loop of [section 3](#3-execution-modes).

### The support layer

`include/SushiBLAS/support/` sits below the kernels and the engine and includes neither.
[Its README](../../include/SushiBLAS/support/README.md) holds the reasoning behind each header.

| Header | Holds |
| --- | --- |
| `device_policy.hpp` | `Device::Profile`, probed once per device by `profile_for`: CPU or not, dedicated local memory, `fp64` and `fp16` support, local memory bytes, work-group and sub-group sizes, compute units, cache bytes. The launch geometry (`plan_1d`, `flat_work_group_size`, `vector_width`, `hosts_local_tile`). `Device::CacheBlocking`, the byte budgets a cache-blocked kernel sizes its blocks from |
| `launch_1d.hpp` | `Launch::map`, `zip_inplace` and `indexed`: the device launch of every per-element kernel, a grid-stride `nd_range` with a 128-bit wide pass where the device wants one and every operand is 16-byte aligned |
| `serial_1d.hpp` | `Serial::map`, `zip_inplace` and `indexed`: the same three shapes as one loop on the calling thread |
| `reduce_1d.hpp` | `Reduce::reduce` and `reduce_indexed`, the fixed-order whole-tensor fold, and the accumulators: `CompensatedSum`, `CompensatedComplexSum`, `ScaledSumOfSquares`, `RunningProduct`, `NaNPropagatingMax`, `NaNPropagatingMin`, `ExactSum`, `RunningArgMax`, `RunningArgMin` |
| `reduce_axis.hpp` | `Reduce::AxisSpan`, `reduce_axis` and `reduce_segments`: many folds side by side over one buffer |
| `serial_reduce.hpp` | `Serial::reduce`, `reduce_indexed` and `reduce_axis`: the host folds |
| `accumulator.hpp` | `AccumulatorOf<T>`: the type a device kernel sums `T` in, which is `T` except that HALF sums in FLOAT32 |
| `reference_accumulator.hpp` | `ReferenceAccumulatorOf<T>` and `Kernels::ReferenceSum<T>`: a reference kernel sums a real type in `double` and a complex type in `std::complex<double>`, compensated where the wide type is the element type itself |
| `running_solution.hpp` | `RunningSolution`: carries a substitution's solved elements in the accumulation type between steps |
| `complex_arithmetic.hpp` | Conjugation, the four-product complex product and Smith's quotient, written so a device can run them |
| `orthonormal_sweep.hpp` | The vector layout and the two-sweep constant the orthonormalisation kernels share |
| `sparse_assembly.hpp` | `Sparse::assemble_csr` and its input checks, on the host |
| `spatial_geometry.hpp` | `Spatial::ConvGeometry`, `output_extent`, `WindowSpan`, `ColumnSpan`, `TapSpan` |

**Cache blocking.** `Device::CacheBlocking` holds three byte budgets (the bytes the innermost
loop keeps resident, the packed block the inner loops sweep again, and the block walked once per
sweep), two thresholds below which a kernel does not pack, and the work-items a blocked launch
starts per compute unit. `cache_blocking_for(cache_bytes)` derives the budgets from the cache a
device reports, and from an assumed 256 KiB when it reports none. The values are starting points
derived from cache sizes; none has been tuned against a measurement.

**Determinism.** SushiBLAS links `sushiruntime_deterministic_fp` PUBLIC, and its kernels keep the
order of every floating-point sum a function of the device, the element type and the extents.
That is why the floating-point folds use `Reduce::reduce` and not `sycl::reduction`, whose
combination order the SYCL specification leaves free, and why every backward in the spatial
family and every sparse scatter is written without an atomic add. The design aim is a result
that repeats bit for bit on one device; the regression tests that hold it have not been run on
this tree. Two devices, or the two modes, may plan different shapes and differ in the last bits
of a sum.

### Dense BLAS

**Flags.** `core/blas_types.hpp` declares the four flag enumerations, with the CBLAS values:

| Enumeration | Enumerators |
| --- | --- |
| `Core::Transpose` | `NoTrans`, `Trans`, `ConjTrans` |
| `Core::Uplo` | `Upper`, `Lower` |
| `Core::Diag` | `NonUnit`, `Unit` |
| `Core::Side` | `Left`, `Right` |

A flag outside its enumeration throws `std::invalid_argument` at record time (`require_flag`).

**Scalars.** `BlasScalar` (`engine/blas/scalar.hpp`) is the scalar argument of a BLAS call. It
converts from `double` and from `std::complex<double>`, holds the value as given, and
`as<T>()` rounds it once to the element type inside the task. A scalar with a non-zero imaginary
part passed to a real tensor throws at record time (`require_real_scalar`). The scalars the
standard defines as real (`rot`'s `c` and `s`, the `alpha` of `her`, `hpr` and `herk`, the `beta`
of `herk` and `her2k`, and `ger`'s `alpha`) are `double`.

```cpp
auto Z = engine.create_tensor({8, 8}, Core::DataType::Complex64);
auto H = engine.create_tensor({8, 8}, Core::DataType::Complex64);
auto zx = engine.create_tensor({8}, Core::DataType::Complex64);
auto zy = engine.create_tensor({8}, Core::DataType::Complex64);

BLASOps blas(engine);
blas.herk(Z, H, Core::Uplo::Upper, Core::Transpose::ConjTrans, 1.0, 0.0);   // H = Z^H Z
blas.axpy(std::complex<double>(0.0, 1.0), zx, zy);                           // zy = i*zx + zy
blas.trsv(H, zx, Core::Uplo::Upper, Core::Transpose::NoTrans, Core::Diag::NonUnit);
```

**Routines.** The dtypes are HALF, FLOAT32, FLOAT64, COMPLEX32 and COMPLEX64 unless a row says
otherwise.

| Level | Routines |
| --- | --- |
| 1 | `axpy`, `scal`, `copy` (every dtype, the integers included), `swap`, `dot` (real only), `dotu`, `dotc`, `nrm2`, `asum`, `iamax` (zero-based `INT64` result), `rot`, `rotg`, `rotm` (real only), `rotmg` (FLOAT32 and FLOAT64) |
| 2, general | `gemv`, `ger` (real only), `geru`, `gerc` |
| 2, symmetric and Hermitian | `symv`, `hemv`, `syr`, `her`, `syr2`, `her2` |
| 2, triangular | `trmv`, `trsv` |
| 2, band | `gbmv`, `sbmv`, `hbmv`, `tbmv`, `tbsv` |
| 2, packed | `spmv`, `hpmv`, `tpmv`, `tpsv`, `spr`, `hpr`, `spr2`, `hpr2` |
| 3 | `gemm`, `symm`, `hemm`, `trsm`, `syrk`, `herk`, `syr2k`, `her2k` |

`trmm` is not implemented. The Level 3 routines are batched above rank 2: the output sets the
batch count, and an input of rank 2 is reused for every matrix. A Level 2 matrix has rank 2. A
band operand is the band storage of the standard with its `kl`, `ku` or `k` stated in the call,
and a packed operand is a rank-1 tensor of `n * (n + 1) / 2` elements. A Hermitian routine does
not read the imaginary part of the diagonal and leaves an updated diagonal real.

**Operands are described once.** `matrix_operand.hpp`, `vector_operand.hpp`, `band_operand.hpp`,
`packed_operand.hpp` and `scalar_operand.hpp` each turn a tensor into the extents, leading
dimension and increment a kernel takes, and throw `std::invalid_argument` on a view the kernel
cannot read (a non-contiguous matrix, a zero or negative increment, mixed layouts, a batch count
that does not match). `task_metadata.hpp` records the scalars and flags in the task's metadata.

**Kernel shapes.** These are the shapes as written; none is a claim about speed.

- *Level 1.* The maps (`axpy`, `scal`, `copy`, `swap`, `rot`, `rotm`) launch through
  `Launch::indexed`. `asum`, `nrm2`, `dot`, `dotu`, `dotc` and `iamax` fold through
  `Reduce::reduce_indexed` with a leased scratch for the partials. `rotg` and `rotmg` are scalar
  routines.
- *GEMV, SYMV, HEMV, TRMV.* `row_reduction.hpp` launches one work-group per row for long rows and
  one work-item per row otherwise, each folding in a fixed order. TRMV takes a snapshot of `x`
  first, then computes one row product per element.
- *TRSM and TRSV.* `trsm.hpp` is a blocked substitution: each diagonal block is solved for every
  right-hand-side vector in parallel, then subtracted from the unknowns still to be solved in a
  second parallel launch. `triangular_system.hpp` decodes every side and flag into one
  substitution, and `triangular_plan.hpp` picks the block size from the device. TRSV calls the
  same kernel with one right-hand side. The solved elements are carried in the accumulation type
  between steps through `RunningSolution`, in a leased workspace when that type is wider than the
  element.
- *TBMV and TPMV.* `triangular_product.hpp` snapshots x, then computes one row product per
  work-item, as TRMV does.
- *TBSV and TPSV.* `triangular_solve.hpp` is the blocked substitution of TRSM for one vector:
  one work-item solves a diagonal block, then one work-item per row subtracts it from the rows
  it reaches. `plan_vector_solve` keeps a band too narrow to repay the launches in one block.
- *GEMM.* `Kernels::SYCL::gemm` chooses among three shapes from the `Device::Profile` and logs
  the choice at `SB_LOG_INFO`. `gemm_local_tiled` stages tiles of A and B in work-group local
  memory, for a device with dedicated local memory. `gemm_no_local` uses no local accessor and no
  barrier. The packed product (`packed_product.hpp`, `packed_panel.hpp`,
  `packed_micro_kernel.hpp`) packs panels of both operands into leased scratch and runs a
  register-block micro-kernel over them, one work-item per worker; `selects_packed_product`
  chooses it on a CPU device when the output spans at least four register blocks on each axis and
  the operands hold at least the device's cache in bytes. Without the scratch, or when the plan is
  not selected, the kernel falls back to one of the other two.
- *SYMM and HEMM* run the GEMM kernels through `symmetric_access.hpp`, which reads the mirrored
  triangle. *SYRK and HERK* use the packed product with a triangular store where the plan selects
  it and a local-memory tile otherwise. *SYR2K and HER2K* use a local-memory tile. The tiled SYRK
  and SYR2K throw `std::runtime_error` at execute time on a device that cannot host the tile.

`beta` exactly zero skips reading the output in every routine that takes it, and the operand set
declares the output as written only in that case.

### Sparse BLAS

The sparse operands are plain structs of tensors, so they are views like any other:

- `SparseMatrix` (`sparse_matrix.hpp`) is CSR as three rank-1 tensors, `row_offsets`
  (`rows + 1` elements), `column_indices` and `values`, with `rows`, `cols` and an
  `index_base`. The index tensors are `INT32` or `INT64` and share a dtype.
- `SparseVector` (`sparse_vector.hpp`) is `indices` and `values` with a `size` and an
  `index_base`, no position stored twice.
- `Core::IndexBase` (`core/sparse_types.hpp`) is `Zero` or `One`.

`SparseBuilder` (`sparse_builder.hpp`) builds them from host data on one device of a
`TensorFactory`. `matrix_from_triplets` takes COO arrays, adds the values of repeated
`(row, column)` pairs and orders each row by ascending column. `matrix_from_dense` keeps every
element that is not zero. `vector_from_entries` takes positions and values. The builder throws
`std::runtime_error`, as the factory does.

```cpp
const std::vector<int32_t> rows = {0, 0, 1, 2};
const std::vector<int32_t> cols = {0, 2, 1, 2};
const std::vector<float> values = {4.0f, 1.0f, 3.0f, 2.0f};

SparseBuilder builder(engine.get_tensor_factory());
const SparseMatrix S = builder.matrix_from_triplets(3, 3, rows, cols, values);

SparseOps sparse(engine);
sparse.spmv(S, x, y);                                         // y = S x
sparse.spmv(S, x, y, Core::Transpose::Trans, 1.0, 1.0);       // y = S^T x + y
```

`SparseOps` records thirteen routines:

| Kind | Routines |
| --- | --- |
| Sparse vector with dense vector | `axpyi`, `doti` (real only), `dotui`, `dotci`, `gthr`, `gthrz`, `sctr`, `roti` |
| CSR matrix with dense vector | `spmv`, `spsymv`, `sphemv`, `sptrsv` |
| CSR matrix with dense matrix | `spmm` |

The record-time checks (`engine/sparse/`) cover the index base, the rank, length, stride and
storage of each array, the index dtype and whether it can hold the largest index. They do not
read the index values: a kernel takes the structure on trust, and
`kernels/reference/sparse/structure.hpp` holds the host checks the reference kernels assert in a
build without `NDEBUG`.

**The scattering scheme.** A product with `NoTrans` gives each output row to one work-item,
which adds that row's terms in stored order. A transposed product has no such owner: output `j`
is column `j`, and every row scatters into the output. `kernels/sycl/sparse/row_blocks.hpp` makes
that safe without atomic adds, in two stages. The rows are cut into consecutive blocks that hold
equal shares of the stored entries, found by a binary search of the row offsets, and one
work-item per block zero-fills a private slice of leased scratch and adds its rows' terms into
it. A second launch then gives each output to one work-item, which adds that output's partial
from every block in ascending block order and stores the result with `alpha` and `beta`. No
element has two writers in either stage, and each output still receives its terms in ascending
row order. `scatter_blocks` picks the block count as the smallest of the work-items the device
keeps resident, two partial sums per stored entry, and 256. `spsymv` and `sphemv` use the same
scheme, since each stored entry also contributes to its mirror. A transposed `spmm` gives each
column of the result to one work-item, which scatters that column serially. `sptrsv` is one
work-item.

### Spatial

`SpatialOps` moves numbers around an NHWC grid: gathers (`im2col`, `col2im`,
`upsample_nearest`, `concat_channels`, `slice_channels`, `layout_convert`), windowed reductions
(`pool_max`, `pool_avg`), broadcasts (`channel_affine`, `broadcast_spatial`), the backward form
of each that has one, and the windowed product. The family serves `Domain::Real`, except
`concat_channels` and `slice_channels`, which move elements and serve every dtype. Parameter
ownership, training mode and differentiation belong to the layer above.

`Spatial::ConvGeometry` carries the kernel size, stride, dilation and the leading padding.
`Spatial::window_span_for` (`engine/spatial/window.hpp`) resolves the operands' strides into a
`WindowSpan` of element counts on the host, so a kernel carries no layout branch.

**The windowed product.** `window_dot(in, weights, out, g)` sums each channel's window weighted
by that channel's taps: `in` is `[N, H, W, C]`, `weights` is `[kernel_h, kernel_w, C]`, and a tap
outside the input adds nothing. It is the per-channel (depthwise) product, computed without
materialising a column matrix. `window_dot_backward` is the gradient with respect to the input
and `window_dot_weights_backward` the gradient with respect to the taps.

```cpp
Spatial::ConvGeometry g;
g.kernel_h = 3;
g.kernel_w = 3;
g.pad_top = 1;
g.pad_left = 1;

auto image = engine.create_tensor({1, 32, 32, 16});   // NHWC
auto taps = engine.create_tensor({3, 3, 16});         // [kernel_h, kernel_w, C]
auto out = engine.create_tensor({1, 32, 32, 16});

SpatialOps(engine).window_dot(image, taps, out, g);
```

**Every backward is a gather.** `col2im` written as "each column entry adds into its pixel" needs
a floating-point atomic add wherever windows overlap, and the order of those additions varies
between runs. Each output element instead finds the inputs that contributed to it and sums them in
a fixed order. `pool_max_backward` adds a test on the recorded index to the same walk. A HALF sum
in these kernels is formed in FLOAT32 and rounded once, and HALF storage on a device without the
`fp16` aspect is refused at record time.

### Reductions

`ReductionOps` folds a contiguous tensor into a one-element result tensor, or along an axis into
a vector. The value reductions serve HALF, FLOAT32 and FLOAT64.

| Operation | Result | Empty input | NaN |
| --- | --- | --- | --- |
| `sum`, `asum`, `nrm2` | Same dtype | Writes 0 | IEEE arithmetic: a NaN term gives NaN |
| `prod` | Same dtype | Writes 1 | IEEE arithmetic: a NaN factor gives NaN |
| `count_nonfinite` | `INT64` | Writes 0 | Counts every NaN and every infinity |
| `sum_along` | Rank-1, same dtype | Writes 0 to every segment | IEEE arithmetic, per segment |
| `mean`, `var`, `std`, `logsumexp` | Same dtype | Throws `std::invalid_argument` | IEEE arithmetic: a NaN term gives NaN |
| `max`, `min` | Same dtype | Throws `std::invalid_argument` | Any NaN element makes the result NaN |
| `max_along`, `min_along` | Rank-1, same dtype | Throws `std::invalid_argument` | A NaN makes its segment's result NaN |
| `mean_spatial` | `[N, C]` from `[N, H, W, C]` | Throws `std::invalid_argument` | IEEE arithmetic, per segment |
| `argmax`, `argmin` | `INT64` index | Throws `std::invalid_argument` | A total order, stated below |

The zero-element rule is decided at record time, before either path runs, so both modes answer
the same way. `var` and `std` are the population forms, dividing by `n`.

`argmax` and `argmin` return the lowest index of the winning value under the total order of
`Reduce::RunningArgMax`: a NaN at index 0 wins, a NaN at any other index loses to every element
that is not a NaN, and ties go to the lower index. That is the answer of a serial scan that starts
from element 0 and replaces its best only on a strictly greater value.

**Shapes.** A whole-tensor fold is `Reduce::reduce`: a grid-strided pass per work-item, a
hand-written halving tree per work-group, then one task that adds a fixed number of partials in
ascending order. The partials live in a scratch leased at record time. `sum`, `mean` and `asum`
use `CompensatedSum` (Neumaier), `nrm2` uses `ScaledSumOfSquares` so that squaring does not
overflow or underflow, and a HALF input accumulates in FLOAT32. `var`, `std` and `logsumexp` are
two fixed-order stages through `CentredReductionDispatch`: a centre (the mean, or the maximum),
then a compensated sum around it. An axis reduction is `Reduce::reduce_axis` over an `AxisSpan`,
which names no axis: `axis_span_for` resolves the axis, the layout and the strides on the host.
`mean_spatial` is the one shape with a pair of segment axes. An axis fold stages no partials and
leases no scratch.

Each executor takes a dispatcher with two members, the SYCL fold and `reference`, the serial host
fold. `execute_folding_reduction<Accum>` builds one from a transform and a finalize functor, and
is the seam a consumer uses for a fold of its own.

`LogicOps::all` and `any` are reductions too, and the one place a `sycl::reduction` remains: a
minimum and a maximum over ones and zeros, whose value no combination order can change. On an
empty tensor `all` writes 1 and `any` writes 0.

### Random

`kernels/sycl/random/philox.hpp` is a Philox4x32-10 counter-based generator. Every distribution
is a pure function of `(seed, offset, block index)`, so a fill is reproducible however the work
was scheduled. `RandomOps` has `constant`, `uniform`, `normal`, `log_normal`, `exponential`,
`poisson`, `discrete_uniform`, `truncated_normal`, `bernoulli`, the Xavier and He initialisers,
`orthogonal` and `shuffle`. A distribution that draws values refuses HALF and the integer dtypes
at record time; `constant` and `shuffle` only move values. `poisson` draws in double precision
for every dtype and throws `UnsupportedExecution` on a device without `fp64`, under every mode.

The words Philox produces are the same on the host and on the device. A distribution that goes
on through `log`, `sqrt`, `cos`, `exp` or `erfc` uses the math library of whichever side runs it,
so the two modes are not expected to agree to the bit there.

### `convert`

`ElementwiseOps::convert(src, dst)` is the one operation whose operands may differ in dtype. It
serves the nine pairs of HALF, FLOAT32 and FLOAT64, and a pair outside them throws at record time
naming both dtypes. Converting a dtype to itself copies.

---

## 8. The SushiRuntime dependency (D3)

SushiBLAS does not vendor SushiRuntime. `cmake/Runtime.cmake` resolves it in three steps, the
pattern the whole stack uses:

1. **A target guard.** If a superproject already defines `SushiRuntime::SushiRuntime`, that
   target is used.
2. **`find_package(SushiRuntime QUIET)`.** An installed package.
3. **A sibling `add_subdirectory`.** `../sushiruntime` beside this checkout, or
   `$SUSHISTACK_HOME/sushiruntime` inside a workspace.

Whichever step succeeds, every consumer links the same namespaced target,
`SushiRuntime::SushiRuntime`. SushiBLAS inherits SushiRuntime's
`sushiruntime_deterministic_fp` contract through that link.

---

## 9. Quarantined surface (`include/SushiBLAS/experimental/`)

`LinalgOps` (SVD, QR, eigendecomposition) and `TransformsOps` (FFT, DFT) are declared under
`include/SushiBLAS/experimental/` and have no implementation. The umbrella header does not
include them and no view reaches them. This is decision D1.

`engine/blas/sparse.hpp` still declares an empty `SparseBLAS` class from before the sparse family
existed. No header includes it, and the sparse family is `SparseOps` in `engine/sparse.hpp`.

---

## 10. Packaging

SushiBLAS is found with `find_package`: `SushiBLAS::sushiblas` is a `STATIC` library, aliased
in-tree and exported under the same name, with `install(EXPORT)`, `export(EXPORT)` and a
generated `SushiBLASConfig.cmake` (`SameMinorVersion` compatibility before 1.0). The configure
step checks `include/SushiBLAS/version.hpp` against the CMake project version. `tests/package/`
is an out-of-tree consumer; see [the integration guide](../guides/INTEGRATION.md).

Because `sushiblas` is a `STATIC` library, its build-policy targets (`sushiblas_warnings`,
`sushiblas_perf`, `sushiblas_sanitizers`, composed by `sushiblas_apply_build_policy()` in
`cmake/SushiBLASBuildPolicy.cmake`) put flag values on the real target through
`target_compile_options`, `target_compile_definitions`, `target_link_options` and
`$<TARGET_PROPERTY:...>`, never through `target_link_libraries`. For a static library,
`target_link_libraries(... PRIVATE policyTarget)` records the policy target's name in
`INTERFACE_LINK_LIBRARIES` as `$<LINK_ONLY:...>`, which breaks the export with "requires target X
that is not in any export set". Flag values record no target name.

The CMake modules that define SushiBLAS-specific functions (`SushiBLASBuildPolicy.cmake`,
`SushiBLASCompilerWarnings.cmake`, `SushiBLASOptimization.cmake`, `SushiBLASSanitizers.cmake`)
carry the program's full name. `Vcpkg.cmake`, `Toolchain.cmake` and `SyclTarget.cmake` do not,
because their logic gives the same result whichever sibling repository's copy loads first.
[CONTRIBUTING.md, section 4](../CONTRIBUTING.md#build-system-names) states the naming rule.

---

## 11. Exceptions

A record call throws before any task reaches the graph, so the failure names the line that asked
for the operation. Four types cover the library:

| Type | Base | Thrown when |
| --- | --- | --- |
| `std::invalid_argument` | `std::logic_error` | An operand, flag, scalar or parameter is wrong: missing storage, a dtype, rank, shape, layout, stride or length mismatch, a dtype outside the operation's domain, a flag outside its enumeration, a complex scalar for a real tensor, a bad window geometry |
| `SushiBLAS::UnsupportedExecution` | `std::invalid_argument` | The operands are right and the operation cannot run under this mode or on this device: a device-only operand under `ForceReference`, FLOAT64 or COMPLEX64 storage on a device without `fp64`, HALF storage on a device without `fp16`, `poisson` on a device without `fp64` |
| `std::runtime_error` | `std::exception` | Creation and I/O: a `Tensor` or `Storage` rule (rank above 6, a negative dimension, an overflowing size, a failed allocation, a view past its storage), `TensorFactory::create`, `SparseBuilder` and `Sparse::assemble_csr`, a scratch allocation, a file that cannot be read or written |
| `SushiRuntime::Core::SushiException` | `std::runtime_error` | The runtime refused a graph or a run |

`UnsupportedExecution` (`core/errors.hpp`) derives from `std::invalid_argument`, so a handler
written for the base still catches it, and a caller that wants the two apart lists it first:

```cpp
try
{
    SushiBLAS::BLASOps(engine).gemm(A, B, C);
}
catch (const SushiBLAS::UnsupportedExecution& refusal)
{
    // The operands are right; this mode or this device cannot run the operation.
    std::printf("%s\n", refusal.what());
}
catch (const std::invalid_argument& rejected)
{
    // An operand, a flag or a scalar is wrong.
    std::printf("%s\n", rejected.what());
}
```

Where each type comes from:

- The operand checks of `engine/operand_check.hpp` (`require_storage`, `require_dtype`,
  `require_same_shape`, `require_same_layout`, `require_same_element_count`,
  `require_min_elements`, `require_contiguous`, `require_rank`, `require_scalar`,
  `require_extent`, `require_operand`) throw `std::invalid_argument` with a message of the form
  `<op>: operand '<name>' <rule>`. `require_device_dtype` throws `UnsupportedExecution`.
- `Dispatch::require_domain` and `set_task_params` throw `std::invalid_argument`.
- `TaskRecorder::require_host_reachable`, which `record_pair` calls, throws
  `UnsupportedExecution`.
- `SB_THROW_IF` (`core/logger.hpp`) throws `std::runtime_error`. It ends every dtype switch, so a
  dtype that slipped past the record-time check cannot fall through to nothing.
- `IO` throws `std::invalid_argument` for an operand rule (a strided tensor, a dtype or byte-count
  mismatch) and `std::runtime_error` for a file.

Every helper logs at error level before it throws.

`Engine::execute` can throw what the graph compiler throws (`std::runtime_error` for a cycle),
whatever the first failing task threw, a `SushiException` from the scheduler, and a
`sycl::exception` for a device error. Two SYCL kernels throw `std::runtime_error` at execute
time: the tiled SYRK and SYR2K, on a device that cannot host their local tile. An `IO` call that
reads or writes a tensor runs `execute()` first and can throw any of these. `SB_ASSERT` is not an
exception: it logs and aborts in a build without `NDEBUG`.

## Project layout

See the [root README](../../README.md#project-layout) for the directory tree.

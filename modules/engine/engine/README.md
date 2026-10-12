# Engine headers

`modules/engine/engine/include/SushiBLAS/engine` declares the operation views and the machinery they record through.
Each operation family is a module of its own under `modules/operations/`, with its README. The device policy, the launch and reduction shapes, the serial host loops and the window
geometry are support headers; see [the support module](../../foundation/support/README.md).

| File | Declares |
| --- | --- |
| `task_recorder.hpp` | `TaskRecorder`, the one caller of `SushiRuntime::Recorder::task` in SushiBLAS |
| `task_handle.hpp` | `TaskHandle`, the value a record call returns for the task it recorded |
| `task_description.hpp` | `TaskDescription`, what an operation states about the task it records |
| `operand_set.hpp` | `OperandSet`, the tensors one task reads and writes, and the storage it retains |
| `execution_mode.hpp` | `ExecutionMode`, which names the kernel a recorded task runs |
| `kernel_pair_dispatcher.hpp` | `KernelPairDispatcher`, which runs one kernel of a reference and SYCL pair |
| `task.hpp` | `execute_typed_task`, the recorder of one dtype-dispatched task that runs a kernel pair |
| `dispatch.hpp` | What every dispatch family shares: the element domain and the dtype switch |
| `operand_check.hpp` | The record-time checks an operation runs on its tensor operands |
| `scratch.hpp` | `ScratchBuffer`, the lease on one pooled device scratch block |
| `scratch_pool.hpp` | `ScratchPool`, the owner of the device scratch blocks an `Engine` lends |
| `scratch_shelf.hpp` | The shelf of idle device scratch blocks and the block it lends |
| `rng_stream.hpp` | `RNGStream`, the seed and counter offset random operations draw from |
| `blas.hpp` | `BLASOps`, the view that records every dense BLAS operation |
| `sparse.hpp` | `SparseOps`, the view that records the sparse BLAS operations |
| `spatial.hpp` | `SpatialOps`, the windowed and strided movement operations over an NHWC grid |

The other five views are declared one folder down: `ElementwiseOps`, `NonLinearOps`, `RandomOps`
and `ReductionOps` in `math/`, `LogicOps` in `logic/`.

## How an operation is recorded

Every view holds a `TaskRecorder&` and nothing else of the engine; `RandomOps` also holds the
`RNGStream&`. A view is built from an `Engine&` or from the recorder itself. An operation checks
its operands, declares them in an `OperandSet`, and hands the set, a `TaskDescription` and a task
body to the recorder, directly or through a seam (`execute_typed_task`, `execute_blas_task`, an
executor of `Dispatch`). It returns the `TaskHandle` of the recorded task. Nothing runs until
`Engine::execute()`. An operation that has nothing to do, such as one over a tensor without
elements, records no task and returns a default `TaskHandle`, for which `is_valid()` is false.

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `dispatch.hpp`

The shared vocabulary of SushiBLAS's generic op-dispatch seam.

`SushiBLAS::Dispatch` is the low-level, functor-templated layer every op in this library is built
on: one `execute_*` helper per op family that records a task on the engine's graph, switches on the
tensor's runtime dtype, and inlines a caller-supplied functor into a single SYCL kernel. The five
family headers are:

`<SushiBLAS/elementwise/dispatch.hpp>` `<SushiBLAS/nonlinear/dispatch.hpp>`
`<SushiBLAS/reductions/dispatch.hpp>` `<SushiBLAS/random/dispatch.hpp>`
`<SushiBLAS/logic/dispatch.hpp>`

These used to be private `*_internal.hpp` headers under source/, uninstalled and therefore unreachable
from outside this tree. They are public now because they are the seam a layer above (an autodiff/NN
framework) needs in order to fuse: composing several math ops into one functor and handing it to
`execute_binary` produces ONE kernel launch instead of one per op, and it does so without the
consumer writing a line of SYCL: the functor is a plain C++ lambda. Keeping that capability private
would have forced every consumer to either eat a kernel launch per elementwise op or reimplement
this dispatch layer itself.

This header carries only what all five families share; include the family header for the executors
themselves.

### `Domain`

Not a runtime filter: it selects, at compile time, which `case` arms of the dtype switch actually
instantiate the caller's functor. That distinction matters because the functor is a generic lambda
whose body is only well-formed for some element types: `sycl::max(x, y)` and `sycl::floor(x)`, for
instance, reject `std::complex`, and a generic lambda gives no SFINAE-visible signal of that
(return-type deduction instantiates the body, which is not the immediate context). So the caller
states the domain, the dispatch instantiates only the arms inside it, and every arm outside it
throws naming the op and the dtype.

**A bit set over element-type families, not a list of named combinations.** It began as the
two-valued `{Real, RealAndComplex}` this codebase still spells everywhere, and the arrival of
`INT32`/`INT64` would have turned that into six enumerators whose relationships only the reader can
see. As flags, a new family costs one bit and the query stays `domain_admits`; the two original
names survive as the combinations they always were, so no existing call site changes.

Warning: `Real` is the default deliberately: a functor written for real scalars always compiles
under it. Widen it only when the functor is well-formed for the family being added: see the `safe_*`
helpers in `<SushiBLAS/support/safe_math.hpp>` for the complex case, and note that an integer
functor must not use `sycl::` floating-point builtins.

### `domain_admits`

The one query every dispatch arm asks, always in an `if constexpr`, so an arm outside the domain is
never instantiated, only compiled as a throw.

### `domain_of`

The runtime companion to `domain_admits`, for the few places that must reject a whole family before
the switch rather than inside it, a shape or dtype-agreement check that runs at record time, where
the compile-time domain is known but the arm has not been chosen yet.

### `require_domain`

Throws `std::invalid_argument` through `Detail::throw_dtype_error`. `with_element_type` would
reject it too, but not until the task runs, which is a worse place to find
out: by then the graph is recorded, the caller is inside `Engine::execute()`, and the failure is
attributed to a submission rather than to the line that queued it. This is the same question asked
at record time, where a caller can still act on the answer.

### `ElementTag`

The C++17 way to say "here is a type" to a callable. A C++20 templated lambda `[]<typename T>(){}`
takes its type as an explicit template argument, which is a spelling C++17 has no lambda syntax for;
a plain generic lambda takes an *argument* and reads the type back off it, which is what this empty
tag exists to be:

```cpp
[&](auto tag) { using T = typename decltype(tag)::type; ... }
```

### `with_element_type`

Runs `arm` instantiated for the C++ type `dtype` denotes, or throws.

The dtype switch, once. Every per-element executor in this library used to write its own: seven
`case` arms each, four of them differing only in a type name, two of them wrapped in an `if
constexpr` domain guard, and one `default` carrying the failure message. Eight copies of that is
eight places to forget an arm when a `DataType` is added, and eight slightly different wordings when
one is rejected. This is the whole thing in one function: the caller supplies a generic callable and
says nothing about dtypes at all.

An arm outside `Dom` is never instantiated: the `if constexpr` sees to that, which is what lets a
functor that is only well-formed for real scalars be dispatched on a tensor whose dtype *could* have
been complex.

**Two ways to write the arm, both supported.** The tag form is what this library writes and what a
C++17 caller can spell; the explicit-template-argument form is what a C++20 caller's templated
lambda offers, and it kept working when the library moved to C++17 rather than being taken away from
consumers already using it. `Detail::invoke_arm` picks whichever the arm provides.

```cpp
// C++17 and later - a generic lambda over an ElementTag:
Dispatch::with_element_type<Dom>(dtype, "Elementwise operation", name,
    [&](auto tag)
    {
        using T = typename decltype(tag)::type;
        Launch::map(h, q, n, ok, op, (T*)out, (const T*)in);
    });

// C++20 - a templated lambda, still accepted:
Dispatch::with_element_type<Dom>(dtype, "Elementwise operation", name,
    [&]<typename T>() { Launch::map(h, q, n, ok, op, (T*)out, (const T*)in); });
```

- `family`: What kind of operation this is, for the failure message ("Elementwise operation", "Logic
  operation", ...).
- `name`: The op's own name, for the failure message.
- `arm`: Callable either as `arm(ElementTag<T>{})` or as `arm.template operator()<T>()`, for every
  element type in `Dom`.

Throws `std::runtime_error` naming `family`, `name` and `dtype` when the dtype falls outside `Dom`,
or is not an element type at all. An executor calls `require_domain` first, at record time, so a
caller sees `std::invalid_argument` there and this throw is not reached.

### `Detail::arm_takes_tag`

Warning: Only ask this for a `T` the caller's domain admits. Detecting the call deduces the arm's
return type, which instantiates its body, and a body that rejects `T` fails *there*, outside the
immediate context, so it is a hard error rather than a false answer. That is the same property
`SushiBLAS::Dispatch::Domain` exists to work around, and it holds here because `run_arm_if_admitted`
asks only inside its `if constexpr`.

### `Detail::invoke_arm`

At `static_assert(arm_takes_explicit_type<Arm, T>::value,`: stated as a sentence rather than left to
the call expression that follows the assert, whose failure is a page of overload-resolution noise
pointing at `dispatch.hpp` instead of at the arm the caller actually wrote.

### `Detail::run_arm_if_admitted`

Out of line from the switch in `with_element_type`, and not inlined back into it, because an `if
constexpr` written directly inside a `switch` in a template makes this project's clang lose the
switch: instantiating it reports "'case' statement not in switch statement" for every label after
the first. Hoisting the compile-time test into its own function leaves the switch a plain switch of
plain calls, which it parses correctly.

### The unserved-dtype throw in `with_element_type`

At `SB_THROW_IF(!served, "{} '{}' does not support dtype {}.",`: not served means a dtype outside
Dom, or a DataType holding a value outside the enumeration at all. Both are "this op cannot do
that", and both say so by name: a silent no-op here is how seven operations came to do nothing at
all.

### `set_task_params`

`TaskDescription::set_parameter` ignores an out-of-range index in a build with `NDEBUG`, so a caller
that writes past `K_MAX_PARAMETERS` loses the value with no diagnostic at all. This wrapper throws
`std::invalid_argument` at record time instead (P1: fail loudly, never silently).

## `operand_check.hpp`

The record-time checks an operation runs on its tensor operands before it records a task. Each
function checks one rule and throws with a message that names the operation, so the failure
points at the call that queued the operation instead of at a kernel that read past a view. They
live in `SushiBLAS::Dispatch` beside the dtype checks of `dispatch.hpp`.

| Function | Rule | Throws |
| --- | --- | --- |
| `require_storage` | A tensor with `num_elements > 0` has storage. | `std::invalid_argument` |
| `require_dtype` | The tensor holds the expected dtype. | `std::invalid_argument` |
| `require_same_shape` | Two tensors have the same rank and the same size in every dimension. | `std::invalid_argument` |
| `require_same_layout` | Two tensors of rank 2 or more have the same layout. | `std::invalid_argument` |
| `require_same_element_count` | Two tensors hold the same number of elements. | `std::invalid_argument` |
| `require_min_elements` | The tensor holds at least the stated number of elements. | `std::invalid_argument` |
| `require_contiguous` | The strides are the dense strides `Tensor::layout_strides` gives for the shape under the tensor's layout. | `std::invalid_argument` |
| `require_rank` | The tensor has the stated rank. | `std::invalid_argument` |
| `require_scalar` | The tensor holds exactly one element. | `std::invalid_argument` |
| `require_extent` | A named extent has the expected value. | `std::invalid_argument` |
| `require_operand` | A rule the caller states and has already evaluated holds. | `std::invalid_argument` |
| `require_device_dtype` | The tensor's device can compute in its dtype: fp64 for FLOAT64 and COMPLEX64, fp16 for HALF. | `UnsupportedExecution` |

### `require_contiguous`

An operation whose kernel indexes its operand as a flat array cannot honour a stride, and must
reject a view it would read wrongly. The check accepts a storage offset, because the kernel starts
at `data()`, and skips the stride of a size-1 dimension, because that stride is never multiplied
by a non-zero index. It rejects a zero or negative stride in any other dimension, a transposed
view, and a slice that narrows an inner dimension. A slice along the outermost dimension keeps
dense strides and passes. An empty tensor passes, since no element is read.

### `Detail::throw_operand_error`, `throw_execution_error`, `throw_dtype_error`

Each logs its message at error level and throws it, so every check reports in the same shape.
`throw_operand_error` throws `std::invalid_argument` as `<op>: operand '<name>' <rule>`.
`throw_execution_error` throws `UnsupportedExecution` as `<op>: <refusal>`; it takes no operand
name, because the recorder's refusal has none to give: an `OperandSet` knows that one of its
tensors is device-only, not which. `throw_dtype_error` throws `std::invalid_argument` naming the
family, the operation and the dtype it does not serve. `core/README.md` says how
`UnsupportedExecution` relates to `std::invalid_argument`.

## `execution_mode.hpp`

`ExecutionMode` has three enumerators. `Auto` lets the library choose, and it chooses the SYCL
kernel. `ForceSYCL` runs the SYCL kernel. `ForceReference` runs the reference path on the host:
a reference kernel where the operation has one, and a serial loop over the operation's own
functor in the elementwise, non-linear, logic, random and reduction families. The mode is a
property of the recorder, read once per record call and fixed in the task; changing it later does
not change a task already recorded.

## `task_recorder.hpp`

`TaskRecorder` is the only caller of `SushiRuntime::Recorder::task` in SushiBLAS. It is bound to
the recorder of one `SushiRuntime::Graph` and to one `ScratchPool`, and holds the execution mode.
`Engine` owns one; a test or a layer above may build its own over a graph it owns.

### `TaskRecorder::record_pair`

Takes a `TaskDescription`, an `OperandSet`, the family name for diagnostics, and a body callable as
`(const Internal::KernelPairDispatcher&, sycl::queue&, const std::vector<sycl::event>&) ->
sycl::event`. It first calls `require_host_reachable`. It then wraps the body in a
`Detail::RecordedPairTask` with a dispatcher that holds the current mode and with the set's
`StorageKeepAlive`, records the task with the set's read and write regions and with the name, the
operation and the parameters as `SushiRuntime::NodeOptions`, and returns a `TaskHandle` of its own
identity and the task's sequence number. If the runtime's recorder refuses the recording,
`record_pair` returns the handle of no task, and the next run returns the refusal.

### `TaskRecorder::require_host_reachable`

Under `ForceReference`, throws `UnsupportedExecution` when `OperandSet::has_device_only_operand()`
is true, that is, when a declared tensor's storage has `Residency::Device`. The host reference
path would otherwise dereference device-only USM. Under any other mode it returns at once. The
message is `<name>: <family> cannot run under ForceReference: an operand is device-only USM,
which the host reference kernel cannot dereference`. It is public for the caller that must refuse
before a side effect: `execute_random` calls it before it reserves a stream offset, and the
reduction executors before they lease scratch.

### `Detail::RecordedPairTask`

The callable stored in the graph. It holds the body, the dispatcher and one reference to each
distinct storage the task declared, so an operand's allocation stays alive until the graph drops
the task even if the caller's `Tensor` is destroyed first.

## `task_handle.hpp`

`TaskHandle` names one task a recorder recorded, or none. It holds the recorder's identity, a
process-wide counter value that is never 0, and the count of tasks that recorder had recorded
before this one. A default handle has identity 0 and `is_valid()` false. Two handles are equal when
both fields are. The handle exposes no event and no way to wait: the work finishes in
`Engine::execute()`.

## `task_description.hpp`

`TaskDescription` is what an operation states about one task besides its operands: the `name` it
reports under, the `SushiRuntime::OperationID` in `operation`, and `K_MAX_PARAMETERS` (eight) slots
of 64 bits in `parameters`. `set_parameter<T>(index, value)` copies the bytes of a trivially
copyable value of at most 64 bits into one slot and `parameter<T>(index)` reads them back; a slot
never written holds 0. An index outside the slots is an `SB_ASSERT`, and a build with `NDEBUG`
ignores the write and reads `T{}`. `TaskRecorder` hands the slots to the runtime as
`NodeOptions::parameters`, and a `static_assert` beside that call holds `K_MAX_PARAMETERS` at or
below `SushiRuntime::NodeOptions::K_MAX_PARAMETERS`.

The type belongs to SushiBLAS and names nothing of the runtime but `OperationID`.
`TaskRecorder` turns it into `SushiRuntime::NodeOptions` when it records: the name, the
operation, and all eight slots as the node's parameters. The runtime copies the name and the
slots, so `name` has to outlive the record call only as far as the runtime is concerned; a seam
that keeps the pointer for its own diagnostics, as `execute_typed_task` does, needs it for the
life of the task, and every operation passes a string literal.

## `operand_set.hpp`

`OperandSet` is what an operation declares about one task. `reads`, `writes` and `updates` take a
tensor, append its `Tensor::region()` to the read list, the write list or both, and retain its
storage; a tensor without storage adds nothing. `scratch` declares a `ScratchBuffer` as written
and retains its block's storage. `merge` adds another set's declarations, which is how a GEMM
epilogue's operands join the task's. The recorder reads the two region lists for the dependency
tracker, `retained()` for the keep-alive, and `has_device_only_operand()` for the refusal under
`ForceReference`.

### `Internal::StorageKeepAlive`

Holds one reference per distinct `Storage`. The first six sit in an inline array and a seventh
spills to a `std::vector`, so the common task allocates nothing for it. Retaining a storage twice
holds it once.

## `kernel_pair_dispatcher.hpp`

`Internal::KernelPairDispatcher` holds the execution mode the recorder had when the task was
recorded. A task body calls `dispatcher.run(q, deps, reference_fn, sycl_fn)`. Under
`ForceReference` it waits on the dependencies, calls `reference_fn` on the calling thread and
returns an empty event; under `Auto` and `ForceSYCL` it returns what `sycl_fn(q, deps)` returns.
It is the one place the two branches are written.

## `rng_stream.hpp`

`RNGStream` holds one counter-based random stream: a seed and an offset counter, both guarded by
one `std::mutex`. The engine owns one by value and hands it out through `Engine::rng()`. A
`RandomOps` view holds a reference to a stream, the engine's or one the caller built, and passes
it to `Dispatch::execute_random` in `<SushiBLAS/random/dispatch.hpp>`, which reserves
the offset of each task. No random operation reaches the engine.

- `reseed(seed)` stores the seed and rewinds the offset to zero under the lock, so one seed
  followed by one call sequence always draws the same numbers. It is what
  `RandomOps::set_seed` runs.
- `seed()` returns the current seed.
- `reserve(count)` returns an `RNGPosition` holding the seed and the current offset, then adds
  `count` to the offset, all under the lock. Every random task reserves one offset, so tasks
  recorded from different threads get distinct `(seed, offset)` pairs, and a `reseed` racing a
  reservation can never pair one seed with the other seed's offset.

The lock is taken once per recorded random operation, never inside a kernel.

## `scratch.hpp`

`ScratchBuffer` leases one device scratch block from a `ScratchPool`. The constructor takes the
pool, a byte count and the device index of the operand's storage, so the scratch sits on the
device the work was planned for. A size of 0 leases nothing and `data()` is null. Copies share
the block through a handle count; a task body captures the buffer by value, and the block goes
back to the shelf when the last copy is destroyed, which is when the graph drops the task.
`bytes()` is what the caller asked for, not the size of the block. `region()` is the write region
a task declares for the scratch, and `OperandSet::scratch` declares it and retains the block's
storage. The reduction executors in `engine/math/reductions/dispatch.hpp` and
`Internal::allocate_blas_scratch` in `operands/scratch_lease.hpp` lease through it; none of them
builds a `Storage` for scratch itself.

## `scratch_pool.hpp`

`ScratchPool` owns the shelf through a shared pointer, and a lent block holds the shelf too, so
the shelf outlives the pool for as long as a lease is out. `idle_block_count()` reports how many
blocks sit idle and `trim()` frees them; neither touches a block out on lease. `Engine` owns one
pool and its recorder hands it to operations through `TaskRecorder::scratch_pool()`.

## `scratch_shelf.hpp`

`Detail::ScratchShelf` keeps idle blocks per device queue and per size class. There are six
classes, 128 bytes to 4096 bytes in powers of two, and a request is rounded up to the smallest
class that holds it. `take` hands out an idle block of that class or allocates a new
`Residency::Device` storage. A request above 4096 bytes is allocated at its own size and freed
when it comes back. `give_back` keeps at most 64 idle blocks per class per device and frees the
rest. The idle lists are reserved to that ceiling when the shelf is built, so returning a block
never allocates. One `std::mutex` guards the lists; it is taken at record time and when a task
is destroyed, never inside a kernel.

## `task.hpp`

This is `blas/blas_task.hpp`'s `execute_blas_task` with the one thing that was BLAS-specific
about it, the word "BLAS" in its diagnostics, lifted into a parameter. `execute_blas_task` still
exists, still says "BLAS operation", and forwards here.

The move happened because a second domain arrived. `spatial.im2col` reported as a "BLAS operation"
when its dtype is refused sends a reader to the wrong half of the library, and the alternative, a
second copy of the seven-arm dtype switch, is seven arms to forget when a `DataType` is added.

The header includes the recorder, the operand set and the dispatcher, and not
`<SushiBLAS/engine/engine.hpp>`: the seam takes a `TaskRecorder&` and never sees an `Engine`.

### `execute_typed_task`

Calls `Dispatch::require_domain<Dom>` and then `TaskRecorder::record_pair` with a
`Detail::TypedTaskBody`: one `switch (dtype)` that calls the operation's closure with a value of
the element type. An unsupported dtype fails at record time (P1 corollary), never silently
no-ops. Its arguments are the recorder, the `TaskDescription`, the `OperandSet`, the dtype, the
family and `op_func`, which is callable as `(auto scalar_type, const KernelPairDispatcher&,
sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`. The BLAS, sparse and spatial
operations record through it, so none of them reads the execution mode: the dispatcher the body
receives already holds it.

- `Dom`: Which element types to instantiate `op_func` for, in exactly the sense `Dispatch::Domain`
  defines for the math and logic families. It defaults to `Domain::RealAndComplex` because that
  is what every built-in BLAS *arithmetic* op wants; the parameter exists for callers whose functor
  is written outside this library and is therefore no more likely to be well-formed for
  `std::complex` than an elementwise lambda is, and for operations that pass another domain:
  `window_dot` passes `Domain::Real`.
- `family`: What kind of operation this is, for diagnostics: "BLAS operation", "Spatial operation".
  A string literal, because the task body keeps the pointer.

Returns: The `TaskHandle` of the recorded task; the work is deferred until `Engine::execute()`.

Throws `std::invalid_argument` at record time if `dtype` falls outside `Dom`, and
`UnsupportedExecution` at record time as `TaskRecorder::require_host_reachable` describes.

At `switch (dtype)`: spelled out rather than routed through Dispatch::with_element_type, which is
what the elementwise, non-linear and logic families use. `op_func` takes a *value* of the element
type alongside the queue and the dependency list and returns the submitted event, which is not the
one-argument, void-returning arm that helper calls. Wrapping it to fit with a nested generic lambda
crashes this project's clang outright under -fdelayed-template-parsing (a null-dereference during
instantiation of the HALF arm; the device pass also fails with "SmallVector unable to grow"). The
body is therefore a class, `Detail::TypedTaskBody`, whose `operator()` holds the switch, not a
lambda. Seven arms, the same seven as with_element_type and the same domain rule on the complex
and integer arms, kept in step by hand until that compiler bug is gone.

The `SB_THROW_IF(true, ...)` that ends the dtype switch is unreachable in practice, because
`require_domain` rejected the dtype at record time. A switch that can fall through to nothing is how
a silent no-op gets in, so it ends in a throw rather than in a default return.

`Detail::TypedTaskBody::operator()` throws `std::runtime_error` from that last statement.

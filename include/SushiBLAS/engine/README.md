# Engine headers

`include/SushiBLAS/engine` declares the operation families the `Engine` exposes and the shared
machinery under them: the dispatch vocabulary (`dispatch.hpp`), the operand checks
(`operand_check.hpp`) and the task recorder (`task.hpp`). The sub-folders `blas/`, `logic/`,
`math/` and `spatial/` each have a README of their own. The device policy, the launch and
reduction shapes and the window geometry are support headers; see
[../support/README.md](../support/README.md).

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `dispatch.hpp`

The shared vocabulary of SushiBLAS's generic op-dispatch seam.

`SushiBLAS::Dispatch` is the low-level, functor-templated layer every op in this library is built
on: one `execute_*` helper per op family that records a task on the engine's graph, switches on the
tensor's runtime dtype, and inlines a caller-supplied functor into a single SYCL kernel. The five
family headers are:

`<SushiBLAS/engine/math/elementwise/dispatch.hpp>` `<SushiBLAS/engine/math/nonlinear/dispatch.hpp>`
`<SushiBLAS/engine/math/reductions/dispatch.hpp>` `<SushiBLAS/engine/math/random/dispatch.hpp>`
`<SushiBLAS/engine/logic/dispatch.hpp>`

These used to be private `*_internal.hpp` headers under src/, uninstalled and therefore unreachable
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
two-valued `{REAL, REAL_AND_COMPLEX}` this codebase still spells everywhere, and the arrival of
`INT32`/`INT64` would have turned that into six enumerators whose relationships only the reader can
see. As flags, a new family costs one bit and the query stays `domain_admits`; the two original
names survive as the combinations they always were, so no existing call site changes.

Warning: REAL is the default deliberately: a functor written for real scalars always compiles under
it. Widen it only when the functor is well-formed for the family being added: see the `safe_*`
helpers in `<SushiBLAS/engine/math/safe_math.hpp>` for the complex case, and note that an integer
functor must not use `sycl::` floating-point builtins.

### `domain_admits`

The one query every dispatch arm asks, always in an `if constexpr`, so an arm outside the domain is
never instantiated, only compiled as a throw.

### `domain_of`

The runtime companion to `domain_admits`, for the few places that must reject a whole family before
the switch rather than inside it, a shape or dtype-agreement check that runs at record time, where
the compile-time domain is known but the arm has not been chosen yet.

### `require_domain`

`with_element_type` would reject it too, but not until the task runs, which is a worse place to find
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
or is not an element type at all.

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

`TaskMetadata::set_param` silently ignores an out-of-range index, so a caller that writes past
`K_MAX_PARAMS` loses the value with no diagnostic at all. This wrapper turns that into a loud
failure at record time (P1: fail loudly, never silently).

## `operand_check.hpp`

The record-time checks an operation runs on its tensor operands before it records a task. Each
function checks one rule and throws `std::invalid_argument` with a message of the form
`<op>: operand '<name>' <rule>`, so the failure points at the call that queued the operation
instead of at a kernel that read past a view. They live in `SushiBLAS::Dispatch` beside the dtype
checks of `dispatch.hpp`.

| Function | Rule |
| --- | --- |
| `require_storage` | A tensor with `num_elements > 0` has storage. |
| `require_dtype` | The tensor holds the expected dtype. |
| `require_same_shape` | Two tensors have the same rank and the same size in every dimension. |
| `require_same_element_count` | Two tensors hold the same number of elements. |
| `require_min_elements` | The tensor holds at least the stated number of elements. |
| `require_contiguous` | The strides are the dense strides `Tensor::layout_strides` gives for the shape under the tensor's layout. |

### `require_contiguous`

An operation whose kernel indexes its operand as a flat array cannot honour a stride, and must
reject a view it would read wrongly. The check accepts a storage offset, because the kernel starts
at `data()`, and skips the stride of a size-1 dimension, because that stride is never multiplied
by a non-zero index. It rejects a zero or negative stride in any other dimension, a transposed
view, and a slice that narrows an inner dimension. A slice along the outermost dimension keeps
dense strides and passes. An empty tensor passes, since no element is read.

### `Detail::throw_operand_error`

Logs the message at error level and throws it, so every check reports in the same shape.

## `rng_stream.hpp`

`RngStream` holds the counter-based random stream of one `Engine`: a seed and an offset counter,
both guarded by one `std::mutex`. The engine owns one by value and hands it out through
`Engine::rng()`. Random operations never call it directly; they go through `Dispatch::reseed` and
`Dispatch::reserve_offsets` in `<SushiBLAS/engine/math/random/dispatch.hpp>`, so an operation
source still knows the engine only through `get_graph()`.

- `reseed(seed)` stores the seed and rewinds the offset to zero under the lock, so one seed
  followed by one call sequence always draws the same numbers. It is what
  `engine.random().set_seed(...)` runs.
- `seed()` returns the current seed.
- `reserve(count)` returns an `RngPosition` holding the seed and the current offset, then adds
  `count` to the offset, all under the lock. Every random task reserves one offset, so tasks
  recorded from different threads get distinct `(seed, offset)` pairs, and a `reseed` racing a
  reservation can never pair one seed with the other seed's offset.

The lock is taken once per recorded random operation, never inside a kernel.

## `scratch.hpp`

`ScratchBuffer` owns a device-resident `Storage` allocation for the partial results of one
operation, through a shared `sushi_ptr`. A task lambda captures the handle by value, so the
allocation lives until the graph drops the task. The constructor takes the device index of the
operand's storage, so the scratch sits on the device the partials were planned for. A size of 0
allocates nothing and `data()` is null. `region()` is the write region the task declares for the
scratch. The reduction seams in `engine/math/reductions/dispatch.hpp` and
`Internal::allocate_level1_scratch` use it; none of them builds a `Storage` for scratch itself.

## `spatial.hpp`

The word "convolution" does not appear in this domain and should not be added to it. `im2col` is a
gather, `pool_max` is a windowed reduction, `channel_affine` is a broadcast elementwise. Parameter
ownership, training mode and differentiation belong to the layer above; what belongs here is moving
numbers around a grid.

Every operation in this domain has both a pure-C++ reference kernel and a SYCL one, and the
reference is the specification. The two are held to each other by this domain's unit tests, which
run `ExecutionMode::FORCE_SYCL` and `ExecutionMode::FORCE_REFERENCE` into two tensors and compare
them. `ExecutionMode::VERIFY` is not implemented, and `Engine::set_mode` refuses it.

**Every backward here is a gather, never an atomic scatter.** SushiBLAS links
`sushiruntime_deterministic_fp` PUBLIC, and a floating-point atomic add has a combination order that
varies between runs, so `col2im` written as "each column entry atomically adds into its pixel" would
silently break the library's bit-reproducibility guarantee. Each output element gathers its own
contributions in a fixed order instead. This is the same argument
`docs/architecture/ARCHITECTURE.md` gives for rejecting `sycl::reduction` in the folding reductions,
and `tests/functional/regression/test_spatial_determinism.cpp` is what keeps it true.

### `SpatialOps::im2col` with stated extents

`[N, H, W, C]` to `[N * H_o * W_o, R * S * C]`, with the patch index running `(r, s, c)`: channel
innermost. Out-of-bounds taps are zero; padding is a parameter and never a materialised copy.

**This is the overload to call when the padding is asymmetric.** `Spatial::ConvGeometry` carries
`pad_top` and `pad_left` only (see `Spatial::output_extent` for why), so the trailing padding that
decides `H_o` and `W_o` is not recoverable from the geometry alone. A caller that has both extents,
which SushiAI's shape inference always does, states them here rather than letting the two-operand
form assume symmetry.

### `SpatialOps::im2col` with derived extents

Convenience over the explicit-extent overload, for the symmetric padding every convolution in the
ResNet-18 and YOLOv3-tiny bodies uses: `W_o` follows from `Spatial::output_extent` with `pad_left`
on both sides, and `H_o` from dividing the column matrix's row count by it. Asymmetric padding must
use the explicit overload, which is why that one is documented first.

### `SpatialOps::col2im`

`im2col`'s transpose, and its gradient: entry `(row, patch)` of `cols` is added into the pixel that
`im2col` read it from, and entries whose tap fell outside the image are dropped, which is the same
crop a materialised zero-padding would have needed.

Implemented as a gather over output pixels rather than a scatter over column entries, so overlapping
windows need no floating-point atomic add. The opening of the `spatial.hpp` section says why that is
a hard constraint rather than a preference.

Every pixel of `image` is written, including pixels no window read; those receive an explicit zero,
so the destination needs no prior clear.

### `SpatialOps::pool_max`

The running maximum starts at the lowest finite value of the element type, never at zero: padding is
the absence of an element, not a zero-valued candidate, and a zero-initialised maximum silently
turns every all-negative border window into 0. Ties keep the first element in scan order, which is
PyTorch's convention and what makes the recorded index single-valued.

`indices` must be INT64 and shaped like `out`. It holds the **logical** flat position `((n*H + h)*W
+ w)*C + c` of each window's maximum, which stays comparable across differently-strided views of the
same image; `pool_max_backward` needs it to route the gradient without a scatter.

### `SpatialOps::pool_max_backward`

A gather over the windows that could have selected each input pixel, testing the recorded index
rather than scattering through it, so overlapping windows need no floating-point atomic add. A pixel
selected by several windows accumulates all of their gradients in a fixed order.

`g` must be the geometry the forward used; the index buffer alone does not determine the window set.

### `SpatialOps::pool_avg`

The divisor is `kernel_h * kernel_w` and counts padded taps that contributed nothing: PyTorch's
`count_include_pad=True`, which is its default and therefore what a network ported from it expects.
The alternative, dividing by the in-bounds count, gives different numbers at every border pixel of
every padded pool.

No index buffer: unlike `pool_max`, this operation's backward needs nothing the geometry does not
already say.

### `SpatialOps::pool_avg_backward`

Each pixel gathers `dy / area` from every window that covered it. Kept separate from
`pool_max_backward` rather than folded behind a mode flag: the two share a loop shape and nothing
else, since one selects through a recorded index and this one broadcasts a constant.

### `SpatialOps::upsample_nearest`

`out[n, h_o, w_o, c] = in[n, h_o / scale_h, w_o / scale_w, c]`.

Carries no `Spatial::ConvGeometry:` this operation has two numbers, not eight, and handing it a
geometry it does not read would be six fields a reader has to rule out before trusting the call. It
builds a `Spatial::WindowSpan` for the strides alone.

### `SpatialOps::upsample_nearest_backward`

Each input pixel gathers the fixed `scale_h * scale_w` output elements that read it, in ascending
order, so this needs no atomic add and no index buffer.

### `SpatialOps::concat_channels`

The write half of a channel concatenation: call it once per source, each with the offset its
channels begin at. `src` and `dst` must agree on `N`, `H` and `W`.

**Why not a zero-copy view.** A slice's gradient has to be a materialised operand for the next node,
and a graph whose values are sometimes views and sometimes buffers is a distinction the memory
planner would have to carry everywhere. The copy is paid once; the complexity would be paid forever.

**A known conservatism.** In NHWC a channel range is strided rather than a contiguous byte interval,
so two concatenations into one destination produce overlapping `Tensor::region()`s and
SushiRuntime's dependency tracker serialises them. That is correct and possibly slower than
necessary. Measure before addressing it.

`dst`'s channel count.

### `SpatialOps::slice_channels`

`concat_channels`'s inverse, and its backward. One kernel serves both; two names exist because
`slice_channels(dy, offset, dx)` says what it means and a shared `copy_channels(dy, offset, dx, 0,
count)` does not.

`src`'s channel count.

### `SpatialOps::channel_affine`

`out[n,h,w,c] = in[n,h,w,c] * gamma[c] + beta[c]`. BatchNorm's apply half, and the whole of
BatchNorm in evaluation mode.

### `SpatialOps::broadcast_spatial`

`out[n,h,w,c] = in[n,c] * scale`. Global average pooling's backward, with `scale = 1/(H*W)`.

A **per-`(n, c)`** broadcast, which is neither of the two the library already has: `GEMM_BIAS`
broadcasts a row vector down every row, and `channel_affine` broadcasts one value per channel across
the whole batch.

### `SpatialOps::layout_convert`

The boundary transpose between NCHW and NHWC. **Its direction is not a flag**: it reads both
tensors' strides and copies by logical `(n, c, h, w)` position, so NCHW-to-NHWC and NHWC-to-NCHW are
the same call with the operands' strides swapped.

This buys interoperability, NCHW data and NCHW checkpoints in, and is not an NCHW implementation.
Two O(N*C*H*W) passes per step are immaterial beside the convolutions they bracket, which is exactly
the trade PyTorch's `channels_last` makes.

## `task.hpp`

This is `engine/blas/utils.hpp`'s `execute_blas_task` with the one thing that was BLAS-specific
about it, the word "BLAS" in its diagnostics, lifted into a parameter. `execute_blas_task` still
exists, still says "BLAS operation", and forwards to the mode-aware overload here.

The move happened because a second domain arrived. `spatial.im2col` reported as a "BLAS operation"
when its dtype is refused sends a reader to the wrong half of the library, and the alternative, a
second copy of the seven-arm dtype switch, is seven arms to forget when a `DataType` is added.

The include list is `blas/utils.hpp`'s, minus what only the vector helper needed.
`<SushiBLAS/engine.hpp>` is included outright rather than forward-declared: both templates in
`task.hpp` need `Engine` complete (they call `get_graph()`, `get_mode()` and `get_context()`), and a forward declaration that only works because every includer happens to have
pulled `engine.hpp` first is a trap rather than a decoupling.

### `execute_typed_task`

One `add_task` whose body is a `Detail::TypedTaskBody`: one `switch (dtype)` that calls the
operation's closure with a value of the element type. An unsupported dtype fails loudly at record
time (P1 corollary), never silently no-ops. There are two overloads.

The first takes `op_func` as `(auto scalar_type, sycl::queue&, const std::vector<sycl::event>&) ->
sycl::event`. It knows nothing of the execution mode. No operation in `src/engine` uses it now;
`tests/functional/unit/test_dispatch_seam.cpp` still records through it.

The second takes a `ReferenceOperands` list before `op_func`, and `op_func` takes a
`const KernelPairDispatcher&` after the scalar. At record time it reads `Engine::get_mode()` once,
runs `require_host_reachable` on the listed pointers, and fixes the mode inside the dispatcher the
body receives. Every BLAS and spatial operation records through it, so none of them calls
`get_mode()`, which the layering rule "an operation knows the engine only through `get_graph()`"
forbids.

- `Dom`: Which element types to instantiate `op_func` for, in exactly the sense `Dispatch::Domain`
  defines for the math and logic families. It defaults to REAL_AND_COMPLEX because that is what
  every built-in BLAS *arithmetic* op wants; the parameter exists for callers whose functor is
  written outside this library and is therefore no more likely to be well-formed for `std::complex`
  than an elementwise lambda is, and for movement ops like `copy` and the whole spatial family,
  which relocate elements without doing arithmetic to them and so pass `Domain::ALL`.
- `family`: What kind of operation this is, for diagnostics: "BLAS operation", "Spatial operation".
- `host_operands` (second overload): every pointer the reference kernel reads or writes. A null
  entry, an operand without storage, is skipped.

Returns: An empty sycl::event; the work is deferred until `Engine::execute()`.

Throws `std::runtime_error` at record time if `dtype` falls outside `Dom`, and the second overload
throws `std::invalid_argument` at record time as `require_host_reachable` describes.

At `switch (dtype)`: spelled out rather than routed through Dispatch::with_element_type, which is
what the elementwise, non-linear and logic families use. `op_func` takes a *value* of the element
type alongside the queue and the dependency list and returns the submitted event, which is not the
one-argument, void-returning arm that helper calls. Wrapping it to fit with a nested generic lambda
crashes this project's clang outright under -fdelayed-template-parsing (a null-dereference during
instantiation of the HALF arm; the device pass also fails with "SmallVector unable to grow"). The
body is therefore a class with a member template `call<T>`, not a lambda. Seven arms, the same seven
and the same domain rule as with_element_type, kept in step by hand until that compiler bug is gone.

The `SB_THROW_IF(true, ...)` that ends the dtype switch is unreachable in practice, because
`require_domain` rejected the dtype at record time. A switch that can fall through to nothing is how
a silent no-op gets in, so it ends in a throw rather than in a default return.

### `dispatch_with_mode`

The one implementation of the FORCE_REFERENCE / SYCL branch. Under FORCE_REFERENCE it waits on the
dependencies and runs the reference kernel on the host; under AUTO and FORCE_SYCL it submits the
SYCL kernel. It never sees VERIFY, which `Engine::set_mode` refuses; a mode it does not handle
throws `std::logic_error`. Its only caller is `KernelPairDispatcher::run`.

### `KernelPairDispatcher`

Holds the execution mode `execute_typed_task` read at record time. A task body calls
`dispatcher.run(q, deps, reference_fn, sycl_fn)`, which forwards to `dispatch_with_mode`. Changing
the engine's mode after an operation is recorded does not change how that operation runs.

### `ReferenceOperands`

The list of pointers an operation's reference kernel dereferences on the host, built in the
operation's recording step from the `data_or_null()` values it already holds:
`Internal::ReferenceOperands{read_A, read_B, rw_C}`.

### `require_host_reachable`

Under FORCE_REFERENCE, asks `sycl::get_pointer_type` about each listed pointer in the context of
every device queue of the engine's runtime context. If one reports `usm::alloc::device`, it throws
`std::invalid_argument` naming the family and the operation, before the task reaches the graph.
Without this check the reference kernel dereferenced device-only USM on the host. Under any other
mode it returns at once. A GEMM epilogue's declared operands are regions, not pointers, so they are
not checked.

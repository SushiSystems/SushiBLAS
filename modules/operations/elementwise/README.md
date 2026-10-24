# elementwise

`ElementwiseOps`, the per-element math operations, and the executors that record them.

It depends on `core`, `support`, `tensor`, `engine` and `functor_map`. Its headers are included as
`<SushiBLAS/elementwise/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `dispatch.hpp` | The elementwise executors of the dispatch seam |
| `elementwise.hpp` | ElementwiseOps, the per-element math operations on tensors |
| `ordered_sum_element.hpp` | The element functor of `ordered_sum` and the type its weights take |
| `wrapping_math.hpp` | The arithmetic functors that wrap modulo 2^N on integer element types |

`source/` holds 33 translation units.

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to.

## `elementwise.hpp`

### `ElementwiseOps::convert`

The one operation in this library whose two operands are *allowed* to disagree on dtype, and it
exists because before it there was no way to produce an fp16 view of an fp32 tensor at all:
`Level1::copy` throws on a dtype mismatch, and every other op in the library dispatches on exactly
one `Core::DataType` shared by all its operands. A mixed-precision training step needs exactly the
missing thing: fp32 master weights cast down to the fp16 the matrix multiply runs in, and fp16
gradients cast back up before they reach an optimiser that must not accumulate in fp16.

**Why it lives here and not beside `copy`.** `Level1` is the BLAS level-1 surface, and reference
BLAS has no converting copy at any level: every routine there is fixed to a single precision, which
is why `?copy` carries one type letter and not two. Putting a routine that is not in the standard
on `BLASOps` would cost that view the only thing it means. What is left is the family
this op already belongs to by shape: `dst[i]` is a function of `src[i]` and of nothing else, which
is the sentence `ElementwiseOps` opens with. The only thing separating it from every other
elementwise op is that its input and output element types differ, and that is a property of the
dispatch it needs rather than of what it computes.

**The dtype cross product is bounded to REAL x REAL: nine arms.** This is the real design
constraint. A converting op dispatches on *two* runtime dtypes, so the naive spelling,
`with_element_type` nested inside `with_element_type`, is 7 x 7 = 49 kernel instantiations for one
operation, more than this library's entire elementwise family costs put together. It is also a shape
this tree's clang is known to miscompile: the nested generic lambda is exactly what crashes it
during instantiation, which is why `Internal::execute_typed_task` spells its own switch out by
hand (`<SushiBLAS/engine/task.hpp>`). So the pair is restricted to `Dispatch::Domain::Real` on
**both** sides, HALF, FLOAT32, FLOAT64, nine arms, and the switch is written as two plain, un-nested
`switch` statements over ordinary function templates rather than as lambdas.

REAL x REAL is not an arbitrary trim. It is what mixed precision consists of, and every pair outside
it is either meaningless or a bug wearing a cast: a complex operand has no single real value to
narrow to, and a float-to-int conversion is a *rounding policy* decision (truncate, round-half-even,
saturate?) that this layer has no basis to make on the caller's behalf. Both therefore throw at
record time, **naming the dtype**, rather than silently doing nothing: the defect this project has
called out by name more than once. An integer conversion, if one is ever wanted, belongs in its own
op whose name states the rounding it performs.

**Converting from a dtype to itself is legal and copies.** The alternative, making `convert(f32,
f32)` an error, would push a branch back into every caller, and the caller least able to take it is
the one this op was written for: a mixed-precision policy holds its compute dtype as runtime data,
and when that dtype happens to equal the master dtype the op degenerates naturally into the copy the
policy meant. It is not a silent no-op either: the task is still recorded and the elements are still
written, so the dependency edges a caller gets are the same whatever the two dtypes are, which is
the property that actually has to hold for a graph to be correct.

**What it declares to the dependency tracker.** `src.region()` as a read and `dst.region()` as a
write, taken from `Tensor::region` and never from a bare pointer. A `BufferRegion` converts
implicitly from a pointer and a bare pointer widens to the *whole* allocation, so two views of one
arena become two unrelated resources, `overlaps` answers false and the tracker emits no edge at all,
not a wrong edge, none. That mistake has produced a silently wrong training loss in this stack twice
(the `reductions/dispatch.hpp` section of this page records one of them), and it is exactly the
arena-packed fp32/fp16 pair this op exists to serve.

- `src`: The tensor to read. Must be contiguous and have storage.
- `dst`: The tensor to overwrite, in its own dtype. Must be contiguous, have storage, have the same
  shape and, at rank 2 or more, the same layout as `src`, and not overlap it: each work-item reads one element and writes another of a
  different width, so a partial overlap is a race no launch shape can fix.

Returns: The `TaskHandle` of the recorded task; the work is deferred until the caller runs the graph.

Throws `std::invalid_argument`, naming the offending operand, if it has elements but no storage
or is non-contiguous, if the two shapes or layouts disagree, if its dtype falls outside `Domain::Real`,
and if the operands overlap. Throws `UnsupportedExecution` as `Dispatch::require_device_dtype` and
`TaskRecorder::record_pair` state.

### `ElementwiseOps::ordered_sum`

`ordered_sum(inputs, weights, out)` writes `out = w0 * x0 + w1 * x1 + ...` in one kernel. Each
work-item reads input 0 to input n - 1 for its own element, so the order is fixed by the index and
never by arrival. The reference path and the SYCL path run the same functor, so they match bit for
bit, and a test compares both against a serial sum written with the same order and type.

| Element type | Accumulates in | Rounds |
| --- | --- | --- |
| `Half` | `Float32` | Once, after the last input |
| `Float32`, `Float64` | Its own type | Each multiply and each add |
| `Complex32`, `Complex64` | The component type | Each component apart |
| `Int32`, `Int64` | Its own type, wrapping | Never; adds exactly modulo 2^32 or 2^64 |

- A complex element is multiplied by the real weight one component at a time
  (`re += w * x.real()`, `im += w * x.imag()`), so no complex product and no NaN handling of one
  comes into the rounding. The functor turns floating-point contraction off, so a multiply and an
  add never fuse into one fma on one path and not on the other.
- An integer sum refuses a weight other than 1 when it is recorded. It adds through
  `wrapping_add`, which converts to the unsigned type of the same width.
- The signature takes `SushiRuntime::span<const Tensor>` and `SushiRuntime::span<const double>`
  (the C++17 span the library already uses for shapes), and the output last, as `add` takes it.
  A weight is a `double` and is narrowed once, on the host when the call is recorded, to the type
  `Detail::OrderedSumWeight` names (`ordered_sum_element.hpp`): `Float32` for `Half`, `Float32`,
  `Complex32`, `Float64` for `Float64` and `Complex64`. The kernel holds no `double` for the other
  types, so a device without `aspect::fp64` runs them. The rounding is the one the kernel did before
  each multiply, so no result moves.
- The kernel carries the input addresses and the weights by value, so a call takes at most
  `ElementwiseOps::K_MAX_ORDERED_SUM_INPUTS` (32) inputs.
- `out` matches the inputs in shape, layout and type, and overlaps none of them.

Throws `std::invalid_argument`, naming the operand, if `inputs` is empty or longer than the limit,
if `weights` holds another count, if an operand has no storage or is not contiguous, if a shape,
a layout or a type differs from `inputs[0]`, if `out` overlaps an input, or if an integer sum has a
weight other than 1.

### `ElementwiseOps::clamp`

`min_val > max_val` throws `std::invalid_argument` when the call is recorded.

## `elementwise/dispatch.hpp`

### Empty operands

An operand with zero elements is legal. Every executor in this header, in `nonlinear/dispatch.hpp`
and in `logic/dispatch.hpp` runs its shape, layout, dtype and domain checks, then returns a default
`TaskHandle` without recording a task when the output holds no elements. A tensor with zero
elements needs no storage, and the executor never calls `data()` on it. `ElementwiseOps::convert`
follows the same rule.

### Integer operands

`add`, `sub`, `mul`, `neg` and `abs` accept every dtype (`Domain::All`); `min`, `max` and `clamp`
accept the real and integer dtypes (`Domain::RealAndIntegral`); on a floating dtype `min` and
`max` return NaN when either operand is NaN, as the reductions do. Division, `fmod`, `remainder`,
`pow` and the transcendental ops refuse the integer dtypes. Integer arithmetic goes through
`elementwise/wrapping_math.hpp`: each functor converts to the unsigned type of the same width,
operates there and converts back, so an overflow wraps modulo 2^32 or 2^64 instead of being
undefined behaviour. `abs` of the most negative integer returns that integer.

### `execute_unary_inplace`

The generic seam behind every `ElementwiseOps` unary entry point, and usable directly by a
downstream consumer that wants several elementwise steps collapsed into a single kernel: compose
them into one functor and call this once.

Warning: This is a low-level generic seam, not a convenience wrapper. `op_func` is inlined into a
SYCL device kernel, so it must be device-copyable and trivially copyable, must capture by value
only, must not dereference host pointers or touch host-only state, must not allocate, and must not
throw: none of which the compiler will always diagnose for you. Element type conversions are the
functor's job: whatever it returns is assigned straight back into the tensor's storage. A tensor
the functor reads goes through `bind_reads`, not through a captured pointer. A `params` list longer
than the task metadata's inline capacity throws `std::invalid_argument`.

### `execute_binary`

The generic seam behind every `ElementwiseOps` binary entry point. A downstream consumer fusing an
expression tree into one functor and calling this once gets a single SYCL kernel for the whole
expression.

Warning: Same device-kernel constraints as `execute_unary_inplace:` `op_func` is inlined into the
kernel, so it must be device-copyable, capture by value, hold no host pointers, allocate nothing and
throw nothing.

### `execute_zip_inplace`

The variadic sibling of `execute_unary_inplace` and `execute_binary`, and the seam a consumer
reaches for when the thing it wants to run is not "one output from some inputs" but "a group of
tensors advanced together". `op_func` receives an **lvalue reference** to each tensor's element, so
it may read and write any of them in a single pass.

That is what turns a whole optimiser step into one kernel. An AdamW update issues roughly eleven
launches per parameter when it is spelled as separate ops, decay the parameter, scale each moment,
accumulate each moment, two bias corrections, the epsilon, the division, the step, and every one of
them re-reads and rewrites state that is already in registers. Written through this seam it is one
launch and one pass over memory:

```cpp
// theta, g, m, v and ema advanced together, in one kernel.
Dispatch::execute_zip_inplace(recorder, std::array{&theta, &g, &m, &v, &ema},
    "optim.adamw", "optim.adamw"_op,
    [b1, b2, lr, wd, eps, inv_bc1, inv_bc2, decay]
    (auto& theta_i, auto& g_i, auto& m_i, auto& v_i, auto& ema_i)
    {
        using T = std::decay_t<decltype(theta_i)>;
        m_i = T(b1) * m_i + T(1 - b1) * g_i;
        v_i = T(b2) * v_i + T(1 - b2) * (g_i * g_i);
        const T step = (m_i * T(inv_bc1))
                     / (Dispatch::safe_sqrt(v_i * T(inv_bc2)) + T(eps));
        theta_i = theta_i * T(1 - lr * wd) - T(lr) * step;
        ema_i   = T(decay) * ema_i + T(1 - decay) * theta_i;
    });
```

**Why `std::array<Tensor*, N>` and not a pack or a span.** A pack would have to be the trailing
parameter, which would push `name`, `op_id`, `op_func` and the defaulted `params` in front of the
tensors and break the argument order every other executor in this family uses. A `std::span` would
make N a run-time value, and the kernel needs it at compile time: it stages one register per operand
per element and expands the call to `op_func` through an `index_sequence`. The array also says the
true thing about these operands, that they are one homogeneous group, not a list of separately-named
roles, and `std::array{&a, &b, &c, &d, &e}` stays legible at the N=5 the example above needs.
Pointers rather than references only because a reference cannot be an array element.

**Graph bookkeeping.** All N tensors are declared as both reads and writes, since the functor is
free to touch any of them and this layer cannot see which. The regions come from `Tensor::region()`,
which reports a precise byte interval, so two disjoint views of one arena still do not serialise
against each other.

- `Dom`: Which element types to instantiate the functor for. Defaults to Domain::Real; pass
  Domain::RealAndComplex only when `op_func` is well-formed for `std::complex` (see
  safe_math.hpp).
- `tensors`: The tensors to advance together, in the order `op_func` expects them. None may be null;
  all must share one shape, layout and dtype.
- `op_func`: Callable `(auto&...) -> void` taking exactly N lvalue references, invoked once per
  element index, or a `bind_scalars` or `bind_reads` bundle of one.

Returns: The `TaskHandle` of the recorded task, or a default one when the tensors hold no
elements; the work is deferred until the caller runs the graph.

Warning: Same device-kernel constraints as `execute_binary:` `op_func` is inlined into the kernel,
so it must be device-copyable, capture by value, hold no host pointers, allocate nothing and throw
nothing.

Warning: The N tensors must not overlap one another. Each element index is staged into registers,
handed to `op_func`, and written back, so a write through one operand is not visible to a second
operand that happens to name the same address. Disjoint views of one arena are fine; two views of
the same bytes are not.

Throws `std::invalid_argument` if a tensor is null, has elements but no storage or is non-contiguous,
if its shape, layout or dtype differs from operand 0's (the message names which one), if
the dtype falls outside `Dom`, or if `params` exceeds the metadata's inline capacity.

`execute_zip_inplace` declares every operand as both a read and a write: the functor may touch any
of them, and nothing at this layer can tell which ones it actually did.

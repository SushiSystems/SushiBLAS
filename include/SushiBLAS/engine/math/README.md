# Math headers

`include/SushiBLAS/engine/math` declares the elementwise, non-linear, random and reduction operation
views, the executor of each family under `<family>/dispatch.hpp`, and the scalar helpers in
`safe_math.hpp`.

| File | Declares |
| --- | --- |
| `elementwise.hpp` | `ElementwiseOps`, the per-element math operations |
| `nonlinear.hpp` | `NonLinearOps`, the activation functions and their backward passes |
| `random.hpp` | `RandomOps`, the random fills and initialisers |
| `reductions.hpp` | `ReductionOps`, the whole-tensor and per-axis reductions |
| `safe_math.hpp` | Scalar math helpers that stay well-formed for real and complex element types |
| `elementwise/dispatch.hpp` | `execute_unary_inplace`, `execute_binary`, `execute_zip_inplace` |
| `elementwise/map_task.hpp` | `MapTask` and `ZipTask`, the task bodies of the per-element seams |
| `elementwise/scalar.hpp` | `ElementScalar`, a double parameter narrowed once to every element type |
| `elementwise/scalar_binding.hpp` | `bind_scalars`, which narrows a functor's scalar parameters on the host |
| `elementwise/read_binding.hpp` | `bind_reads`, which declares the tensors a functor reads at execution time |
| `elementwise/wrapping_math.hpp` | The arithmetic functors that wrap modulo 2^N on integer element types |
| `nonlinear/dispatch.hpp` | `execute_nonlinear_forward`, `execute_nonlinear_backward` |
| `random/dispatch.hpp` | `execute_random`, `RandomFill`, `Sampling` and `require_random_dtype` |
| `reductions/dispatch.hpp` | The reduction executors and the span builders they share |
| `reductions/arg_reduction.hpp` | `ArgReductionDispatch`, the dispatcher of `argmax` and `argmin` |
| `reductions/centred_reduction.hpp` | `CentredReductionDispatch`, the dispatcher of `var`, `std` and `logsumexp` |

Every executor takes a `TaskRecorder&` as its first argument and returns a `TaskHandle`. A
record-time refusal of an operand, a dtype or a parameter is `std::invalid_argument`; a
device-only tensor under `ForceReference` and a dtype a tensor's device cannot compute in are
`UnsupportedExecution`. Each family has a host path under `ForceReference`: a serial loop over
the same functor the SYCL kernel runs.

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

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
during instantiation, which is why `Internal::execute_blas_task` spells its own switch out by hand
(`<SushiBLAS/engine/blas/utils.hpp>`). So the pair is restricted to `Dispatch::Domain::Real` on
**both** sides, HALF, FLOAT32, FLOAT64, nine arms, and the switch is written as two plain, un-nested
`switch` statements over ordinary function templates rather than as lambdas.

REAL x REAL is not an arbitrary trim. It is what mixed precision consists of, and every pair outside
it is either meaningless or a bug wearing a cast: a complex operand has no single real value to
narrow to, and a float-to-int conversion is a *rounding policy* decision (truncate, round-half-even,
saturate?) that this layer has no basis to make on the caller's behalf. Both therefore throw at
record time, **naming both dtypes**, rather than silently doing nothing: the defect this project has
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
write, taken from `Tensor::region` and never from a bare pointer. A `ResourceRegion` converts
implicitly from a pointer and a bare pointer widens to the *whole* allocation, so two views of one
arena become two unrelated resources, `overlaps` answers false and the tracker emits no edge at all,
not a wrong edge, none. That mistake has produced a silently wrong training loss in this stack twice
(`<SushiBLAS/engine/math/reductions/dispatch.hpp>` records one of them), and it is exactly the
arena-packed fp32/fp16 pair this op exists to serve.

- `src`: The tensor to read. Must be contiguous and have storage.
- `dst`: The tensor to overwrite, in its own dtype. Must be contiguous, have storage, have the same
  shape and, at rank 2 or more, the same layout as `src`, and not overlap it: each work-item reads one element and writes another of a
  different width, so a partial overlap is a race no launch shape can fix.

Returns: The `TaskHandle` of the recorded task; the work is deferred until `Engine::execute()`.

Throws `std::invalid_argument`, naming the offending operand, if it is storage-less or
non-contiguous, if the two shapes or layouts disagree, if its dtype falls outside `Domain::Real`,
and if the operands overlap. Throws `UnsupportedExecution` as `Dispatch::require_device_dtype` and
`TaskRecorder::record_pair` state.

### `ElementwiseOps::clamp`

`min_val > max_val` throws `std::invalid_argument` when the call is recorded.

## The per-element seam

The executors in `elementwise/dispatch.hpp`, `nonlinear/dispatch.hpp` and `logic/dispatch.hpp`
share one task body. Each checks its operands, declares them in an `OperandSet` and records a
`Detail::MapTask` (one output from some inputs) or a `Detail::ZipTask` (N tensors advanced
together) through `TaskRecorder::record_pair`. The functor the caller hands over is one of three
things:

- a bare callable over elements;
- a `bind_scalars(fn, s0, s1, ...)` bundle, whose scalars the seam narrows to the element type on
  the host and passes after the elements;
- a `bind_reads(fn, read_as_element(t0), read_as<U>(t1), ...)` bundle, whose tensors the seam
  checks, declares on the task and passes as typed pointers after the elements. `fn` may itself
  be a `bind_scalars` bundle, and its scalars then come last.

### `elementwise/map_task.hpp`

`MapTask` and `ZipTask` hold the dtype, the base addresses, the element count and the functor. Under
`ForceReference` the body runs `Serial::map` or `Serial::zip_inplace` on the host; under any other
mode it submits one kernel through `Launch::map` or `Launch::zip_inplace`, with the launch shape
planned from the queue's device. Both paths call the same functor, so the host path is not a second
implementation of the operation. `make_map_task` builds a `MapTask` from the seam's arguments.

### `elementwise/scalar_binding.hpp`

`bind_scalars` returns a `ScalarBound`, which holds each scalar as an `ElementScalar` and stays on
the host. Inside the `with_element_type` arm the seam calls `narrow_for(tag, op)`, which returns a
`NarrowedOp<Fn, T, N>` holding the scalars in the element type `T`; that is what the kernel
captures. A HALF or FLOAT32 kernel therefore carries no double. A plain functor passes through
`narrow_for` unchanged.

### `elementwise/read_binding.hpp`

A functor that dereferences a tensor the seam does not know about, such as a per-channel table,
reads memory the task never declared, and the graph is free to run the task that writes it at the
same time. `bind_reads` closes that: `read_as_element(t)` names a tensor read as an array of the
operation's own element type and `read_as<U>(t)` one read as an array of `U`. The seam then

- checks each read with `ReadBound::require`: it must hold elements, have storage, hold the dtype
  it was declared with and be contiguous, and its device must be able to compute in that dtype;
- declares each read with `ReadBound::declare_into`, which refuses a read that overlaps a region
  the task writes;
- keeps only the base addresses in the task (`AddressedOp`) and hands the kernel a `ReadingOp`
  typed for `T`.

Declaring a read also retains its storage until the task is destroyed, and a device-only read is
refused under `ForceReference` with the operands. A bare functor declares nothing:
`require_declared_reads` and `declare_reads` are no-ops for it.

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
Dispatch::execute_zip_inplace(engine.get_recorder(), std::array{&theta, &g, &m, &v, &ema},
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
elements; the work is deferred until `Engine::execute()`.

Warning: Same device-kernel constraints as `execute_binary:` `op_func` is inlined into the kernel,
so it must be device-copyable, capture by value, hold no host pointers, allocate nothing and throw
nothing.

Warning: The N tensors must not overlap one another. Each element index is staged into registers,
handed to `op_func`, and written back, so a write through one operand is not visible to a second
operand that happens to name the same address. Disjoint views of one arena are fine; two views of
the same bytes are not.

Throws `std::invalid_argument` if a tensor is null, storage-less or non-contiguous, if its shape,
layout or dtype differs from operand 0's (the message names which one), if
the dtype falls outside `Dom`, or if `params` exceeds the metadata's inline capacity.

`execute_zip_inplace` declares every operand as both a read and a write: the functor may touch any
of them, and nothing at this layer can tell which ones it actually did.

## `elementwise/scalar.hpp`

### `ElementScalar`

A scalar parameter of an elementwise or non-linear op (`pow`'s exponent, `clamp`'s bounds, the
`leaky_relu` and `elu` alpha) is a `double` in the public signature. `ElementScalar` narrows it on
the host, once, to `sycl::half`, `float`, `double`, `int32_t` and `int64_t`, and the kernel reads
the member for its element type through `as<T>()`. A HALF or FLOAT32 kernel therefore does no
double arithmetic. The HALF member is the `double` rounded to binary32, then to binary16. Narrowing to an integer type saturates to the type's range, truncates toward
zero and maps NaN to 0. The task metadata records the `double` as given.

## `nonlinear/dispatch.hpp`

### `execute_nonlinear_forward`

The generic seam behind every `NonLinearOps` forward entry point. Like the elementwise executors it
defaults to `Domain::Real`; `sigmoid` and `tanh` opt into `Domain::RealAndComplex`. A consumer
passes `Domain::RealAndComplex` only when its functor is written against the complex-tolerant
`safe_*` helpers.

Warning: This is a low-level generic seam. `op_func` is inlined into a SYCL device kernel: it must
be device-copyable, capture by value only, hold no host pointers, allocate nothing and throw
nothing. A functor instantiated for the complex dtypes is written against
`<SushiBLAS/engine/math/safe_math.hpp>` rather than the raw `sycl::` math builtins, or it fails to
compile for them.

### `execute_nonlinear_backward`

The generic seam behind every `NonLinearOps` backward entry point. The second operand is whatever
the activation's derivative needs, the saved input for GELU and ELU, the saved output for sigmoid
and tanh, which is why it is named generically here and documented per op at the call site.

Warning: Same device-kernel constraints as `execute_nonlinear_forward`.

## `random.hpp`

### `K_MAX_POISSON_LAMBDA`

`RandomOps::poisson` accepts lambda up to 1e7. Below lambda 10 the kernel runs Knuth's
multiplication loop; from 10 up it runs Hoermann's transformed rejection with squeeze (PTRS,
"The transformed rejection method for generating Poisson random variables", Insurance:
Mathematics and Economics 12, 1993). Hoermann states PTRS for lambda >= 10 and puts no upper
limit on it, so the bound comes from the arithmetic:

- The acceptance test compares `k log(lambda) - lgamma(k + 1)` against `lambda` in double. At
  lambda = 1e7 both sides are near 1.5e8, and the double rounding error there is about 3e-8,
  far below the gap the test resolves.
- FLOAT32 holds every integer up to 2^24, about 1.68e7. A draw at lambda = 1e7 lies within 50
  standard deviations (about 1.6e5) of the mean with certainty for practical purposes, so a
  FLOAT32 result still holds the exact count.

Knuth's loop stops when the running product reaches `exp(-lambda)`; below 10 that is at least
4.5e-5, far from underflow.

### `RandomOps::orthogonal`

Draws N(0,1) and orthonormalizes by two modified Gram-Schmidt sweeps, which produce the same Q a
QR decomposition would. Only the shorter axis can be orthonormal, so the result satisfies `Q Q^T =
gain^2 I` when the matrix is wide and `Q^T Q = gain^2 I` when it is tall.

### `RandomOps::shuffle`

A Fisher-Yates shuffle over the flat element sequence, driven by the view's Philox stream: the
output is always a permutation of the input, and replaying it from the same seed reproduces
the same permutation. It accepts HALF, FLOAT32, FLOAT64, COMPLEX32 and COMPLEX64, since it moves
elements rather than computing on them. INT32 and INT64 are refused when the call is recorded.

### `RandomOps::poisson` on a device without fp64

The kernel draws in double precision for every dtype, so `poisson` throws `UnsupportedExecution`
at record time when `t` has elements and its device lacks the fp64 aspect, whatever the dtype of
`t`.

### `RandomOps::truncated_normal`

Bounds that are not finite with `a < b`, and bounds that enclose no probability mass in `t`'s
dtype, throw `std::invalid_argument`.

## `random/dispatch.hpp`

### `K_RNG_SEED_PARAM`

The seed and offset sit in the two highest slots rather than immediately after the distribution's
own parameters so that their position is the same for every random op, whatever its arity: a
profiler or replay tool reading the metadata does not have to know which distribution produced the
task in order to find them.

They used to be written to slots 10 and 11. `TaskMetadata` holds eight, and `set_param` ignores an
out-of-range index without a word, so every random task recorded so far carried a zero seed and a
zero offset in its metadata. The kernels themselves were unaffected (they capture seed/offset by
value, not through the metadata), so the bug never changed a number: it only blanked the record of
which draw a task made.

### `Sampling` and `require_random_dtype`

Every random op names the element families it serves as a `Dispatch::Domain` and says whether it
draws new values or only moves existing ones. `Sampling::DrawsValues` refuses HALF on top of the
domain: HALF has too little mantissa for the inverse-CDF and Box-Muller transforms.
`Sampling::MovesValues` accepts every dtype of the domain; `shuffle` and `constant` use it.

`require_random_dtype` throws `std::invalid_argument` with one message for all of them: the op,
the operand `t`, its dtype and the dtypes the op accepts. The distributions call it through
`execute_random`; `shuffle`, `constant`, `orthogonal` and `truncated_normal` call it directly,
before any other work.

### `submit_random_fill`

Every distribution in this library has that shape: the value at an index is a pure function of the
index, because a counter-based generator is what makes a fill reproducible from `(seed, offset,
index)` regardless of how the work was scheduled.

One work-item computes one Philox block and writes every value the block yields: `draw_block(b)`
returns a `std::array<T, K>` and the fill writes it to `out[b*K .. b*K + K - 1]`, stopping at `n`.
A uniform or normal fill gets four FLOAT32 or two FLOAT64 values per block, so it uses all four
32-bit words of the block. Poisson and `discrete_uniform` return a one-element array, because
their per-element draw needs a variable number of words or all of them.

The launch comes from `SushiBLAS::Launch::indexed` over blocks, so a distribution gets the same
device-dependent shape as an elementwise op without knowing that shape exists.

- `deps`: Events this submission must depend on.
- `draw_block`: Callable `(uint64_t block) -> std::array<T, K>`, invoked once per block on device.

Returns: The submitted work's event.

Warning: `draw_block` is inlined into a SYCL device kernel: it must be device-copyable, capture by
value only, hold no host pointers, allocate nothing and throw nothing.

### `serial_random_fill` and `RandomFill`

`serial_random_fill` is the host path of the same fill: it walks the blocks in order on the
host and writes what `draw_block` returns. `RandomFill` is the launcher a task body
receives. It holds the task's `KernelPairDispatcher`, the queue and the dependencies, and
`fill(n, out, draw_block)` runs `serial_random_fill` under `ForceReference` and
`submit_random_fill` otherwise. A distribution writes `draw_block` once and gets both paths.

### `execute_random`

The generic seam behind every `RandomOps` distribution. It owns the parts every distribution shares,
graph bookkeeping, reserving a fresh counter offset from the `RngStream` it is given so two tasks
never reuse a Philox (key, counter) pair, and the dtype switch, and leaves the actual draw to the
caller's functor.

- `recorder`, `stream`: The recorder the task is recorded through and the stream it draws from.
  `RandomOps` passes its own two references.
- `Dom`: The element families the distribution draws, `Domain::Real` or
  `Domain::RealAndComplex`. The task instantiates `task_func` only for the types `Dom` admits,
  so a distribution that refuses complex carries no complex branch.
- `params`: The distribution's own parameters, recorded in metadata slots 0..n-1.
- `task_func`: Callable `(auto scalar_type, const Dispatch::RandomFill& fill, uint64_t seed,
  uint64_t offset, int64_t size, auto* pT) -> sycl::event`, generic over the runtime dtype via
  `decltype(scalar_type)`. It returns what `fill(...)` returns.

Returns: The `TaskHandle` of the recorded task; the work is deferred until `Engine::execute()`.

Warning: This is a low-level generic seam. The `draw_block` that `task_func` hands to `fill` runs
on the device: it must be device-copyable, capture by value, hold no host pointers, allocate
nothing and throw nothing. Draw through `Kernels::SYCL::Random` rather than a stateful host RNG.

Throws `std::invalid_argument` through `require_random_dtype` for HALF, the integer dtypes and any
dtype outside `Dom`, for a tensor without storage or not contiguous, and if `params` exceeds
`K_MAX_RNG_PARAMS`. Throws `UnsupportedExecution` for a dtype the device of `t` cannot compute in
and for a device-only `t` under `ForceReference`. All of these fire at record time and before the
offset is reserved, so a refused call does not move the stream.

## `reductions.hpp`

### `ReductionOps::max` and `ReductionOps::min`

Both propagate NaN, as NumPy does: one NaN element anywhere makes the result NaN, on every device
and launch shape. They fold through `execute_folding_reduction` with
`Reduce::NaNPropagatingMax` and `Reduce::NaNPropagatingMin` from
`<SushiBLAS/support/reduce_1d.hpp>`. An empty tensor throws `std::invalid_argument` when the call
is recorded, as it does for `mean`, `argmax`, `argmin`, `var`, `std` and `logsumexp`.

### `ReductionOps::argmax`, `argmin`

Each records through `execute_index_reduction` with a `Dispatch::ArgReductionDispatch` over
`Reduce::RunningArgMax` or `RunningArgMin`. The partials are (value, index) pairs, and the lowest
index of the extreme value wins. `reductions/arg_reduction.hpp` holds the dispatcher.

### `ReductionOps::var`, `std`, `logsumexp`

Each records through `execute_reduction` with a `Dispatch::CentredReductionDispatch`, which runs
two fixed-order stages: a first fold that yields a centre (the mean, or the maximum for
`logsumexp`), then a compensated sum of terms taken against it. `reductions/centred_reduction.hpp`
holds the dispatcher and the functors of the three operations.

### `ReductionOps::max_along`

The axis-wise counterpart of `max:` instead of one scalar for the whole tensor, one value per
surviving position. `axis == 1` folds each row to its own maximum, which is the shift a per-row
log-sum-exp needs: shifting a batch by its *global* maximum is exact in real arithmetic and returns
`+inf` in binary32 for any row about 104 below it, because every exponential in that row underflows
and its row sum becomes exactly zero.

Deterministic by construction, like every other folding reduction here; see
`<SushiBLAS/support/reduce_axis.hpp>`.

A NaN in a segment makes that segment's result NaN: the fold is `Reduce::NaNPropagatingMax`.

### `ReductionOps::min_along`

The per-axis minimum, through the same seam with `Reduce::NaNPropagatingMin`.

### `ReductionOps::sum_along`

The axis-wise counterpart of `sum`, and compensated the same way: each segment is folded with
Neumaier's correction in an order fixed by the launch geometry, so the answer is both accurate and
reproducible to the bit.

### `ReductionOps::mean_spatial`

The per-channel, per-sample average a normalisation layer and a global-average-pool both want, in
one launch. It is not `mean` of a reshaped view and it is not `sum_along` of one either: the
surviving position is a *pair* `(n, c)`, so the input offset of a fold is `n * strides[0] + c *
strides[3]` and no single segment stride produces that from one index. Folding both spatial axes as
one run of `H * W` elements is the easy half; the pair is the reason this exists.

Neumaier-compensated and deterministic like every other folding reduction here: the division is the
finalizer, so it costs the same launch as the sum would; see `<SushiBLAS/support/reduce_axis.hpp>`.

### `ReductionOps::count_nonfinite`

The screen a loss-scaling step runs before it applies an update: if the answer is not zero the
gradients overflowed at the current scale, so the step is skipped and the scale halved.
`count_nonfinite(t, r) == 0` is the boolean "all finite"; the count itself is what says how badly,
which is strictly more than the boolean and costs the same fold.

**Both** kinds of non-finite value, and they need separate tests: `x != x` is true only of a NaN,
and an infinity compares equal to itself perfectly happily. Overflow in fp16 produces infinities and
division by an overflowed quantity produces NaNs, so a screen that caught one and not the other
would pass exactly half the runs it exists to stop.

**The result is an INT64 scalar**, and the reason is the consumer rather than tidiness: the caller
reads this value from a *device* kernel with no host synchronisation, because a sync inside the
optimiser step would drain the graph the whole architecture exists to keep full. So the answer has
to be a device-resident value a kernel can load and branch on, which an INT64 tensor element is. It
must also be **exact**, which is what rules out putting the count in a tensor of the input's own
dtype: a count is exact in FLOAT32 only below 2^24 and in HALF only below 2048, and a HALF count
past 65504 is an infinity, so the tensors most worth counting are exactly the ones whose count would
come back wrong. That is the same trap `Dispatch::execute_index_reduction` was cut out of
`execute_reduction` to escape, arriving by a different route.

**Why not `asum`.** It nearly works: `sum(|x|)` is non-finite whenever any element is, so a caller
could have tested `isfinite(asum(g))` and shipped nothing new. It is rejected for one decisive
reason and two supporting ones. The decisive one: `asum` accumulates in the *input's* dtype, so on a
HALF tensor the running total can leave binary16's range all by itself. Eight elements of 20000,
each exactly representable, not one of them remotely bad, give a non-finite total, at which point
the screen reports an overflow that never happened, the scale is halved, and it is halved again on
the next step, for ever, on a run that was fine. The op meant to rescue a mixed-precision run would
be the thing that stalls it.
`Unit_CountNonFiniteTest.AsumReportsOverflowOnAHalfTensorThisOpCallsClean` is that exact tensor put
through both ops, so this is a measurement rather than a worry. The supporting ones: even in fp32 a
loss-scaled gradient is *deliberately* pushed up in magnitude, which is the one situation where a
sum of magnitudes overflows without any element doing so, and `asum` therefore conflates "an element
is bad" with "the total is big"; and its answer is a float in the input's dtype, so a consumer
branching on it does floating-point comparisons where an integer `!= 0` would do. A predicate count
cannot produce a false positive, and that is the property that matters for something whose job is to
decide whether to throw a training step away.

**Deterministic**, like every folding reduction here, and for a stronger reason. The fold is
`Reduce::ExactSum` over integers, which is exactly associative and exactly commutative, so no
combination order can change the answer at all; the fixed-order shape of
`<SushiBLAS/support/reduce_1d.hpp>` is inherited from the shared seam rather than being what makes
this op reproducible.

- `t`: Input tensor; HALF, FLOAT32 or FLOAT64. The integer and complex dtypes throw by name rather
  than answering: an integer element cannot be infinite or NaN, so a confident count of zero would
  tell a caller their tensor had been screened when the question was never asked of it.
- `result`: Scalar output tensor; must be rank-anything with exactly one element and dtype INT64.

Returns: The `TaskHandle` of the recorded task.

Throws `std::invalid_argument` if `result` is not a scalar or not INT64, or if `t`'s dtype is not one
of HALF / FLOAT32 / FLOAT64.

## `reductions/dispatch.hpp`

### `ReductionScratch`

Handed to the dispatcher instead of an allocator, because the buffer is obtained **once, at
task-record time** and captured by the task, not allocated per launch. That is the right trade
specifically because this library is compile-once/replay: a `sycl::malloc_device` plus a `host_task`
to free it would add two submissions to every reduction on every replay of the graph, which is
exactly the launch cost the last two commits existed to remove. A buffer taken when the task is
recorded costs nothing on any subsequent replay.

The buffer is a `ScratchBuffer` leased from the recorder's `ScratchPool` on the device of the
operand. The task holds the lease, so the block is lent for as long as the graph holds the task.
`leased_partials` returns 0 under `ForceReference`, whose host fold stages nothing, so no block
is leased in that mode.

### `execute_reduction`

The generic seam behind every `ReductionOps` entry point. Unlike the elementwise and logic families,
the caller supplies the entire command group rather than a per-element functor: a reduction's shape
is the caller's choice and only the graph bookkeeping, the scratch and the dtype switch are shared.
Callers that want the library's own deterministic shape should reach for `execute_folding_reduction`
instead, which is `execute_reduction` with that shape already filled in.

- `result`: The scalar (single-element) output tensor; must share `t`'s dtype.
- `dispatch`: A dispatcher object with two paths. The SYCL path is `sycl::event
  operator()(sycl::queue& q, const T* in, Out* out, int64_t size, ReductionScratch<P> scratch,
  const std::vector<sycl::event>& deps) const`; the host path is `void reference(const T* in,
  Out* out, int64_t size) const`. A lambda has no `reference` and is refused by a `static_assert`
  in `Detail::ReductionTask`: write a class, as `FoldingReductionDispatch` and
  `CentredReductionDispatch` are.
- `scratch_elements`: How many per-work-group partials the dispatcher wants room for. Zero, the
  default, leases nothing and hands the dispatcher an empty `ReductionScratch`.
  `planned_partials_for(t)` computes it from the device `t` lives on.

Returns: The `TaskHandle` of the recorded task; the work is deferred until `Engine::execute()`.

Warning: This is a low-level generic seam. Whatever the SYCL path submits runs on the device, so
any lambda it nests must be device-copyable, capture by value, hold no host pointers, allocate
nothing and throw nothing. The SYCL path itself runs on the host at execution time and owns the
`depends_on(deps)` call: dropping it silently breaks the graph's ordering guarantees.

Throws `std::invalid_argument` if an operand has no storage, if `t` is not contiguous, if `result`
is not scalar, if the dtypes disagree, or if the dtype is one reductions do not support (HALF,
FLOAT32 and FLOAT64 are; the complex types are not, since a reduction's operators need an ordering
and an identity that `std::complex` does not supply). Throws `UnsupportedExecution` for a dtype a
tensor's device cannot compute in and for a device-only tensor under `ForceReference`; the
refusal comes before the scratch is leased.

### `Detail::ReductionTask`

The task body of the three whole-tensor executors. Under `ForceReference` it calls
`dispatch.reference(in, out, size)` on the host; otherwise it calls the SYCL path with
the leased scratch. `has_reference_path` is the trait its `static_assert` tests.

### `finalize_keeps_width`

The host fold of a sum runs in `double` whatever the element type, and hands `finalize` that
`double`. A `finalize` that returns a narrower type would round the host result differently from
the SYCL one, so `require_finalize_keeps_width` refuses it at compile time. Write `finalize` as
`[](auto value, std::size_t n) { ... }` and return the type it is handed; the seam narrows the
result once, on the store.

At `operands.reads(t).writes(result);`: `Tensor::region()`,
never the bare `data()` pointer. A region is keyed on the *allocation base* and narrowed to the
view's byte interval; a bare pointer converts to the whole-buffer region of whatever it points at.
For a tensor at offset zero those are the same thing, which is why this went unnoticed, but for two
views inside one arena the bare pointers are two different bases, `ResourceRegion::overlaps` returns
false, and the tracker emits no edge at all. Not a wrong edge: none. A reduction then races the op
that produced its input, and the result is a plausible wrong number with no error anywhere.
SushiAI's first arena-packed training run lost exactly this way (a loss of -57 where ln 6 was
expected).

At `operands.scratch(scratch);`: scratch is a declared resource, not a hidden one. It comes from
a pool, and a block goes back to the shelf only when the task that leased it is destroyed, so two
tasks in one graph never hold the same block. It is declared as a write with its own region all
the same, for the reason the operands are.

### `execute_index_reduction`

`argmax` and `argmin` are not the same shape as `sum` or `nrm2` and were being forced through
`execute_reduction` as though they were. That helper insists the result tensor share the input's
dtype and hands the dispatcher a `T*`, so an index had to be stored as `static_cast<T>(i)`: a float.
Below 2^24 elements that is exact and nobody noticed; above it, two adjacent indices round to the
same float and the answer is quietly wrong. In HALF it is quietly wrong above 2048.

Here the input dtype and the result dtype are separate questions: the input may be anything an
ordering is defined on, and the result is an `int64_t` in an INT64 tensor, which is what an index
is.

- `dispatch`: A two-path dispatcher, as `execute_reduction` takes, whose output is `int64_t` and
  whose partials are `Reduce::IndexedValue<T>`: a (value, index) pair, 16 bytes for every element
  type served.
- `scratch_elements`: Per-work-group partials to lease; zero leases nothing.

Returns: The `TaskHandle` of the recorded task; the work is deferred until `Engine::execute()`.

Warning: Same device-kernel constraints as `execute_reduction`.

Throws `std::invalid_argument` if `t` is empty, if `result` is not a scalar, if it is not INT64, or
if the input dtype is one an ordering is not defined on (the complex types). Throws
`UnsupportedExecution` as `execute_reduction` does.

`execute_index_reduction` dispatches over `Domain::RealAndIntegral`: "which element is largest"
needs an ordering, which the complex types do not have and the integer types do.

### `execute_counting_reduction`

The third of this family's three executors, and the two it is not are the clearest way to say what
it is. `execute_reduction` folds a tensor into a scalar of the *same* dtype and leases a scratch
of that dtype's accumulation type; `execute_index_reduction` answers with an `int64_t` index and
stages (value, index) pairs. This one is the remaining corner: an `int64_t` answer that is a
count, with `int64_t` partials, because the question it serves, how many elements satisfy a
predicate, is a fold over the whole tensor and wants the two-stage shape rather than one work-item
walking every element.

They stay three rather than one because merging them means making the result element type a template
parameter of a helper whose entire job is to hide the element type, and every call site would then
spell it. Three short functions that each say what they answer with is the cheaper thing to read.

**Why the count goes in an INT64 tensor.** It is exact for every tensor a `Tensor` can describe, and
it is a value a *device* kernel can consume with no host round trip, which is the requirement, not a
nicety: the caller is an optimiser step that must decide whether to apply an update, and a host sync
there would serialise the training loop against the graph it is trying to keep full. Storing it in
the input's own dtype instead would reintroduce exactly the defect `execute_index_reduction` exists
to have escaped: a count is only exact in fp32 below 2^24 and in fp16 below 2048, and a HALF count
past 65504 comes back as an infinity, so the tensors most worth counting are the ones whose count
would be wrong.

- `dispatch`: A two-path dispatcher, as `execute_reduction` takes, whose output is `int64_t` and
  whose scratch is a `ReductionScratch<int64_t>`.
- `scratch_elements`: How many per-work-group partials the dispatcher wants room for; zero leases
  nothing. `planned_partials_for(t)` is how to compute it.

Returns: The `TaskHandle` of the recorded task; the work is deferred until `Engine::execute()`.

Warning: Same device-kernel constraints as `execute_reduction`.

Throws `std::invalid_argument` if `result` is not a scalar, if it is not INT64, or if the input dtype
is outside `Domain::Real`. Throws `UnsupportedExecution` as `execute_reduction` does.

At `operands.reads(t).writes(result);`: `Tensor::region()`,
never the bare `data()` pointer: the reason is spelled out at length under `execute_reduction`, and
it is the same reason here.

At `with_element_type<Domain::Real>(t.dtype, "Reduction", name,`: REAL only. An integer or complex
tensor is rejected by name rather than answered with the zero that would technically be true of the
integer case: this is a screen for floating-point overflow, so a caller handing it a dtype that
cannot overflow has confused two tensors, and a confident "0 bad elements" is the worst possible
thing to tell them.

The scratch in `execute_counting_reduction`: partials are INT64 whatever the input dtype is, so the
scratch is sized from `sizeof(int64_t)` rather than from the tensor's element width.

### `execute_folding_reduction`

The seam every order-sensitive reduction in the library goes through: `sum`, `mean`, `asum`, `nrm2`,
`prod`, `max` and `min` are each one call to this, differing only in `Accum`, `transform` and
`finalize`. It is `execute_reduction` with a `FoldingReductionDispatch`, whose SYCL path is
`Reduce::reduce<Accum>` and whose host path is `Serial::reduce<Accum>`. That is
the point: the part of a reduction that is hard to get right is the *combination order*, and putting
it in five places is five chances to get it wrong. See `<SushiBLAS/support/reduce_1d.hpp>` for the
shape and for why `sycl::reduction` and `sycl::reduce_over_group` are both unusable here.

It sizes and requests the partials scratch itself, from the device the tensor's storage lives on, so
the caller never sees an allocator.

```cpp
// A root-mean-square reduction a consumer could write, deterministic and
// compensated, in one call and one kernel:
Dispatch::execute_folding_reduction<Reduce::CompensatedSum>(
    engine.get_recorder(), t, result, "nn.rms", "nn.rms"_op,
    [](auto x) { return x * x; },
    [](auto v, std::size_t n) { return sycl::sqrt(v / decltype(v)(n)); });
```

- `Accum`: The accumulation policy: `SushiBLAS::Reduce::CompensatedSum` for anything additive,
  `SushiBLAS::Reduce::RunningProduct` for a product, `SushiBLAS::Reduce::NaNPropagatingMax` or
  `NaNPropagatingMin` for an extreme.
- `result`: The scalar (single-element) output tensor; must share `t`'s dtype.
- `transform`: Callable `(auto x) -> auto`, applied to each element before it is folded in. Pass
  `ReduceIdentityTransform{}` for a plain sum.
- `finalize`: Callable `(auto value, std::size_t n) -> auto`, applied once to the folded result.
  Pass `ReduceIdentityFinalize{}` when there is nothing to do; it is folded into the last kernel
  either way, so `mean` and `nrm2` cost no more launches than `sum`. `value` is `double` on the
  host path of a sum; see `finalize_keeps_width`.

Returns: The `TaskHandle` of the recorded task; the work is deferred until `Engine::execute()`.

Warning: `transform` and `finalize` are inlined into a SYCL device kernel and carry the same
constraints as every other functor in this family: device-copyable, capture by value, no host
pointers, no allocation, no exceptions.

Throws `std::invalid_argument` and `UnsupportedExecution` for the same reasons `execute_reduction`
does. The storage check runs before the partials are sized.

At `const std::size_t partials =`: sized from the device the data lives on, which is the device the
scheduler routes the task to. Should the two ever disagree, Reduce::plan clamps to whatever scratch
actually exists and falls back to the single-pass shape: slower, never wrong.

### `axis_span_for`

The whole of "which axis, which layout, which view" lives here, on the host, once. `Tensor::strides`
already carries the `Core::Layout` the tensor was built under and any non-unit stride a view
inherited, so reducing along either axis of either layout is the same arithmetic with the two
strides swapped, which is why `SushiBLAS::Reduce::AxisSpan` names no axis and the kernel carries no
layout branch.

Public rather than tucked into a detail namespace because it is the reusable half of the seam: an
index-valued axis reduction (`argmax` along an axis) has a different *result dtype* from
`execute_folding_axis_reduction` and so needs its own executor, the way `execute_index_reduction` is
a twin of `execute_reduction`, but it has exactly this geometry, and should take it from here rather
than derive it a second time.

- `t`: The tensor being reduced; must be rank 2.
- `axis`: Which axis is folded away: 0 reduces down the columns (result length `t.shape[1]`), 1
  reduces along the rows (result length `t.shape[0]`).
- `result`: The rank-1 output, one element per surviving position.
- `name`: The op's own name, for the failure messages.

Returns: The span describing every segment.

Throws `std::invalid_argument` if `t` is not rank 2, if `axis` is not 0 or 1, if `result` is not rank
1, or if `result`'s length does not match the surviving extent.

`const int kept = 1 - axis;` in `axis_span_for`: the axis that survives is the one that is not
folded away, and its extent is how many independent folds there are.

### `spatial_span_for`

The spatial mean a normalisation layer needs, and the one shape `axis_span_for`'s five numbers
cannot describe. Folding two axes at once is not the difficulty: `[N*H*W, C]` is a zero-copy rank-2
view and axis 0 of it folds to `[C]` with nothing new. The difficulty is that the surviving position
is a *pair* `(n, c)` whose input offset is `n * strides[0] + c * strides[3]`, and no single
`segment_stride` turns one index into that. So `SushiBLAS::Reduce::AxisSpan` grew a second segment
axis, and `axis_span_for`'s documented rank-2 restriction is lifted exactly this far and no further:
this is a named shape with its own validation, not a general rank-N axis reduction.

H and W are folded as a single run of `H * W` elements at stride `strides[2]`, which is only the
right answer when the rows are contiguous with each other: hence the row-stride check, which rejects
a row-sliced view by name rather than reading past the end of one.

- `t`: The tensor being reduced; must be rank 4, NHWC.
- `result`: The rank-2 `[N, C]` output, one element per surviving pair.
- `name`: The op's own name, for the failure messages.

Returns: The span describing every `(n, c)` fold.

Throws `std::invalid_argument` if `t` is not rank 4, if `result` is not rank 2, if `result`'s shape is
not `[N, C]`, or if `t` is a row-sliced view whose H and W cannot be walked as one run.

### `execute_folding_span_reduction`

The axis-wise counterpart of `execute_folding_reduction`, and the seam every axis reduction in the
library goes through. `max_along`, `min_along` and `sum_along` are each one call to this,
differing only in `Accum`; a per-axis mean is `Reduce::CompensatedSum` with `v / n` as its
finalizer, which re-cuts nothing here. The task body is a `Detail::SpanReductionTask`: the SYCL
path is the segmented launch of `<SushiBLAS/support/reduce_axis.hpp>`, and the host path folds
each segment in index order.

```cpp
// Row-wise maximum of a [rows, cols] tensor into a [rows] vector -
// the shift a numerically honest log-softmax needs.
Dispatch::execute_folding_axis_reduction<Reduce::NaNPropagatingMax>(
    engine.get_recorder(), x, 1, row_max, "math.max_along", "math.max_along"_op,
    Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
```

**It asks for no scratch**, unlike `execute_folding_reduction`, and that is a property of the shape
rather than an omission: a segmented reduction's work-groups never contribute to a shared answer, so
there are no partials to stage between two kernels. See `<SushiBLAS/support/reduce_axis.hpp>` for the
two launch shapes and for the determinism argument, which is the same one `reduce_1d.hpp` makes.

- `Accum`: The accumulation policy: `SushiBLAS::Reduce::NaNPropagatingMax` or
  `NaNPropagatingMin`, `SushiBLAS::Reduce::CompensatedSum` for anything additive,
  `SushiBLAS::Reduce::RunningProduct` for a product.
- `span`: The segment geometry, from `axis_span_for` or `spatial_span_for`. Nothing here re-derives
  it: `execute_folding_span_reduction` takes the layout as given and only owns the graph bookkeeping, the dtype switch
  and the launch.
- `result`: The output tensor; must share `t`'s dtype.
- `transform`: Callable `(auto x) -> auto`, applied to each element before it is folded in.
- `finalize`: Callable `(auto value, std::size_t n) -> auto`, applied once per segment with `n` that
  segment's element count. `value` is `double` on the host path of a sum.

Returns: The `TaskHandle` of the recorded task; the work is deferred until `Engine::execute()`.

Warning: `transform` and `finalize` are inlined into a SYCL device kernel and carry the same
constraints as every other functor in this family: device-copyable, capture by value, no host
pointers, no allocation, no exceptions.

Throws `std::invalid_argument` if `t` or `result` holds elements but has no storage, if the
dtypes disagree, or if the dtype is one reductions do not support
(HALF, FLOAT32 and FLOAT64 are; the complex types have no ordering or identity to fold with, and an
integer result is what the index reductions answer with, not this). Strided operands are
accepted. Throws `UnsupportedExecution` as `execute_reduction` does.

At `operands.reads(t).writes(result);`: `Tensor::region()`,
never the bare `data()` pointer, for exactly the reason spelled out under `execute_reduction`: a
bare pointer widens to the whole-buffer region of whatever it points at, so two views inside one
arena look like two different allocations, overlap() says false, and the tracker emits no edge at
all. That cost a training run once already.

At `const Reduce::AxisPlan geometry =`: planned here, at execution time, from the device the task
was actually routed to: the same reason Reduce::plan is asked inside the kernel launcher rather than
at record time. Nothing is sized from it, so unlike the whole-tensor seam there is no scratch
capacity to clamp against.

`execute_folding_span_reduction` dispatches over `Domain::Real` only, matching `execute_reduction`:
a fold needs an ordering and an identity, which `std::complex` does not supply, and an integer
tensor is not an arithmetic type in this library (see `Dispatch::Domain`).

### `execute_folding_axis_reduction`

`execute_folding_span_reduction` with `axis_span_for` in front of it, which is the split the spatial
fold forced: the geometry a segmented reduction runs on is no longer always derivable from "a tensor
and an axis number", but everything downstream of the span is identical, so the span is the seam and
the axis is one way of building one.

Throws `std::invalid_argument` for the reasons `axis_span_for` and
`execute_folding_span_reduction` list.

Warning: Same device-kernel constraints as `execute_folding_span_reduction`.

## `safe_math.hpp`

Scalar math that stays well-formed for both real and complex element types.

The problem these solve: an op functor handed to a Dispatch executor is a generic lambda,
instantiated once per dtype the executor's Domain admits. A body written as `sycl::exp(x)` is fine
for HALF/FLOAT32/FLOAT64 and a hard compile error for `std::complex`, because the SYCL math builtins
only accept device scalar types. Writing `safe_exp(x)` instead picks `sycl::exp` for real scalars
and the complex overload for complex ones by overload resolution, so one functor covers
Domain::RealAndComplex without any `if constexpr` at the call site.

The complex overloads of `safe_exp`, `safe_tanh`, `safe_log`, `safe_sqrt`, `safe_erf` and
`safe_abs` compute on the real and imaginary parts with `sycl::` real math only. They never call
the `std::` complex functions, because MSVC implements those through host-only helpers
(`_FExp`, `_FSinh`) that a SYCL device cannot call. `safe_exp` and `safe_tanh` return a zero
imaginary part unchanged, sign included, when the input's imaginary part is zero.

Only operations with an unambiguous complex meaning appear here. There is deliberately no `safe_max`
/ `safe_floor` / `safe_clamp`: those need a total order on the element type, complex numbers do not
have one, and inventing a lexicographic stand-in would be a silently-wrong answer rather than a
missing feature. Ops built on them declare Domain::Real and throw for complex inputs.

Lived in src/engine/math/nonlinear/nonlinear_internal.hpp before the dispatch seam went public;
hoisted here so the elementwise and nonlinear families share one definition instead of growing a
second copy each.

### `safe_erf` for a complex value

The complex error function is not elementary and no activation in this library needs it; GELU only
ever reaches `safe_erf` with a real-valued argument.

### `safe_abs` for a complex value

The result is real-valued, but elementwise `abs` is an in-place op writing back into the tensor's
own complex storage, so the magnitude is returned with a zero imaginary part rather than as a bare
scalar.

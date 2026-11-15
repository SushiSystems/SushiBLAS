# random

`RandomOps`, the random fills and initialisers.

It depends on `core`, `support`, `tensor`, `engine` and `random_kernels`. Its headers are included
as `<SushiBLAS/random/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `dispatch.hpp` | The random-fill executor of the dispatch seam |
| `positioned_dispatch.hpp` | The executor of a random fill that reads its stream position when it runs |
| `positioned_random.hpp` | PositionedRandomOps, the random fills whose stream position is an operand |
| `random.hpp` | RandomOps, the random fills and initialisers |
| `rng_position_tensor.hpp` | The position tensor, its checks and `write_rng_position` |

`source/` holds 18 translation units.

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to.

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

## `positioned_random.hpp`, `positioned_dispatch.hpp` and `rng_position_tensor.hpp`

`RandomOps` reserves one stream position when a fill is recorded and captures it by value, so a plan
that is executed again draws the same numbers each time. `PositionedRandomOps::bernoulli(t, p,
position)` records the same fill with the position as an operand: the draw reads `position[0]`, the
seed, and `position[1]`, the offset, inside the kernel at each execution, so one compiled plan draws a
new mask whenever the host writes a new position between executions. This is the shape of a value
that varies between replays, held in memory the kernel reads.

`position` is a contiguous rank-1 INT64 tensor of `K_RNG_POSITION_ELEMENTS` elements in
`Residency::Shared` or `Residency::Host`. `write_rng_position(position, {seed, offset})` fills it on the
calling thread. The caller keeps one rule: no recorded task that reads the position is pending or
running while it writes. A view into a larger allocation is accepted. `position` may not overlap `t`;
the task declares `reads(position).writes(t)`.

At the same `(seed, offset)` the fill writes the same bytes as `RandomOps::bernoulli` recorded at that
stream position, on the reference path and on the device. The metadata slots `K_RNG_SEED_PARAM` and
`K_RNG_OFFSET_PARAM` hold 0, since neither value is known when the call is recorded; `p` is in slot 0.
`t` is FLOAT32 or FLOAT64 as for `RandomOps::bernoulli`. HALF, a position that is not INT64, has
other than two elements or is not rank 1, a device-only position, an overlap and a `p` outside [0, 1]
throw `std::invalid_argument` before anything is recorded.

`Dispatch::execute_random_at` is the generic form: a further distribution is one method and one source
file. Its task body has the shape `(T, const RandomFill&, const std::uint64_t* position, int64_t size,
T*) -> sycl::event`, and its draw reads the position inside the kernel.

The Bernoulli threshold in `positioned_bernoulli.cpp` repeats the four lines of `bernoulli.cpp`; the
equality test `test_positioned_bernoulli.cpp` keeps the two draws together.

## `random/dispatch.hpp`

### `K_RNG_SEED_PARAM`

The seed and offset sit in the two highest slots rather than immediately after the distribution's
own parameters so that their position is the same for every random op, whatever its arity: a
profiler or replay tool reading the metadata does not have to know which distribution produced the
task in order to find them.

### `Sampling` and `require_random_dtype`

Every random op names the element families it serves as a `Dispatch::Domain` and says whether it
draws new values or only moves existing ones. `Sampling::DrawsValues` refuses HALF on top of the
domain: HALF has too little mantissa for the inverse-CDF and Box-Muller transforms.
`Sampling::MovesValues` accepts every dtype of the domain; `shuffle` and `constant` use it.

`require_random_dtype` throws `std::invalid_argument` with one message for all of them: the op,
the operand `t`, its dtype and the dtypes the op accepts. The distributions call it through
`execute_random`; `shuffle`, `constant`, `orthogonal` and `truncated_normal` call it directly,
before a task is recorded.

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
graph bookkeeping, reserving a fresh counter offset from the `RNGStream` it is given so two tasks
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

Returns: The `TaskHandle` of the recorded task; the work is deferred until the caller runs the graph.

Warning: This is a low-level generic seam. The `draw_block` that `task_func` hands to `fill` runs
on the device: it must be device-copyable, capture by value, hold no host pointers, allocate
nothing and throw nothing. Draw through `Kernels::SYCL::Random` rather than a stateful host RNG.

Throws `std::invalid_argument` through `require_random_dtype` for HALF, the integer dtypes and any
dtype outside `Dom`, for a tensor with elements but no storage or not contiguous, and if `params` exceeds
`K_MAX_RNG_PARAMS`. Throws `UnsupportedExecution` for a dtype the device of `t` cannot compute in
and for a device-only `t` under `ForceReference`. All of these fire at record time and before the
offset is reserved, so a refused call does not move the stream.

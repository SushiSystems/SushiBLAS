# 12. The fixed-order fold (`reduce_1d.hpp`)

`modules/foundation/support/include/SushiBLAS/support/reduce_1d.hpp` holds `Reduce::reduce`, the
whole-tensor fold every order-sensitive reduction goes through, and the accumulators it folds
with. This page holds the reasoning behind the header and behind each symbol in it; the header
states each rule in one line. [The support README](../../modules/foundation/support/README.md)
covers the other support headers, and [section 7](KERNELS.md) says which operations fold.


`launch_1d.hpp` does this job for the per-element families; this is its counterpart for the folding
ones, and it exists for a correctness reason rather than a performance one.

**Why not `sycl::reduction`.** An additive reduction can be written as the one line
`sycl::reduction(p, T(0), sycl::plus<T>())`. The SYCL specification does not fix the order in which
such a reduction combines its partial results, and floating-point addition is not associative, so
the order is visible in the result's bits and the implementation is free to change it with the
launch configuration, the driver version, or the device. SushiBLAS links
`sushiruntime_deterministic_fp` PUBLIC, whose guarantee is that the same binary on the same device
produces byte-identical results, so that guarantee was resting on an implementation detail nobody
had promised. It happened to hold on this project's OpenCL CPU device. On the NVIDIA GPUs this
project targets, where a reduction's combination order does vary with the grid, it would have broken
exactly where the project was going.

**What replaces it.** A two-stage reduction whose every step is fixed by the geometry:

1. Each work-item walks a grid-strided slice of the input and accumulates it
serially, with `CompensatedSum`'s Neumaier correction. The slice, and
the order within it, are pure functions of the launch shape.
2. One work-group's items are folded into one partial by a hand-written
halving tree over local memory, **not** `sycl::reduce_over_group`, whose
order the spec leaves just as free as `sycl::reduction`'s.
3. A single task sums the fixed number of partials in ascending index order.

Two runs of the same binary on the same device therefore perform the same additions on the same
values in the same order, which is what byte equality actually requires.

**On a CPU, below a measured size threshold, there is no stage two at all**: one sequential pass is
both faster and has no cross-work-item combination to pin down. Above it, the two-stage shape wins
by more than an order of magnitude. See `planned_partials` for the measurements, and `reduce` for
the one caveat about what "sequential" buys.

## `K_MAX_GROUPS`

Stage two walks the partials sequentially in one task, so the count is a direct latency cost there;
and the scratch buffer that holds them is sized from it at record time. 256 partials is enough to
fill any GPU this project will meet (a 108-SM part at eight resident groups per SM wants 864 groups
for a *saturating* launch, but a reduction is bandwidth-bound and 256 groups of 256 items is already
65,536 work-items in flight), and 256 sequential adds is a rounding error against reading the input
at all.

## `K_CPU_TWO_STAGE_MIN_ELEMENTS`

Both shapes are deterministic, so this threshold is a pure performance choice, but it is a
*measured* one, and it is not the answer that was expected. See `planned_partials` for the numbers.

It does not weaken the reproducibility guarantee. The shape is a function of the device and the
element count only, both fixed for a given call, so two runs of the same reduction always take the
same branch and perform the same additions in the same order. A *different* size taking a different
shape is not a determinism problem, because it is a different computation.

## `floor_pow2`

The halving tree in stage one folds `local / 2`, `local / 4`, ... down to one, which only visits
every element when `local` is a power of two. A non-power-of-two group width would silently drop the
odd items, a wrong answer, not a slow one, so the width is rounded here rather than assumed.

## `planned_partials`

This is the function `Dispatch::execute_reduction` sizes its scratch buffer from, at task-record
time, and it is also what selects between the library's two reduction shapes. It depends only on the
device and the element count, both of which are fixed for a given call, so the choice is
reproducible.

**On a CPU it depends on the size, and the crossover was measured, not guessed.** The expectation
going in was that a CPU should always take the single pass: one work-item, one order, nothing to
argue about, and none of the cost of a work-group tree, a local allocation, two barriers per level
and a second kernel submission. That turned out to be true only for small inputs. Measured on this
project's OpenCL CPU device (12 compute units), best of seven runs of 200 recorded-and-executed
`sum` launches, single-pass against two-stage:

|        n | single pass | two stage |
|---------:|------------:|----------:|
|    1,024 |      7.2 ms |   10.7 ms |
|    4,096 |      8.3 ms |   10.3 ms |
|   16,384 |     10.1 ms |   10.4 ms |
|   32,768 |     13.4 ms |   10.4 ms |
|   65,536 |     19.7 ms |    9.9 ms |
|  262,144 |     53.6 ms |   11.8 ms |
| 1,048,576|    195.7 ms |   16.6 ms |

Below the crossover the single pass wins by up to 1.5x, because at that size the whole reduction is
launch overhead and the two-stage shape pays it twice. Above it the single pass is doing what it
says, one thread, one element at a time, while eleven other cores sit idle, and by 1M elements that
costs 11.8x. The two shapes are level at 16K, which is where `K_CPU_TWO_STAGE_MIN_ELEMENTS` sits.

A GPU takes the two-stage shape whenever the plan comes to more than one group: there a single task
uses one lane of one SM and leaves the entire machine idle at every size.

Returns: The partial count, or 0 to request the single-pass shape.

`planned_partials` returns 0 when the plan comes to one group. One group is a tree with nothing to
fold across, so it is the single-pass shape wearing a more expensive hat.

## `plan`

`planned_partials` is asked at record time and this is asked at execution time, so in principle they
could see different devices: the scratch is sized from the device the *data* lives on, and the task
is run on the device the scheduler routes it to. Clamping to `scratch_capacity` here means such a
disagreement costs performance and never correctness: too little scratch falls back to the single
pass rather than writing past the buffer.

## `CompensatedSum`

Plain summation throws away the low-order bits of every addition whose operands differ in magnitude,
and over a long input those losses accumulate into a result that can be arbitrarily wrong: summing
0.1f a million times naively lands near 100958 instead of 100000. This keeps a second register
holding exactly what each addition lost and adds it back at the end.

**Neumaier, not Kahan.** Kahan's correction assumes the running total is the larger operand, which
is true for a well-behaved input and false the moment a single large value arrives before many small
ones: exactly the case `1e8 + 1 + 1 + ...` that a gradient norm or a loss accumulation looks like.
Neumaier tests which operand is larger and is exact in both directions, for one extra comparison per
element in a kernel that is waiting on memory anyway.

- `T`: The element type. The correction is only as good as T's own precision, so it buys the most
  for `float` and comparatively little for `sycl::half`, whose ten mantissa bits leave little to
  recover.

## `CompensatedSum::rescale`

For an accumulator whose terms are expressed relative to a scale that can change mid-fold,
`ScaledSumOfSquares` is the one in this library, the total already folded has to move with it. Doing
that here rather than in the caller is what keeps the correction term correct: `comp_` is in the
same units as `sum_`, so rescaling one and not the other would quietly stop compensating anything.

- `factor`: The multiplier to apply. Exact when it is a power of two, which is the only way
  `ScaledSumOfSquares` ever calls this.

## `ScaledSumOfSquares`

The accumulation policy behind `nrm2`, and the reason it is a policy rather than a `transform` is
that squaring cannot be made safe one element at a time. `sqrt(fold(x * x))` is the obvious spelling
and it is wrong at both ends of the range: an fp32 element past ~1.8e19 squares to infinity, so the
norm of a vector whose largest entry is 1e20 comes back `inf` when the true answer is about 1e20;
and an element below ~1e-22 squares to zero, so a vector of denormals comes back `0` when the true
answer is a perfectly representable denormal.

Neither is academic in a training loop. Gradient-norm clipping calls `nrm2`, and a gradient that has
already exploded is exactly the input that overflows, so the mechanism meant to rescue the run is
the one that returns `inf` and ends it.

**What it does instead**, which is what reference BLAS's `snrm2`/`dnrm2` do: it keeps a running
`scale_` and accumulates `sum((x / scale_)^2)`, so what is squared is always a value near 1 however
large or small the input is. The finalised norm is `scale_ * sqrt(sum)`, and that product only
overflows when the true answer does.

**The scale climbs by binade, and is applied by multiplying.** Two details, both of which are about
cost rather than about the mathematics. The scale is raised only when an element reaches *twice* it,
not merely when it beats the running maximum, which caps the number of rescales at one per binade,
about 250 for fp32 however long the vector is, instead of one per element for a sorted input. And
the reciprocal is taken once per rescale so that the per-element step is a multiply, not the divide
the textbook form performs. Measured over 100 launches of a 1M-element fp32 `nrm2` on this project's
OpenCL CPU device: 5.9 ms for the naive form this replaced, 17.9 ms for the textbook
divide-per-element one, 11.0 ms for this with an unrounded scale. What is left over the naive form
is the per-element `fabs`, compare and multiply; `climb_to` gives the cost of rounding the scale to
a power of two, which the current form does.

**The scale is bounded**, to `[min(), 1 / min()]`, for fp32 from 2^-126 to 2^126. That is what keeps
its reciprocal a normal number, and it costs nothing: a magnitude below the bottom of that window
still scales to a value whose square is comfortably normal, and one above the top still scales to at
most 4. It is also what lets the hot path drop an `isfinite` test. An infinity clamps to the
ceiling, then scales to infinity and saturates the sum, which is the answer BLAS gives; a NaN fails
the `>=` that would rescale, then propagates through the multiply. Neither needs a branch of its
own.

**How it fits `reduce`.** `value()` returns a finished partial *norm*, a single `T`, not the (scale,
sum) pair, and `combine` merges two of them with `hypot`, which is itself overflow-safe. So the
two-stage reduction needs no wider partials buffer and no change to `reduce` at all: stage one folds
elements, the work-group tree merges partial norms, and stage two folds those partial norms back
through `accumulate`, which is correct because the norm of a set of partial norms is the norm of the
whole.

**Determinism** is unchanged. Every branch here is a function of the values folded and the order
they arrive in, and `reduce` fixes that order from the geometry alone.

## `ScaledSumOfSquares::min_scale`

The smallest positive *normal* value, which for every IEEE format is exactly a power of two.
Bounding the scale here is what keeps its reciprocal finite, and it costs nothing at the bottom of
the range: a denormal magnitude scaled up by `1 / min()` lands somewhere its square is comfortably
normal, which is the entire point: squaring it in place would have given zero.

Returns: `2^(min_exponent - 1)`.

## `ScaledSumOfSquares::max_scale`

The reciprocal of `min_scale`, and therefore itself a power of two whose own reciprocal is normal.
In every IEEE format that lands one binade below the top, which is what makes `4 * max_scale()`
overflow to infinity: the property `climb_to` relies on to stop the scale climbing.

Returns: `2^(max_exponent - 2)`.

## `ScaledSumOfSquares::combine`

`hypot`, not `sqrt(a*a + b*b)`, for the same reason this class exists at all: the squares are what
overflow, and `hypot` is specified to compute the result without forming them. It is called once per
tree level and once per work-group partial, never per element, so its cost is nothing.

## `ScaledSumOfSquares::accumulate`

Accepts either a raw element or an already-finished partial norm; the two are the same operation,
which is what lets stage two of `reduce` fold partials with the same accumulator that stage one
folds elements with.

At `if (a == T(0)) return;`: an exact zero contributes nothing, and letting it reach `climb_to`
would send the ladder there all the way to the bottom of the range for no gain.

At `if (a >= hi_) climb_to(a);`: leaving the current binade, not merely beating the running maximum.
`hi_` is `2 * scale_`, so on a well-conditioned vector this fires a handful of times in total rather
than once per element, and a NaN fails it, which is how a NaN reaches the multiply `a * inv_scale_`
unrescaled and propagates the way BLAS says it should.

`const T m = a * inv_scale_;` is exact: `inv_scale_` is a power of two, and the bounds `climb_to`
keeps guarantee the product neither overflows nor underflows.

## `ScaledSumOfSquares::climb_to`

**The scale is a power of two.** `climb_to` clamps the magnitude to `[min_scale, max_scale]` and
rounds it down to its binade with `ldexp(1, ilogb(clamped))`. Both bounds are powers of two, so the
rounded value stays inside them. The reciprocal `1 / next`, the rescale factor `shrink * shrink` and
the per-element multiply `a * inv_scale_` are then all exact, which is what the member briefs in
`reduce_1d.hpp` state. A value folded right after a climb scales into `[1, 2)`, so its square lies in
`[1, 4)`.

**What it costs.** Measured before this form was adopted, over 100 launches of a 1M-element fp32
`nrm2` on this project's OpenCL CPU device, with the branch that reaches here taken a handful of
times per launch:

| form                                   |   best |
|----------------------------------------|-------:|
| naive `sqrt(sum(x*x))` (wrong)          | 5.9 ms |
| scaled, `ldexp`/`ilogb` binade          |14.7 ms |
| scaled, doubling-ladder binade          |21.2 ms |
| scaled, unrounded scale                 |11.0 ms |

The `ldexp`/`ilogb` row is the current form. The two library calls sit in a branch the compiler
does not look past. The owner chose exact arithmetic over the 3.7 ms. The current form has not been
re-measured.

- `a`: A magnitude strictly above zero and at or above the current `hi_`. Never a NaN: the `>=` in
  `accumulate` excludes one.

At `T clamped = a;`: clamped so the reciprocal `inv_next` is a normal number. Neither end costs
anything: a magnitude under the floor still scales to something whose square is normal, and one over
the ceiling still scales to at most 4. An infinity lands exactly on the ceiling, which is what makes
it scale to infinity and saturate the sum rather than produce a NaN.

At `hi_ = (next < ceiling) ? (next + next) : (next * T(4));`: `hi_` is `2 * next`, the bottom of
the next binade. At the ceiling `4 * next` overflows to infinity by construction (the ceiling is one
binade below the top), which is what stops a vector of huge values from re-entering `climb_to` once
per element.

`const T shrink = scale_ * inv_next;` in `climb_to`: `shrink` is zero on the very first value
folded, which correctly clears an empty sum rather than dividing by a scale that does not exist yet.

## `RunningProduct`

No compensation, because there is nothing to compensate: Neumaier recovers the bits an *addition*
rounds away, and multiplication's rounding error is not of that shape. What this shares with
`CompensatedSum`, and the only reason it is spelled the same way, is the fixed combination order,
which is what the determinism argument actually rests on.

## `NaNPropagatingMax`, `NaNPropagatingMin`

The accumulators of every maximum and minimum in the library: `ReductionOps::max` and `min` through
the whole-tensor seam, `max_along` and `min_along` through the segmented one. Once a NaN is folded
the result is NaN, and `combine` returns NaN when either side is NaN, so the answer does not depend
on the combination order and a per-axis extreme agrees with a whole-tensor one on NaN. `combine`
also orders the two zeros, `+0` over `-0` for the maximum and `-0` over `+0` for the minimum, so it
gives the same result with its arguments swapped.

## `NaNPropagatingMax::identity`, `NaNPropagatingMin::identity`

Negative infinity for the maximum and positive infinity for the minimum when `T` has one,
otherwise `lowest()` and `max()`. Folding the identity in against any value is a no-op, so a
segment of negative infinities answers negative infinity rather than `lowest()`. Not `constexpr`,
because the `std::numeric_limits` members of `sycl::half` are not: the same reason
`ScaledSumOfSquares::min_scale` is a function.

## `ExactSum`

The integer counterpart of `CompensatedSum`, and it is a separate class rather than a specialisation
because it is doing something different: there is nothing to compensate. Neumaier recovers the bits
a floating-point addition *rounds away*, and an integer addition rounds nothing away: the correction
term would be identically zero, and `sycl::fabs`, which `CompensatedSum` reaches for to decide which
operand kept its bits, is not defined on an integer at all.

It exists for `reduce`'s counting shape: a reduction whose per-element transform answers a
*predicate* folds a stream of 0s and 1s, and the natural accumulator for that is an exact integer
sum, not a compensated float one.

**Determinism is free here, for a stronger reason than elsewhere in this file.** Integer addition is
exactly associative and exactly commutative, so no combination order can change the answer: the
fixed order `reduce` imposes is not what makes a count reproducible, it is simply the order the
shared shape already has. That is worth stating rather than leaving implicit, because the
determinism argument for the *floating-point* accumulators does rest on the fixed order and it would
be easy to read one as the other.

- `T`: The accumulation type; an integer type in every current use.

## `reduce`

Computes `finalize(fold(transform(in[i])), n)`, where `fold` is `Accum`'s operation applied in an
order fixed entirely by the launch geometry. Submits one kernel on the single-pass shape and two on
the two-stage one; see `plan` for which, and the opening of this section for why the order is
nailed down.

**The accumulation type is separate from the element type**, and every caller through
`Dispatch::FoldingReductionDispatch` deduces them equal without saying so. `sum`, `mean`, `asum`,
`nrm2` and `prod` all fold a `T` into a `T`, which is why this was one parameter for as long as
there was nothing else to do, but a reduction whose transform answers a *predicate* folds a count,
and a count is not an element. Storing one in the input's own dtype is the trap
`Dispatch::execute_counting_reduction` was cut out to escape: a count is exact only below 2^24 in
fp32 and only below 2048 in fp16, so a sufficiently bad tensor would report a rounded number of bad
elements, and in fp16 a count past 65504 would come back as an infinity.
Splitting `A` off costs the existing callers nothing (both are deduced, and both deduce to `T`) and
lets the counting reductions share this shape rather than hand-write a second copy of a two-stage
fold.

- `Accum`: The accumulation policy, `CompensatedSum`, `RunningProduct`, `NaNPropagatingMax`,
  `NaNPropagatingMin` or `ExactSum`. A class template on the *accumulation* type, supplying
  `identity()`, `combine(a, b)`, `accumulate(x)` and `value()`.
- `T`: The input element type, deduced from `in`.
- `A`: The accumulation and result type, deduced from `out` and `partials`. Equal to `T` for every
  reduction whose answer is a value of the thing it reduced.
- `partials`: Scratch for one `AccumulationOf<A>` per work-group; may be null when
  `partial_capacity` is zero.
- `transform`: Callable `(T) -> A`, applied to each element before it is folded in. `x * x` for a
  2-norm, `fabs(x)` for an absolute sum, `isfinite(x) ? 0 : 1` for a count of the elements that are
  not finite.
- `finalize`: Callable `(W value, std::size_t n)`, with `W` being `AccumulationOf<A>`, applied once
  to the folded result and converted to `A` when stored. `sqrt`
  for a 2-norm, `/n` for a mean. Folding it in here rather than submitting a second kernel is what
  lets `mean` and `nrm2` cost the same number of launches as `sum`.
- `deps`: Events this submission must depend on.

Returns: The event for the last submitted kernel.

Warning: `transform` and `finalize` are inlined into a SYCL device kernel: they must be
device-copyable, capture by value, hold no host pointers, allocate nothing and throw nothing.

At `return queue.submit([&](sycl::handler& h)`: one task, one ascending walk. Worth being precise
about what that does and does not promise: it fixes the order *this code* asks for, not necessarily
the order the machine executes. Removing the compensation from CompensatedSum and re-running this
path on the OpenCL CPU device produced 100003840 where a strict element-by-element float sum gives
exactly 100000000: the device JIT had vectorised the loop and folded the lanes, which is a
reassociation. With the compensation in place it does not, and either way the same binary on the
same device produces the same bits on every run, which is the guarantee that was asked for. So
"sequential" here means "no cross-work-item combination this code has to pin down", not "provably
index order all the way to the silicon".

At `Accum<W> acc;`: A grid-strided slice: work-item g owns indices g, g + stride, g + 2*stride, ...
Which indices those are, and the order they are visited in, come only from the geometry, so they are
the same on every run. Consecutive items also read consecutive addresses at every step, which is
what keeps the loads coalesced.

At `for (std::size_t off = local / 2; off > 0; off >>= 1)`: A hand-written halving tree, and NOT
sycl::reduce_over_group. The spec permits that function to combine in any order it likes, which for
a non-associative operator means it may return different bits on different launches of the same
kernel. This loop combines (lid, lid + off) for a fixed descending sequence of offsets, so the tree
is the same tree every time. `local` is a power of two by construction (see floor_pow2), which is
what makes the halving visit every lane exactly once.

At `Accum<W> acc;`: ascending index order over a count fixed at launch. Compensated as well, which
costs nothing at this length and recovers the spread between partials that a plain walk would drop.

`reduce` is a thin layer over `reduce_indexed`: it passes `read(i) = AccumulationOf<A>(transform(in[i]))`.
The geometry, the per-item walk, the tree and stage two are those of `reduce_indexed`, so the
operations performed and their order are the same as before the split.

## `reduce_indexed`

The same two shapes as `reduce`, over indices `[0, n)` instead of a pointer. The caller supplies
`read(i)`, which returns the accumulator input for element `i`, so a strided read (`x[i * incx]`), a
read of two operands (`x[i] * y[i]` for a dot product) or a read that yields a pair (value and index
for `iamax`) is expressible. The level 1 kernels in `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level1/level1.hpp` call it
directly.

- `Accum`: The accumulation policy, instantiated on `W`.
- `W`: The accumulator's value type and the partials' element type, deduced from `partials`.
- `Out`: The result type, deduced from `out`. `finalize`'s result is converted to it on store, so
  `iamax` can fold `IndexedValue<R>` and store an `int64_t`.
- `read`: Callable `(std::size_t i) -> W`; device-copyable, as `transform` is for `reduce`.

## `Accumulation` and `AccumulationOf`

Names the type a reduction over `T` accumulates in. It is `T` for every type except `sycl::half`,
which accumulates in `float`: HALF has 11 significant bits and a largest finite value of 65504, so a
running HALF sum of 60000 and 60000 is already an infinity. `reduce` folds in `AccumulationOf<A>`
and rounds to `A` once, when it stores the result. `Dispatch::execute_reduction` sizes a HALF
reduction's partials for `float` through `Dispatch::partial_bytes`, and the two stages of `var`,
`std` and `logsumexp` fold in `AccumulationOf<T>` as well.

## `CompensatedComplexSum`

A complex running sum kept as two `CompensatedSum`, one for the real part and one for the imaginary
part. `CompensatedSum` itself calls `sycl::isfinite` and `sycl::fabs`, which have no complex
overloads. `combine` is plain complex addition, as `CompensatedSum::combine` is plain addition.

## `IndexedValue`

A value and the index it was read from. `index` is -1 only in the identity of `RunningArgMax` and
`RunningArgMin`.

## `RunningArgMin`

The running (value, index) minimum, under the NaN rule `RunningArgMax` uses. `argmin` and the
host and SYCL paths of `Dispatch::ArgReductionDispatch` fold with it.

## `RunningArgMax`

The running (value, index) maximum `iamax` folds with. The serial reference kernel takes element 0
unconditionally and then replaces the best only on a strictly greater value. A NaN compares false,
so a NaN at index 0 is never replaced and a NaN anywhere else is never taken. `combine` expresses
that as a total order on candidates, so the result does not depend on how the grid splits the
input:

1. the empty candidate (index -1) loses to everything;
2. a NaN at index 0 beats everything;
3. a NaN at any other index loses to every non-NaN candidate;
4. otherwise the larger value wins, and on equal values (including two NaNs) the lower index wins.

The order is total, so the per-item walk, the tree and stage two all agree with the serial scan,
including on a tie split across two work-groups.

## `CompensatedSum::accumulate`

The correction step in `CompensatedSum::accumulate`, with `t` the rounded sum: whichever operand is
larger keeps its bits. The smaller one is the one that got rounded off, and `(larger - t) + smaller`
recovers exactly the part that did not fit.

The step runs only while `t` is finite. An infinite or NaN total carries no rounding error to
recover, and `(sum_ - t) + x` on an infinity would turn the correction into a NaN; `value()` returns
the non-finite total alone for the same reason.

## `Plan::groups`

Zero does not mean "no work". It selects a different kernel: one sequential pass with no partials,
no local memory and no stage two.

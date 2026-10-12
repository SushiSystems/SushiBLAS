# Support headers

`modules/foundation/support/include/SushiBLAS/support` holds the shared machinery that kernels and the `Engine` layer both
build on. It is the lowest layer: a support header includes only `SushiBLAS/core/`, other support
headers, SYCL and the standard library, and `tools/layering/check_layering.py` enforces that under
`rule_support_independence`.

| File | Declares |
| --- | --- |
| `accumulator.hpp` | `AccumulatorOf`, the type a BLAS kernel accumulates an element type in |
| `reference_accumulator.hpp` | `ReferenceAccumulatorOf` and `ReferenceSum`, the type and the running sum of a reference kernel |
| `running_solution.hpp` | `RunningSolution`, which carries a substitution's solution unrounded |
| `complex_arithmetic.hpp` | The conjugation, product and quotient a SYCL device can run on any element type |
| `device_math.hpp` | `Math`, the scalar functions a kernel may call: `std::` on the host, `sycl::` on the device |
| `device_policy.hpp` | `Profile`, `CacheBlocking` and the launch geometry a kernel dispatches on |
| `launch_1d.hpp` | The launch primitives every per-element kernel shares: `map`, `zip_inplace`, `indexed` |
| `serial_1d.hpp` | The serial host loops that are the reference path of those three launch shapes |
| `reduce_1d.hpp` | The deterministic whole-tensor reduction and the accumulators it folds with |
| `reduce_axis.hpp` | The segmented reduction: many independent folds, one result each |
| `serial_reduce.hpp` | The serial host folds that are the reference path of both reductions |
| `orthonormal_sweep.hpp` | The vector layout and the constants the orthonormalisation kernels of both back ends share |
| `spatial_geometry.hpp` | `ConvGeometry`, `WindowSpan`, `ColumnSpan`, `TapSpan` and `output_extent` |
| `sparse_assembly.hpp` | The host-side assembly of CSR arrays from triplets, and its input checks |

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `accumulator.hpp`

### `AccumulatorOf`

`AccumulatorOf<T>::type`, shortened to `Accumulator<T>`, names the type a BLAS kernel sums `T`
values in. It is `float` for `sycl::half` and `T` for every other element type. HALF has 11
significant bits, so a HALF running sum of ones stops growing at 2048: from there the spacing
between HALF values is 2, and 2048 + 1 rounds back to 2048.

The SYCL kernels of GEMM, SYRK, TRSM (and TRSV through it), GEMV, SYMV, TRMV, GER, SYR and SYR2,
and the reference kernels of GER, SYR and SYR2, widen each operand to `Accumulator<T>`, form the
sum in it and round to `T` once, when the element is stored. GEMM's epilogue still receives a `T`.
TRSM and TRSV read the solution elements they already solved through `RunningSolution`, which
hands a later row or column the unrounded value.
For every type but HALF the widening is the identity and the kernels run the same operations in the
same order as before. `Reduce::Accumulation` and `Reduce::AccumulationOf` alias this trait.

## `reference_accumulator.hpp`

### `ReferenceAccumulatorOf`

A reference kernel sums a real element type in `double` and a complex one in
`std::complex<double>`, whatever the element type. The reference is the specification the SYCL
kernel is compared against, so its sum is wider than the one under test wherever a wider type
exists.

### `ReferenceSum`

The running sum of a reference kernel. For HALF, FLOAT32 and COMPLEX32 the wide type already
holds more bits than the element, and the sum is a plain `double` sum. For FLOAT64 and COMPLEX64
the wide type is the element type itself, so `ReferenceSum` keeps the rounding error of every
addition with TwoSum and adds it back when the sum is read. The two differences TwoSum takes are
exact only under strict IEEE arithmetic, which the library builds with.

## `running_solution.hpp`

### `RunningSolution`

A triangular substitution reads the elements it has already solved. `RunningSolution` carries
them in `Accumulator<T>` between steps and stores each one rounded to `T` in the right-hand side
as well, so a later row reads the unrounded value. For every type but HALF `Accumulator<T>` is
`T` and the solution lives in the right-hand side itself. For HALF it needs a workspace of one
`float` per element, which `K_NEEDS_WORKSPACE` reports and
`Internal::running_solution_bytes` sizes at record time.

## `complex_arithmetic.hpp`

`std::complex` multiplication and division are host library calls on some toolchains, and a SYCL
device cannot call them. This header spells both on the real and imaginary parts: the product by
the four-product formula, the quotient by Smith's algorithm, which returns NaN components for a
zero divisor. Each has a real overload that is the plain operator, so a kernel template calls one
name for every element type. `RealType<T>` names the real component type, and the conjugation
helpers return a real value unchanged.

## `device_math.hpp`

`SushiBLAS::Math` names the scalar functions that code shared by the host and a kernel may call:
`abs`, `asin`, `atan2`, `cbrt`, `copysign`, `cos`, `exp`, `floor`, `fmod`, `isfinite`, `log`,
`pow`, `sin` and `sqrt`. On the host each is the `std::` function. In the device pass of a SYCL
compiler each is the `sycl::` function, and `abs` is a template over `sycl::fabs`. A function
written once against `Math::sqrt` therefore runs in both places.

The switch is `__SYCL_DEVICE_ONLY__`, which the compiler defines only while it compiles device
code. Every host translation unit sees the `std::` names, whether it is a SYCL one or not, so the
header gives one definition across a program's host objects.

`<sycl/sycl.hpp>` is included only when `SYCL_LANGUAGE_VERSION` is defined. A translation unit
built without SYCL includes `<cmath>` and nothing else, which is what lets
`modules/foundation/geometry` compile in a consumer's build lane that has no SYCL.

## `device_policy.hpp`

The fastest CPU kernel and the fastest GPU kernel are not the same kernel, and for the two shapes
that matter most here they are close to opposites. A GPU wants a large work-group cooperatively
staging tiles through scratchpad memory, bank-padded, with each work-item holding a wide register
block. A CPU wants none of that: its "local memory" is ordinary cache reached through the same
loads, so staging through it copies the data twice and every `group_barrier` becomes a real
synchronisation between OS threads for no benefit. portBLAS ships exactly this split, `gemm_local`
for real GPUs, `gemm_no_local` for everything generic, and it is the split this header exists to let
a kernel express.

The rule this header enforces is that a kernel dispatches on *facts*, never on a vendor string.
`has_dedicated_local_memory` is a question about the memory hierarchy; `is_cpu` is a question the
SYCL runtime already answers. Neither is "am I on an RTX 4090", which is the kind of test that rots.

Nothing here is GEMM-specific or elementwise-specific on purpose. A Profile describes the machine;
choosing a tile size or an unroll factor from it is the individual kernel's business, because only
the kernel knows how many registers its accumulator costs. Pushing those choices in here would make
this file change every time any kernel is tuned.

This is a public installed header: SushiAI's fused kernels reach the same policy through the
`Dispatch` seam, so a consumer-authored epilogue gets the same launch geometry as a built-in one
rather than an accidental `sycl::range` fallback.

### `K_WAVES_PER_COMPUTE_UNIT`

A grid-stride kernel does not need one work-item per element; it needs enough work-groups to fill
the machine and hide memory latency, and anything beyond that is launch overhead. Eight per compute
unit is the usual occupancy target for a bandwidth-bound kernel on a discrete GPU: enough concurrent
waves that a stalled one always has a neighbour ready, without the register pressure of chasing
maximum occupancy.

### `Profile`

A plain aggregate of answers, queried once and cached. Everything here comes straight from
a device query except `blocking`, which `cache_blocking_for` derives from `cache_bytes`, and
`has_dedicated_local_memory`, which is a judgement the raw query cannot make on its own: see that
field.

### `CacheBlocking`

The byte budgets and thresholds a cache-blocked kernel sizes itself from, held in the profile as
`Profile::blocking`. `cache_blocking_for(cache_bytes)` derives them from the cache size the device
reports: a quarter for the block the inner loops sweep again, half for the block walked once per
sweep, and the whole of it as `min_operand_bytes`, the operand size below which a kernel does not
pack. A device that reports no cache gets `K_ASSUMED_CACHE_BYTES`, 256 KiB. `inner_bytes` is
seven eighths of `K_ASSUMED_INNER_CACHE_BYTES`, 32 KiB, for every device: the probe reads one
cache size, `global_mem_cache_size`, and does not derive `inner_bytes` from it.
The packed GEMM reads these through `Kernels::plan_packed_product`; nothing in this header knows
what a GEMM is.

### `Profile::has_dedicated_local_memory`

`local_mem_type` alone is not enough to answer this. An OpenCL CPU device reports
`local_mem_type::local` and a plausible `local_mem_size`, but backs both with the same cache line
the global pointer already reads through. Staging a tile there is a pure extra copy, and the
barriers guarding it are pure loss. So a kernel that asks "is it worth tiling?" must ask this, not
`local_mem_type`.

### Sub-group width in the device probe

At `const std::vector<std::size_t> widths =`: widest reported sub-group: 32 on NVIDIA, 64 on AMD
wave64, and on Intel the widest of {8,16,32}, which is the one that amortises a shuffle reduction
best. An empty list would be a spec violation, but a zero here would divide by zero downstream, so
it is guarded.

### `profile_for`

Called on every task execution, so it must not re-query: a replayed graph runs the same fifty nodes
thousands of times and six `get_info` round trips per node per step is real overhead against a
few-microsecond kernel.

A std::deque, not a vector: a reference handed to an earlier caller must stay valid when a second
device is probed, and only deque guarantees that across emplace_back.

Deliberately leaked. The entries hold sycl::device handles, and a static destructor running at exit
is not ordered against the SYCL runtime tearing itself down: releasing a device handle afterwards is
a crash on some backends. One small permanent allocation is the standard price for not having that
race.

### `Grid1D`

The kernel this describes walks its elements with a stride loop rather than assuming one work-item
per element:

```cpp
const auto grid = Device::plan_1d(profile, n, unroll);
h.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it) {
    for (std::size_t i = it.get_global_id(0); i < n; i += grid_stride)
        out[i] = op(in[i]);
});
```

The `i < n` test is therefore the bounds check a padded grid would otherwise need, so there is no
divergent tail work-group whatever `global_size` turns out to be.

On a GPU `global_size` is deliberately *not* `n` rounded up: it is capped at roughly what the
machine can hold resident and the loop covers the rest, which stops a large tensor from scheduling a
grid far bigger than the device can run at once. On a CPU it is `n` rounded up, so the loop body
runs once: see `plan_1d` for the measurement that settled that split.

The traversal order is fixed by the geometry, so two runs of the same kernel on the same device
touch elements in the same order, which is what makes a deterministic reduction built on this
reproducible.

### `flat_work_group_size`

Not `work_group_size`, and the difference is the point. `work_group_size` answers "how wide a group
can cooperate cheaply?", which is a question about barriers, and its CPU answer is deliberately
small because a barrier there is a rendezvous between OS threads. A flat elementwise launch has no
barriers at all, so nothing about that answer applies to it; what it pays for instead is *work-group
count*, because on an OpenCL CPU backend each group is a scheduling unit with real per-group cost.
Aiming at one group per hardware thread and letting the group get as wide as it needs to be turns
that cost from "once per 64 elements" into "once per thread".

The GPU answer is just `work_group_size:` there, group count is how occupancy is expressed, and 256
work-items is already the right width.

The rounding `(wanted / p.sub_group_size) * p.sub_group_size` in `flat_work_group_size` keeps whole
sub-groups only, for the same reason `work_group_size` rounds: a partial trailing sub-group is dead
lanes.

### `plan_1d`

- `unroll`: Elements each work-item handles per stride step; 1 for a plain scalar loop, 4 when the
  kernel issues 128-bit vector accesses. Only affects how large the grid needs to be.

A CPU gets a grid that covers the whole problem, so the stride loop body runs at most once and the
launch is "one work-item per element" with an explicit work-group size. That is the opposite of the
GPU choice below, and it is a locality argument rather than an occupancy one: a CPU backend hands
each OS thread a contiguous run of work-groups, which its prefetchers stream through, and capping
the grid replaces that with a strided walk: each thread touching one group's worth of elements, then
jumping past every other thread's share to reach its next one.

Measured on this project's OpenCL CPU device (12 compute units, 8192 max work-group), 200 launches
of a 1M-element elementwise add: 28.3 ms capped at one group per compute unit, 9.7 ms uncapped with
the work-group sizing above, against 8.2 ms for the bare `sycl::range<1>` this replaced. The
residual gap is nd_range setup the CPU backend charges and the flat launch does not; at the small
and medium sizes this library actually launches at (1K-64K), the planned shape is the faster of the
two by 5-7%.

A GPU keeps the cap. Consecutive work-items still read consecutive elements at every step of the
stride loop, so each step is as coalesced as a flat launch would be, oversubscribing resident
work-groups is how memory latency gets hidden, and the cap is what stops a large tensor from
scheduling a grid far larger than the device can hold resident at once.

### `vector_width`

Returns 1 on a CPU. Not an oversight: the OpenCL CPU backend already vectorises *across* work-items,
packing neighbouring items into SIMD lanes, and a hand-written `sycl::vec` fights that by consuming
the same lanes for a single item. On a GPU there is no such implicit vectorisation, and a 128-bit
access is one transaction where four 32-bit accesses are four.

## `launch_1d.hpp`

A bare `sycl::range<1>(n)` launch is one work-item per element, with no work-group size, no
vectorisation, and the runtime free to pick whatever geometry it likes. That is the shape a
device is least likely to want. Every elementwise, non-linear and logic op launches through this
header instead, which has three primitives, `Launch::map`, `Launch::zip_inplace` and `Launch::indexed`, that own the geometry
once, so tuning it is one edit rather than fifty-five.

`Launch::map` and `Launch::zip_inplace` serve the kernels whose memory access pattern this header
can see: contiguous, one element per index, so it may vectorise them. `Launch::indexed` serves the
ones whose pattern is their own business: a Level 1 vector walked with a BLAS `incx`, a GEMV output
row that reduces across a whole matrix row, an RNG fill whose value is a function of the index
rather than of any input array. Those get the device split and nothing else, which is the part that
was missing: before this they were the launch sites still spelling `sycl::range<1>(n)` by hand and
therefore still getting the runtime's default geometry on a GPU.

Unless `Shape1D::flat` is set, all three launch a **grid-stride** `nd_range` planned by
`SushiBLAS::Device::plan_1d:` a grid sized
to what the device can hold resident, walked with a `for (i = gid; i < n; i += stride)` loop. Two
things follow. The `i < n` test *is* the bounds check, so there is no padded, half-idle tail
work-group; and the launch stops growing once the machine is full, which is what keeps a large
tensor from paying to schedule a grid the device cannot run at once.

Where `SushiBLAS::Device::vector_width` says a 128-bit access is worth issuing, the loop is preceded
by a wide pass that loads and stores `sycl::vec<T, W>` but **calls the caller's functor once per
scalar component**. That is deliberate: the functor contract stays scalar, so every existing caller,
including a downstream consumer's own fused lambda, keeps working unchanged and never has to know
whether it was vectorised.

**Determinism.** Both the grid and the scalar/wide split are pure functions of the device, the
element type, `n`, and the operands' alignment, none of which vary between two runs of the same
binary on the same device. Every element is written exactly once, by exactly one work-item, from the
same inputs, in the same arithmetic. So the byte-for-byte reproducibility
`sushiruntime_deterministic_fp` promises survives this header intact, which matters, because a
training framework above compares checkpoints for bit-equality.

Each primitive has two forms. One takes a `Shape1D` the caller already planned; the other takes
the queue and plans the shape from its device. The shape form is what lets a test, or a caller
that has planned once for several launches, run a kernel in a shape the planner would not pick
for that size.

Under `ForceReference` none of this runs: the same functor is called by the loops of
`serial_1d.hpp`.

### `K_VECTOR_WIDTH`

The compile-time companion to `SushiBLAS::Device::vector_width`, which answers the same question at
runtime but cannot be used to *name* a type. They must agree before a wide path is taken, and `plan`
checks exactly that, if the device policy ever returns a width this constant does not predict, the
launch falls back to scalar rather than issuing an access of the wrong shape.

1 for `std::complex` on purpose. There is no `sycl::vec<std::complex<T>, W>`, and reinterpreting a
complex array as 2N reals would hand the functor halves of numbers rather than numbers.

### `is_vector_aligned`

Asked on the **host**, at task-record time, against the pointer the kernel will actually be handed.
A `Tensor` view is only guaranteed to be aligned to its own element type (see
`Tensor::storage_offset`), so an odd `storage_offset` into a shared arena produces a perfectly valid
tensor whose data pointer is not 16-byte aligned. Assuming otherwise and issuing a wide load anyway
is a fault on some devices and silently slow on the rest.

### `typed_ptrs`

The dispatch seam carries operand addresses as `void*` because the element type is a runtime
`Core::DataType`; each arm of the dtype switch needs them typed. This is that cast, once, instead of
once per operand per arm.

### `Shape1D::flat`

The one place this header takes the device split seriously for its own launches rather than only for
GEMM's. A grid-stride kernel exists to let a GPU size its grid to what it can hold resident and walk
the rest; a CPU backend has nothing to hold resident, already iterates work-items in a loop of its
own, and charges real `nd_range` setup for the privilege.

Measured on this machine over 200 chained adds at 1M elements: the grid-stride shape was ~19% slower
than the flat one it replaced. Below 64K the two are within run-to-run noise. So the flat shape is
not merely simpler on a CPU, it is faster, and `device_policy.hpp` says in its own opening paragraph
that these two machines want opposite shapes: it would be strange to measure exactly that and then
ship one shape for both.

Only ever set when the wide path is off, which on a CPU is always (`Device::vector_width` returns 1
there by design), so the flat kernel is the plain scalar body and nothing is lost by skipping the
stride loop.

### `plan_for`

The one place the flat/grid-stride decision and the wide-pass arithmetic are written down. `plan`
and `plan_scalar` differ only in how they arrive at `width`, so neither of them restates any of
this.

At `s.flat = p.is_cpu && width == 1 && n > 0;`: n == 0 stays on the grid-stride path, whose geometry
is never empty, so no caller has to think about whether this backend accepts a zero range.

### `plan`

At `const bool vectorize = (width > 1) && aligned &&`: both conditions, not either: the compile-time
width names the type, the runtime one carries the policy (it returns 1 on a CPU deliberately, see
device_policy.hpp, because the OpenCL CPU backend already vectorises across work-items and a
hand-written sycl::vec fights it for the lanes).

### `plan_scalar`

What `indexed` plans with. There is no element type in the signature because there is no element the
launch knows about: the caller's body decides what it touches and how, so the only question left is
the device one: flat on a CPU, grid-stride on a GPU.

### `Detail::load_wide`

A direct aligned access rather than `sycl::vec::load`, which the spec defines as `W` separate scalar
reads and which therefore only becomes a single transaction if the optimiser both fuses them and
proves the alignment. Here the alignment is already known, checked on the host by
`is_vector_aligned`, so the load can be spelled as what it is. The static_asserts pin the layout
that makes it legal: `sycl::vec<T, W>` is a `T[W]` aligned to its own size, so it is
layout-compatible with `W` consecutive `T`.

### `map`

The read-many/write-one shape: every elementwise, non-linear and logic executor in the library is
one call to this per dtype arm.

`out` may alias any input: the in-place unary executors rely on it. Each index is handled by exactly
one work-item, which loads before it stores, so aliasing at the *same* index is well-defined;
overlapping at *different* indices is not, and no elementwise op in this library does it.

- `T`: The element type, deduced from `out`.
- `Ins`: The input element types, deduced; all must equal `T`.
- `q`: The queue being submitted to, used only to reach the device profile.
- `aligned`: Whether `out` and every input satisfy `is_vector_aligned`.
- `op`: Callable `(auto...) -> auto`, invoked once per element on device.

Warning: `op` is inlined into a SYCL device kernel: it must be device-copyable, capture by value
only, hold no host pointers, allocate nothing and throw nothing.

The grid-strided loop `for (std::size_t i = tail + gid; i < n; i += stride)` in `map` covers
everything when there was no wide pass (`tail == 0`), and only the sub-vector remainder when there
was.

### `zip_inplace`

The all-read-write shape, and the one that lets a whole optimiser step collapse into a single
kernel: the functor takes lvalue references, so it can read and update any operand it likes in one
pass over memory.

Each element index is staged into registers, handed to `op`, and written back, so a functor that
touches an operand repeatedly pays one load and one store for it, not one per mention, and the
scalar and wide paths have identical semantics. The consequence to know about: the `ptrs` arrays
must not overlap each other, because a write through one staged slot is not visible to another slot
that happens to name the same address.

- `q`: The queue being submitted to, used only to reach the device profile.
- `aligned`: Whether every pointer in `ptrs` satisfies `is_vector_aligned`.
- `ptrs`: The arrays to update, in the order `op` expects them.
- `op`: Callable `(auto&...) -> void`, invoked once per element on device.

Warning: Same device-kernel constraints as `map`.

### `indexed`

The shape for a kernel that owns its own memory access pattern and therefore cannot be spelled as
`out[i] = op(in[i]...)`:

- a Level 1 BLAS vector, whose `incx`/`incy` mean index `i` is at `i * incx`
rather than at `i`;
- a GEMV or SYMV output element, which is a reduction across a whole matrix
row and reads `in_len` values to write one;
- a counter-based RNG fill, whose value at `i` is a function of `i` and no
input array at all.

What it provides is exactly the device split, a flat `range<1>` on a CPU, a grid-stride `nd_range`
on a GPU, taken from the same `plan_for` that `map` and `zip_inplace` use, so there is one geometry
in this library rather than one for the kernels that fit `map` and the runtime's default for the
rest.

What it deliberately does **not** provide is the 128-bit wide path. That pass loads `sycl::vec<T,
W>` from a *contiguous* array, which is a claim about the caller's access pattern that this
primitive, by construction, cannot make.

**Determinism.** Every index is visited exactly once by exactly one work-item, and which work-item
that is comes only from the geometry: a pure function of the device and `n`. A body that writes
`out[i]` from index `i` alone is therefore as reproducible here as it was under a flat launch. A
body that accumulates across work-items would not be, and is not what this is for; see
`<SushiBLAS/support/reduce_1d.hpp>` for that.

- `q`: The queue being submitted to, used only to reach the device profile.
- `op`: Callable `(std::size_t i) -> void`, invoked once per index on device.

Warning: `op` is inlined into a SYCL device kernel: it must be device-copyable, capture by value
only, hold no host pointers, allocate nothing and throw nothing.

## `reduce_1d.hpp`

The deterministic whole-tensor fold and its accumulators. The reasoning behind the header and
behind each of its symbols (`plan`, `reduce`, `reduce_indexed`, `CompensatedSum`,
`ScaledSumOfSquares`, `RunningProduct`, `NaNPropagatingMax`, `NaNPropagatingMin`, `ExactSum`,
`RunningArgMax`, `RunningArgMin`, `AccumulationOf`) is in
[the architecture, section 12](../../../docs/architecture/FIXED_ORDER_FOLD.md), one heading per
symbol.

## `reduce_axis.hpp`

`reduce_1d.hpp` folds a whole tensor to one scalar and `launch_1d.hpp` maps every element to its own
output. This is the third shape, and the one neither of those can express: N folds running side by
side, each over its own slice of the same buffer, producing a vector.

**Why it exists.** A log-sum-exp shift has to be *per row*. Shifting a whole batch by the batch's
global maximum is exact in real arithmetic, log-softmax is invariant to subtracting any constant
from a row, and wrong in binary32, because the invariance is destroyed by the exponential
underflowing: a row sitting 104 below the batch maximum has every `exp(x - global_max)` flush to
zero, its row sum is exactly zero, `log(0)` is `-inf`, and the row comes back `+inf`. That is a
silently wrong answer for gaps around 90-104 and a catastrophically wrong one past it. The only fix
is a maximum computed per row, and a per-row maximum is this shape.

**The layout is a stride pair, not an axis number.** `AxisSpan` describes "where do the segments
start, and how do I walk one" in element counts, which makes the axis, the `Core::Layout` and any
non-unit stride a *caller's* problem, resolved once on the host. Reducing along either axis of
either layout is the same kernel with different numbers, so there is no layout branch in the inner
loop and no second instantiation.

**Two shapes, for the two regimes.** A segmented reduction has two independent sources of
parallelism and which one is available is a property of the input:

1. **Item per segment** (`AxisPlan::cooperative` false). Each work-item
owns one whole segment and folds it serially. The parallelism is the
segment count, so this is what a tall, narrow reduction wants, a
row-wise reduction over a batch, which is the case this file was written
for. Goes through `Device::plan_1d`, so it inherits the CPU-flat /
GPU-grid-stride split `launch_1d.hpp` measured.
2. **Work-group per segment** (`AxisPlan::cooperative` true). Each
work-group owns one segment, grid-strides across it and folds through a
halving tree over local memory. The parallelism is the segment *length*,
which is what recovers the machine when there are fewer segments than it
has execution units: a `[2, 10M]` row-wise sum would otherwise run on
two work-items.

Both are correct at every shape; the choice is only ever about speed. In particular neither requires
the segment length to relate to the work-group width in any way: see `reduce_axis` for what the tree
does with a partial group.

**Determinism.** Same standard, and the same argument, as `reduce_1d.hpp`: `sycl::reduction` and
`sycl::reduce_over_group` are both unusable because the specification fixes no combination order for
them and floating-point addition is not associative. Shape 1 has *no* cross-work-item combination at
all: one item folds one segment in ascending index order. Shape 2 combines through the same
hand-written fixed-offset halving tree stage one of `Reduce::reduce` uses. In both, which elements a
given lane folds and the order it folds them in are pure functions of the launch geometry, and the
geometry is a pure function of the device profile and the span. Two runs of the same binary on the
same device therefore perform the same operations on the same values in the same order. A second
segment axis (`AxisSpan::segments2`) leaves that argument alone: it only changes which address a
fold starts at, so the fold order *within* a segment did not move and no combination across segments
was introduced.

That is an argument from what the code does, not a measurement of a GPU: the same limit
`reduce_1d.hpp` records for itself. See `docs/design/GPU_VALIDATION.md` §5.

**No scratch buffer, and that is structural rather than lucky.** A whole-tensor reduction needs one
because its work-groups all contribute to a single answer and something has to hold the partials
between the two kernels. Here a work-group's tree *is* its segment's answer, so there is nothing to
combine across groups, nothing to stage, and one kernel submission either way.

### `AxisSpan`

Everything the kernel needs to address memory, and nothing about *why*: an axis index, a
`Core::Layout` and a strided view all collapse into these five numbers on the host, so the device
never sees any of them. The element at position `k` of segment `s` is `in[s * segment_stride + k *
element_stride]`, and its result goes to `out[s * out_stride]`.

All strides are element counts and all are non-negative, which `Tensor`'s stride-explicit
constructor already guarantees for any view this can be built from.

### `AxisSpan::segments2`

The second **segment** axis, not a second folded one, and which of the two it is decides which loop
changes. Folding two axes at once needs nothing new: `[N*H*W, C]` is a zero-copy rank-2 view and the
existing five numbers fold axis 0 to `[C]`. What the five numbers cannot express is a segment
identified by a *pair*: `[N,H,W,C] -> [N,C]` has one result per `(n, c)`, whose input offset is `n *
H*W*C + c`, and no single `segment_stride` produces that from a single index.

Segment `(s2, s)` starts at `in[s2 * segment_stride2 + s * segment_stride]` and its result goes to
`out[(s2 * segments + s) * out_stride]`.

### `plan_axis`

The rule is a parallelism argument and nothing else, because both shapes are correct at every size:

- **Enough segments to fill the machine?** Then one item per segment,
which is strictly cheaper: no local memory, no barriers, one
coalesced grid-stride launch, and the fold order is trivially fixed.
"Enough" is one work-item per hardware thread on a CPU, and
`K_WAVES_PER_COMPUTE_UNIT` resident groups' worth on a GPU, matching
what `Device::plan_1d` targets for a flat launch.
- **Otherwise, is a segment long enough to be worth a work-group?** A
group whose segment is shorter than the group is mostly idle lanes
paying log2 of `local` pairs of barriers, so the tree only earns its
keep once every lane has at least one element to fold. One full
group-width pass is that threshold.
- **Otherwise** there is nothing to recover; take the cheap shape.

Warning: The GPU side of this is **derived, not measured**: no GPU run has measured the crossover
(see `docs/design/GPU_VALIDATION.md`). The threshold is a floor below which the cooperative shape is
certainly wrong, not a tuned crossover, and it is the first knob to reach for if a row-wise
reduction underperforms on real hardware. It cannot make an answer wrong: the shape is a pure
function of the device and the span, so it is the same on every run, and both shapes compute the
same thing.

Returns: The geometry; `AxisPlan::cooperative` says which shape.

### `reduce_segments`

The segmented fold with the reads and the store left to the caller. It reduces each of `segments`
index ranges `[0, extent)` to one value: `open(segment)` returns a cursor, the cursor returns
term `k` of that segment in the type `W` the accumulator folds in, and `store(segment, value)` is
called once per segment. `reduce_axis` is this with a cursor that walks an `AxisSpan` and a store
that writes `out[s]`. `Kernels::SYCL::window_dot_weights_backward` calls it directly, because a
tap's terms are products of two tensors and not elements of one.

### `reduce_axis`

Computes `out[s] = finalize(fold(transform(in[s][k])), extent)` for every segment `s`, where `fold`
is `Accum`'s operation applied in an order fixed entirely by the launch geometry. One kernel
submission, whichever shape `plan_axis` chose.

A segment shorter than the work-group is still correct on the cooperative shape: a lane with no
element of its own contributes `Accum<AccumulationOf<T>>{}.value()`, which is the accumulator's
identity by construction, and combining the identity changes nothing. `plan_axis` declines that
shape below one full pass for cost, not correctness.

- `Accum`: The accumulation policy (`CompensatedSum`, `NaNPropagatingMax`, `NaNPropagatingMin`,
  `RunningProduct`), a class template on the accumulation type supplying `identity()`,
  `combine(a, b)`, `accumulate(x)` and `value()`.
- `span`: The segment layout, in element counts.
- `in`: The input array; fold `s` starts at `in + (s / segments) * segment_stride2 + (s % segments)
  * segment_stride`, which is `in + s * segment_stride` whenever there is no second segment axis.
- `out`: Where the per-segment results go.
- `geometry`: Which shape to launch, normally `plan_axis`'s answer. Taken rather than decided here
  so that the planner and the launcher stay separable: a test can then drive both shapes over one
  input and check they agree, which is the only way the cooperative tree gets exercised at a size
  the planner would not send to it.
- `transform`: Callable `(T) -> auto`, applied to each element before it is folded in. Pass the
  identity when there is nothing to do.
- `finalize`: Callable `(AccumulationOf<T> value, std::size_t n) -> auto`, applied once per
  segment, with `n` the segment's own element count, so a per-axis mean is this seam plus `v / n`
  rather than a second kernel.
- `deps`: Events this submission must depend on.

Returns: The event for the submitted kernel.

Warning: `transform` and `finalize` are inlined into a SYCL device kernel: they must be
device-copyable, capture by value, hold no host pointers, allocate nothing and throw nothing.

At `const Device::Grid1D grid = Device::plan_1d(Device::profile_for(queue), segments);`: one item
per segment, over the same flat/grid-stride launch shape every elementwise op takes, so a CPU covers
the segments in one pass of a wide group and a GPU walks a capped, resident grid. The stride loop's
`s < segments` test is the bounds check.

At `for (std::size_t off = local / 2; off > 0; off >>= 1)`: the same hand-written halving tree as
stage one of Reduce::reduce, and NOT sycl::reduce_over_group, for the reason recorded there: the
spec lets that function combine in any order it likes, which for a non-associative operator means
different bits on different launches of one kernel. `local` is a power of two (see
Reduce::group_width), which is what makes the halving visit every lane exactly once.

The flat branch of `reduce_segments` folds each segment in ascending index order, and no other
work-item touches it. There is nothing in that loop whose order the SYCL specification leaves free.

`const std::size_t s2 = s / segments_;`, in `Detail::AxisOpen`, for both branches: one integer
division per fold, against an inner loop of `extent` elements. It is not worth a specialised
`segments2 == 1` kernel, which would be a second instantiation of every reduction in the library.

The lane loop in the work-group branch of `reduce_segments` walks a stride-`local` slice of one
segment: lane `l` owns positions `l`, `l + local`, `l + 2*local`, and so on. Which positions those
are comes only from the geometry, so it is the same on every run, and consecutive lanes read
consecutive addresses at every step.

## `serial_1d.hpp`

`Serial::map`, `Serial::zip_inplace` and `Serial::indexed` are the host loops behind
`ForceReference` for every kernel that launches through `launch_1d.hpp`. Each takes the functor
its `Launch` sibling would launch and calls it once per element, in index order, on one thread.
There is no second implementation of an operation to keep in step: the reference path and the
SYCL path differ only in who calls the functor. The functor contract is the device one, so a
functor that runs on the device runs here.

## `serial_reduce.hpp`

The host folds behind `ForceReference` for `reduce`, `reduce_indexed` and `reduce_axis`:
`Serial::reduce`, `Serial::reduce_indexed` and `Serial::reduce_axis`, each one ascending walk.

### `Serial::Fold`

Folds on the host in the accumulator the SYCL path uses, in the type the SYCL path folds in
(`Reduce::AccumulationOf<A>`). The specialisation for `Reduce::CompensatedSum` is the exception:
a sum is folded in `Kernels::ReferenceSum`, that is, in `double` whatever the element type. So
`finalize` is handed a `double` on the host path of a sum, and must return its value in the type
it is handed; `Dispatch::Detail::require_finalize_keeps_width` refuses at compile time one that
does not. The result is converted to the stored type once, on the store.

## `orthonormal_sweep.hpp`

`VectorSet` is the strided layout of the vectors `RandomOps::orthogonal` orthonormalises: a count,
a length, the step between elements of one vector and the step between vectors.
`K_ORTHONORMAL_SWEEPS` is 2, the number of modified Gram-Schmidt sweeps, and `tiny_norm_sq` is
the squared norm at or below which a vector counts as zero. The reference and the SYCL kernel
read all three from here, so the two back ends cannot disagree on them.

## `sparse_assembly.hpp`

Host code that knows no tensor and no engine. `Sparse::assemble_csr` turns COO triplets into the
three arrays of a CSR matrix, `HostCsr`: each (row, column) pair once, by ascending column in a
row, with the values of a repeated pair added in input order through `Kernels::ReferenceSum`.
`require_distinct_positions` checks the stored positions of a sparse vector. Both throw
`std::runtime_error`, as does the check that an index type can hold a count. `SparseBuilder` is
the caller.

## `spatial_geometry.hpp`

Separate from `window.hpp` and free of `Tensor` on purpose. The output-extent formula
(`output_extent`) is the *only* statement of how an input extent, a padding pair, a dilation, a
kernel and a stride combine, and it has two consumers on either side of a repository boundary: this
library's kernels, and SushiAI's host-lane shape inference. That inference is device-free by design,
it is exercised in a unit tier that links no SYCL, so the arithmetic has to be reachable without
dragging the device stack behind it. Two copies of one formula that must agree is the alternative,
and they would agree until the day they did not.

Includes `<cstdint>` and nothing else. Keep it that way.

### `ConvGeometry`

Exactly eight numbers, and the count is load-bearing rather than tidy. SushiAI carries this into
`Graph::Attributes`, whose capacity is bounded by `TaskDescription::K_MAX_PARAMETERS`:
eight.

**`pad_bottom` and `pad_right` are deliberately absent.** A kernel addresses its input as `h_in =
h_o * stride_h - pad_top + r * dilation_h` and bounds-checks against `[0, H)`. The trailing pads
never enter that expression; they affect only *how many output rows exist*, which is
`output_extent`'s answer and which the host has already written into the output tensor's shape by
the time any of this reaches a device. So asymmetric padding costs zero slots over symmetric
padding, and every geometry is expressible: non-square kernels, anisotropic stride and dilation, and
independent padding on all four sides.

### `output_extent`

`floor((in + pad_before + pad_after - dilation*(kernel - 1) - 1) / stride) + 1`, which is the same
expression PyTorch, Darknet and cuDNN all use.

A window wider than the padded input answers **0** rather than letting the numerator go negative.
That is not defensiveness: the result is multiplied into an allocation size two layers up, and a
negative numerator under integer division would produce a plausible-looking positive extent for some
operand combinations.

### `WindowSpan`

Modelled on `SushiBLAS::Reduce::AxisSpan`, and for the same reason: it carries counts and strides
only, so the layout, the axis order and any non-unit stride a view inherited are resolved once on
the host and the device sees none of them. One kernel per operation, no layout branch in the inner
loop, no second instantiation.

The struct names no layout. Every caller in the convolutional stack builds it from an NHWC tensor
(see the spec's CV-3), but adding NCHW later is filling in a different stride assignment in
`Spatial::window_span_for`, not re-cutting this abstraction.

All strides are element counts and all are non-negative, which `Tensor`'s stride-explicit
constructor already guarantees.

### `TapSpan`

How the per-channel tap tensor of `window_dot`, `[kernel_h, kernel_w, channels]`, is addressed:
three strides in elements. It is its own struct for the reason `ColumnSpan` is: a rank-3 operand
beside rank-4 ones. `Spatial::tap_span_for` builds it from a tensor.

### `ColumnSpan`

Kept out of `WindowSpan` because the column matrix is rank-2 and everything else in this family is
rank-4. Folding two unrelated addressing schemes into one struct would leave four fields meaningless
for every operation but two.

Entry `(row, element)` lives at `data[row * row_stride + element * element_stride]`, where `row`
indexes a window in `(n, h_o, w_o)` order and `element` indexes a patch position in `(r, s, c)`
order: channel innermost, so an NHWC gather copies contiguous runs of `channels` elements.

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
missing feature. Ops built on them declare Domain::Real or Domain::RealAndIntegral and
throw for complex inputs.

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

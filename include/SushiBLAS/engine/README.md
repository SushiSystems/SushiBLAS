# Engine headers

`include/SushiBLAS/engine` declares the operation families the `Engine` exposes and the shared
machinery under them: the dispatch vocabulary (`dispatch.hpp`), the task recorder (`task.hpp`), the
device policy (`device_policy.hpp`), the per-element launch (`launch_1d.hpp`) and the two reduction
shapes (`reduce_1d.hpp`, `reduce_axis.hpp`). The sub-folders `blas/`, `logic/`, `math/` and
`spatial/` each have a README of their own.

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

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
`sycl::info::device` except `has_dedicated_local_memory`, which is a judgement the raw query cannot
make on its own: see that field.

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
thousands of times and eight `get_info` round trips per node per step is real overhead against a
few-microsecond kernel.

A std::deque, not a vector: a reference handed to an earlier caller must stay valid when a second
device is probed, and only deque guarantees that across push_back.

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

## `launch_1d.hpp`

Every elementwise, non-linear and logic op used to launch with a bare `sycl::range<1>(n)`: one
work-item per element, no work-group size, no vectorisation, and the runtime free to pick whatever
geometry it liked. That is the shape a device is least likely to want. This header replaces it with
three primitives, `Launch::map`, `Launch::zip_inplace` and `Launch::indexed`, that own the geometry
once, so tuning it is one edit rather than fifty-five.

`Launch::map` and `Launch::zip_inplace` serve the kernels whose memory access pattern this header
can see: contiguous, one element per index, so it may vectorise them. `Launch::indexed` serves the
ones whose pattern is their own business: a Level 1 vector walked with a BLAS `incx`, a GEMV output
row that reduces across a whole matrix row, an RNG fill whose value is a function of the index
rather than of any input array. Those get the device split and nothing else, which is the part that
was missing: before this they were the launch sites still spelling `sycl::range<1>(n)` by hand and
therefore still getting the runtime's default geometry on a GPU.

All three launch a **grid-stride** `nd_range` planned by `SushiBLAS::Device::plan_1d:` a grid sized
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
`<SushiBLAS/engine/reduce_1d.hpp>` for that.

- `q`: The queue being submitted to, used only to reach the device profile.
- `op`: Callable `(std::size_t i) -> void`, invoked once per index on device.

Warning: `op` is inlined into a SYCL device kernel: it must be device-copyable, capture by value
only, hold no host pointers, allocate nothing and throw nothing.

## `reduce_1d.hpp`

`launch_1d.hpp` does this job for the per-element families; this is its counterpart for the folding
ones, and it exists for a correctness reason rather than a performance one.

**Why not `sycl::reduction`.** Every additive reduction in this library used to be a one-line
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

### `K_MAX_GROUPS`

Stage two walks the partials sequentially in one task, so the count is a direct latency cost there;
and the scratch buffer that holds them is sized from it at record time. 256 partials is enough to
fill any GPU this project will meet (a 108-SM part at eight resident groups per SM wants 864 groups
for a *saturating* launch, but a reduction is bandwidth-bound and 256 groups of 256 items is already
65,536 work-items in flight), and 256 sequential adds is a rounding error against reading the input
at all.

### `K_CPU_TWO_STAGE_MIN_ELEMENTS`

Both shapes are deterministic, so this threshold is a pure performance choice, but it is a
*measured* one, and it is not the answer that was expected. See `planned_partials` for the numbers.

It does not weaken the reproducibility guarantee. The shape is a function of the device and the
element count only, both fixed for a given call, so two runs of the same reduction always take the
same branch and perform the same additions in the same order. A *different* size taking a different
shape is not a determinism problem, because it is a different computation.

### `floor_pow2`

The halving tree in stage one folds `local / 2`, `local / 4`, ... down to one, which only visits
every element when `local` is a power of two. A non-power-of-two group width would silently drop the
odd items, a wrong answer, not a slow one, so the width is rounded here rather than assumed.

### `planned_partials`

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

A GPU always takes the two-stage shape: there a single task uses one lane of one SM and leaves the
entire machine idle at every size.

Returns: The partial count, or 0 to request the single-pass shape.

`planned_partials` returns 0 when the plan comes to one group. One group is a tree with nothing to
fold across, so it is the single-pass shape wearing a more expensive hat.

### `plan`

`planned_partials` is asked at record time and this is asked at execution time, so in principle they
could see different devices: the scratch is sized from the device the *data* lives on, and the task
is run on the device the scheduler routes it to. Clamping to `scratch_capacity` here means such a
disagreement costs performance and never correctness: too little scratch falls back to the single
pass rather than writing past the buffer.

### `CompensatedSum`

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

### `CompensatedSum::rescale`

For an accumulator whose terms are expressed relative to a scale that can change mid-fold,
`ScaledSumOfSquares` is the one in this library, the total already folded has to move with it. Doing
that here rather than in the caller is what keeps the correction term correct: `comp_` is in the
same units as `sum_`, so rescaling one and not the other would quietly stop compensating anything.

- `factor`: The multiplier to apply. Exact when it is a power of two, which is the only way
  `ScaledSumOfSquares` ever calls this.

### `ScaledSumOfSquares`

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
divide-per-element one, 11.0 ms for this. What is left over the naive form is the per-element
`fabs`, compare and multiply; see `climb_to` for the two more elaborate variants that were tried and
measured *slower*, and for why the remaining inexactness does not accumulate.

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

### `ScaledSumOfSquares::min_scale`

The smallest positive *normal* value, which for every IEEE format is exactly a power of two.
Bounding the scale here is what keeps its reciprocal finite, and it costs nothing at the bottom of
the range: a denormal magnitude scaled up by `1 / min()` lands somewhere its square is comfortably
normal, which is the entire point: squaring it in place would have given zero.

Returns: `2^(min_exponent - 1)`.

### `ScaledSumOfSquares::max_scale`

The reciprocal of `min_scale`, and therefore itself a power of two whose own reciprocal is normal.
In every IEEE format that lands one binade below the top, which is what makes `4 * max_scale()`
overflow to infinity: the property `climb_to` relies on to stop the scale climbing.

Returns: `2^(max_exponent - 2)`.

### `ScaledSumOfSquares::combine`

`hypot`, not `sqrt(a*a + b*b)`, for the same reason this class exists at all: the squares are what
overflow, and `hypot` is specified to compute the result without forming them. It is called once per
tree level and once per work-group partial, never per element, so its cost is nothing.

### `ScaledSumOfSquares::accumulate`

Accepts either a raw element or an already-finished partial norm; the two are the same operation,
which is what lets stage two of `reduce` fold partials with the same accumulator that stage one
folds elements with.

At `if (a == T(0)) return;`: an exact zero contributes nothing, and letting it reach `climb_to`
would send the ladder there all the way to the bottom of the range for no gain.

At `if (a >= hi_) climb_to(a);`: leaving the current binade, not merely beating the running maximum.
`hi_` is `2 * scale_`, so on a well-conditioned vector this fires a handful of times in total rather
than once per element, and a NaN fails it, which is how a NaN reaches the multiply `a * inv_scale_`
unrescaled and propagates the way BLAS says it should.

Until 2026-10-05 the comment on `const T m = a * inv_scale_;` in `accumulate` read: "Exact:
`inv_scale_` is a power of two, and the bounds `climb_to` keeps guarantee the product neither
overflows nor underflows." The second half holds. The first half does not: `climb_to` sets the scale
to the magnitude itself, clamped to `[min_scale, max_scale]`, and takes `T(1) / next` as the
reciprocal, so the product is rounded (see `ScaledSumOfSquares::climb_to`). The source line now
states only the bounds. Member briefs in `reduce_1d.hpp` that still say "power of two" are left for
the owner.

### `ScaledSumOfSquares::climb_to`

**Straight-line on purpose.** Two earlier versions of `climb_to` rounded the scale down to an exact
power of two, so that `ssq_.rescale` and the per-element multiply were both exact. Both were slower
than the whole scaling form needs to be, and by a lot, because both put something in the loop body
that a compiler will not look past: `ldexp(1, ilogb(a))` puts two library calls there, and a
doubling ladder puts two data-dependent loops. Measured over 100 launches of a 1M-element fp32
`nrm2` on this project's OpenCL CPU device: the branch that reaches here is taken a handful of times
in each of those launches:

| form                                   |   best |
|----------------------------------------|-------:|
| naive `sqrt(sum(x*x))` (wrong)          | 5.9 ms |
| scaled, `ldexp`/`ilogb` binade          |14.7 ms |
| scaled, doubling-ladder binade          |21.2 ms |
| scaled, this                            |11.0 ms |

**And exactness was not worth buying.** Rounding to a binade bought an exact `ssq_.rescale`, but
every rescale shrinks the total already folded by at least four, that is what `hi_` guarantees, so a
segment's rounding error is down-weighted exactly as fast as the segment is. The errors do not
accumulate over the ~250 rescales a full-range vector can provoke; only the last few contribute, at
a few ulp. The reciprocal is likewise a single systematic `(1 + eps)` on the whole answer, because
every element in a segment is multiplied by the same rounded `1 / scale`.

- `a`: A magnitude strictly above zero and at or above the current `hi_`. Never a NaN: the `>=` in
  `accumulate` excludes one.

At `T next = a;`: clamped so the reciprocal `inv_next` is a normal number. Neither end costs
anything: a magnitude under the floor still scales to something whose square is normal, and one over
the ceiling still scales to at most 4. An infinity lands exactly on the ceiling, which is what makes
it scale to infinity and saturate the sum rather than produce a NaN.

At `hi_ = (next < ceiling) ? (next + next) : (next * T(4));`: `hi_` is `2 * next` normally. At the
ceiling `4 * next` overflows to infinity by construction (the ceiling is one binade below the top),
which is what stops a vector of huge values from re-entering `climb_to` once per element.

`const T shrink = scale_ * inv_next;` in `climb_to`: `shrink` is zero on the very first value
folded, which correctly clears an empty sum rather than dividing by a scale that does not exist yet.

### `RunningProduct`

No compensation, because there is nothing to compensate: Neumaier recovers the bits an *addition*
rounds away, and multiplication's rounding error is not of that shape. What this shares with
`CompensatedSum`, and the only reason it is spelled the same way, is the fixed combination order,
which is what the determinism argument actually rests on.

### `RunningMax`

**Not what the whole-tensor `max` uses, and that is deliberate.** `ReductionOps::max` stays on
`sycl::reduction`, because `maximum` is exactly associative and exactly commutative, so every
combination order yields the same value and a hand-written fixed-order tree there would buy nothing
but slower code: the argument is written out in `max.cpp`. This class exists because the *segmented*
seam (`Dispatch::execute_folding_axis_reduction`) is one shape serving every axis reduction, and a
shape needs an accumulator. Order-independence means that costs nothing here either: the fold is
fixed anyway, so a per-axis maximum is bit-reproducible for a stronger reason than the additive
reductions are.

**Selection matches `sycl::maximum` exactly**, down to which operand survives: both keep the left
one unless the right compares greater, so a per-axis maximum and a whole-tensor maximum cannot
disagree on the same data. That carries over `max.cpp`'s one caveat as well: every comparison
against a NaN is false, so a NaN in the input does not propagate and which value survives meeting
one depends on the order. No reduction shape fixes that; only screening the input does.

### `RunningMax::identity`

The lowest finite value of `T`, so folding it in against anything representable is a no-op. Not
`constexpr`, because `std::numeric_limits<sycl::half>::lowest()` is not: the same reason
`ScaledSumOfSquares::min_scale` is a function.

### `ExactSum`

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

### `reduce`

Computes `finalize(fold(transform(in[i])), n)`, where `fold` is `Accum`'s operation applied in an
order fixed entirely by the launch geometry. Submits one kernel on the single-pass shape and two on
the two-stage one; see `plan` for which, and the file header for why the order is nailed down.

**The accumulation type is separate from the element type**, and every existing caller deduces them
equal without saying so. `sum`, `mean`, `asum`, `nrm2` and `prod` all fold a `T` into a `T`, which
is why this was one parameter for as long as there was nothing else to do, but a reduction whose
transform answers a *predicate* folds a count, and a count is not an element. Storing one in the
input's own dtype is the trap `Dispatch::execute_index_reduction` was cut out to escape: a count is
exact only below 2^24 in fp32 and only below 2048 in fp16, so a sufficiently bad tensor would report
a rounded number of bad elements, and in fp16 a count past 65504 would come back as an infinity.
Splitting `A` off costs the existing callers nothing (both are deduced, and both deduce to `T`) and
lets the counting reductions share this shape rather than hand-write a second copy of a two-stage
fold.

- `Accum`: The accumulation policy, `CompensatedSum`, `RunningProduct` or `ExactSum`. A class
  template on the *accumulation* type, supplying `identity()`, `combine(a, b)`, `accumulate(x)` and
  `value()`.
- `T`: The input element type, deduced from `in`.
- `A`: The accumulation and result type, deduced from `out` and `partials`. Equal to `T` for every
  reduction whose answer is a value of the thing it reduced.
- `partials`: Scratch for one value per work-group; may be null when `partial_capacity` is zero.
- `transform`: Callable `(T) -> A`, applied to each element before it is folded in. `x * x` for a
  2-norm, `fabs(x)` for an absolute sum, `isfinite(x) ? 0 : 1` for a count of the elements that are
  not finite.
- `finalize`: Callable `(A value, std::size_t n) -> A`, applied once to the folded result. `sqrt`
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

At `Accum<A> acc;`: A grid-strided slice: work-item g owns indices g, g + stride, g + 2*stride, ...
Which indices those are, and the order they are visited in, come only from the geometry, so they are
the same on every run. Consecutive items also read consecutive addresses at every step, which is
what keeps the loads coalesced.

At `for (std::size_t off = local / 2; off > 0; off >>= 1)`: A hand-written halving tree, and NOT
sycl::reduce_over_group. The spec permits that function to combine in any order it likes, which for
a non-associative operator means it may return different bits on different launches of the same
kernel. This loop combines (lid, lid + off) for a fixed descending sequence of offsets, so the tree
is the same tree every time. `local` is a power of two by construction (see floor_pow2), which is
what makes the halving visit every lane exactly once.

At `Accum<A> acc;`: ascending index order over a count fixed at launch. Compensated as well, which
costs nothing at this length and recovers the spread between partials that a plain walk would drop.

### `CompensatedSum::accumulate`

The correction step in `CompensatedSum::accumulate`, with `t` the rounded sum: whichever operand is
larger keeps its bits. The smaller one is the one that got rounded off, and `(larger - t) + smaller`
recovers exactly the part that did not fit.

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
paying `log2(local)` pairs of barriers, so the tree only earns its
keep once every lane has at least one element to fold. One full
group-width pass is that threshold.
- **Otherwise** there is nothing to recover; take the cheap shape.

Warning: The GPU side of this is **derived, not measured**: this project has no GPU (see
`docs/design/GPU_VALIDATION.md`). The threshold is a floor below which the cooperative shape is
certainly wrong, not a tuned crossover, and it is the first knob to reach for if a row-wise
reduction underperforms on real hardware. It cannot make an answer wrong: the shape is a pure
function of the device and the span, so it is the same on every run, and both shapes compute the
same thing.

Returns: The geometry; `AxisPlan::cooperative` says which shape.

### `reduce_axis`

Computes `out[s] = finalize(fold(transform(in[s][k])), extent)` for every segment `s`, where `fold`
is `Accum`'s operation applied in an order fixed entirely by the launch geometry. One kernel
submission, whichever shape `plan_axis` chose.

A segment shorter than the work-group is still correct on the cooperative shape: a lane with no
element of its own contributes `Accum<T>{}.value()`, which is the accumulator's identity by
construction, and combining the identity changes nothing. `plan_axis` declines that shape below one
full pass for cost, not correctness.

- `Accum`: The accumulation policy (`CompensatedSum`, `RunningMax`, `RunningProduct`), a class
  template on the element type supplying `identity()`, `combine(a, b)`, `accumulate(x)` and
  `value()`.
- `span`: The segment layout, in element counts.
- `in`: The input array; fold `s` starts at `in + (s / segments) * segment_stride2 + (s % segments)
  * segment_stride`, which is `in + s * segment_stride` whenever there is no second segment axis.
- `out`: Where the per-segment results go.
- `geometry`: Which shape to launch, normally `plan_axis`'s answer. Taken rather than decided here
  so that the planner and the launcher stay separable: a test can then drive both shapes over one
  input and check they agree, which is the only way the cooperative tree gets exercised at a size
  the planner would not send to it.
- `transform`: Callable `(T) -> T`, applied to each element before it is folded in. Pass the
  identity when there is nothing to do.
- `finalize`: Callable `(T value, std::size_t n) -> T`, applied once per segment, with `n` the
  segment's own element count, so a per-axis mean is this seam plus `v / n` rather than a second
  kernel.
- `deps`: Events this submission must depend on.

Returns: The event for the submitted kernel.

Warning: `transform` and `finalize` are inlined into a SYCL device kernel: they must be
device-copyable, capture by value, hold no host pointers, allocate nothing and throw nothing.

At `const Device::Grid1D grid = Device::plan_1d(Device::profile_for(queue), total);`: one item per
segment, over the same flat/grid-stride launch shape every elementwise op takes, so a CPU covers the
segments in one pass of a wide group and a GPU walks a capped, resident grid. The stride loop's `s <
total` test is the bounds check.

At `for (std::size_t off = local / 2; off > 0; off >>= 1)`: the same hand-written halving tree as
stage one of Reduce::reduce, and NOT sycl::reduce_over_group, for the reason recorded there: the
spec lets that function combine in any order it likes, which for a non-associative operator means
different bits on different launches of one kernel. `local` is a power of two (see
Reduce::group_width), which is what makes the halving visit every lane exactly once.

The flat branch of `reduce_axis` folds each segment in ascending index order, and no other work-item
touches it. There is nothing in that loop whose order the SYCL specification leaves free.

`const std::size_t s2 = s / segments;`, in both branches of `reduce_axis`: one integer division per
fold, against an inner loop of `extent` elements. It is not worth a specialised `segments2 == 1`
kernel, which would be a second instantiation of every reduction in the library.

The lane loop in the work-group branch of `reduce_axis` walks a stride-`local` slice of one segment:
lane `l` owns positions `l`, `l + local`, `l + 2*local`, and so on. Which positions those are comes
only from the geometry, so it is the same on every run, and consecutive lanes read consecutive
addresses at every step.

## `spatial.hpp`

The word "convolution" does not appear in this domain and should not be added to it. `im2col` is a
gather, `pool_max` is a windowed reduction, `channel_affine` is a broadcast elementwise. Parameter
ownership, training mode and differentiation belong to the layer above; what belongs here is moving
numbers around a grid.

Every operation in this domain has both a pure-C++ reference kernel and a SYCL one, and the
reference is the specification. The two are held to each other by this domain's unit tests, which
run both forced modes into separate buffers and compare, not by `ExecutionMode::VERIFY`, which runs
both but lets the reference overwrite the device's answer rather than checking it.

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
about it, the word "BLAS" in its diagnostics, lifted into a parameter. Nothing else changed, and
`execute_blas_task` still exists and still says "BLAS operation"; it forwards here.

The move happened because a second domain arrived. `spatial.im2col` reported as a "BLAS operation"
when its dtype is refused sends a reader to the wrong half of the library, and the alternative, a
second copy of the seven-arm dtype switch, is seven arms to forget when a `DataType` is added.

The include list is `blas/utils.hpp`'s, minus what only the vector helper needed.
`<SushiBLAS/engine.hpp>` is included outright rather than forward-declared: both templates in
`task.hpp` need `Engine` complete (one calls `get_graph()`, the other compares against
`ExecutionMode`), and a forward declaration that only works because every includer happens to have
pulled `engine.hpp` first is a trap rather than a decoupling.

### `execute_typed_task`

One `add_task`, one `switch (dtype)` dispatching a generically-templated closure. An unsupported
dtype fails loudly at record time (P1 corollary), never silently no-ops.

- `Dom`: Which element types to instantiate `op_func` for, in exactly the sense `Dispatch::Domain`
  defines for the math and logic families. It defaults to REAL_AND_COMPLEX because that is what
  every built-in BLAS *arithmetic* op wants; the parameter exists for callers whose functor is
  written outside this library and is therefore no more likely to be well-formed for `std::complex`
  than an elementwise lambda is, and for movement ops like `copy` and the whole spatial family,
  which relocate elements without doing arithmetic to them and so pass `Domain::ALL`.
- `family`: What kind of operation this is, for diagnostics: "BLAS operation", "Spatial operation".
- `op_func`: A callable `(auto scalar_type, sycl::queue&, const std::vector<sycl::event>&) ->
  sycl::event`, generic over the runtime dtype via `decltype(scalar_type)`.

Returns: An empty sycl::event; the work is deferred until `Engine::execute()`.

Throws `std::runtime_error` at record time if `dtype` falls outside `Dom`.

At `switch (dtype)`: spelled out rather than routed through Dispatch::with_element_type, which is
what the elementwise, non-linear and logic families use. Two reasons, and the second is the binding
one. `op_func` takes a *value* of the element type (`decltype(scalar_type)`) alongside the queue and
the dependency list and returns the submitted event, which is not the one-argument, void-returning
arm that helper calls; and wrapping it to fit, a nested generic lambda, inside this lambda, inside
this template, crashes this project's clang outright (a null-dereference during instantiation of the
HALF arm, under -fdelayed-template-parsing). Seven arms here, the same seven and the same domain
rule as with_element_type, kept in step by hand until that compiler bug is gone.

The `SB_THROW_IF(true, ...)` that ends the dtype switch in `execute_typed_task` is unreachable in
practice, because `require_domain` rejected the dtype at record time. A switch that can fall through
to nothing is how a silent no-op gets in, so it ends in a throw rather than in a default return.

### `dispatch_with_mode`

Collapses the FORCE_REFERENCE / normal / VERIFY execution-mode triad that ops used to hand-write at
every call site (5x in gemm.cpp, 5x in syrk.cpp).

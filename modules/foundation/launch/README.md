# launch

The device policy a kernel dispatches on and the per-element launch shapes built on it: `Device::Profile`, the launch geometry, and `Launch::map`, `Launch::zip_inplace` and `Launch::indexed`. Its headers are included as `<SushiBLAS/launch/<name>.hpp>`.

It depends on [`core`](../core/README.md), SYCL and the standard library.

| File | Declares |
| --- | --- |
| `device_policy.hpp` | `Profile`, `Vendor`, `Architecture`, `CacheBlocking` and the launch geometry a kernel dispatches on |
| `gemm_tile_pin.hpp` | `gemm_tile_pin`, the read of `SB_GEMM_TILE` |
| `launch_1d.hpp` | The launch primitives every per-element kernel shares: `map`, `zip_inplace`, `indexed` |
| `submitted_events.hpp` | `SubmittedEvents`, the events of every kernel one task body submitted |
| `kernel_variant.hpp` | `KernelVariant`, `note_kernel_variant` and the observer a host sets to read the notes |
| `launch_log_category.hpp` | `K_LAUNCH_LOG_CATEGORY`, the category of the probe's records |

Each section below holds the reasoning behind a header, under the name of the file and of the symbol it belongs to. The header states the rule in one line and cites this page.

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

### `Profile::supports_half_matrix`

`Detail::offers_half_matrix` reads `matrix_combinations` from the oneAPI matrix
extension and looks for an entry of 16 x 16 x 16 with HALF A and B and FLOAT32 C and D. It asks a
CUDA device only and answers false for every other backend: the kernel that uses the field is
compiled for NVPTX alone, so a true answer from an Intel or OpenCL device would route to a kernel
with no body there. Asking only CUDA also keeps the query off devices that do not implement it.
The field is a bool, so no public header names an extension type.

### `Vendor`, `Architecture` and `Profile::local_mem_per_compute_unit`

`Detail::vendor_of` maps the PCI vendor id the runtime reports (0x10DE, 0x1002, 0x8086) to
`Vendor`. `Detail::architecture_of` reads `sycl::ext::oneapi::experimental::info::device::architecture`
behind `SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE` and maps nine values (`nvidia_gpu_sm_70` to
`nvidia_gpu_sm_90`, `amd_gpu_gfx90a`, `amd_gpu_gfx942`, `intel_gpu_pvc`) to `Architecture`. Any
other value, a runtime without the extension, and a device whose query throws a `sycl::exception`
give `Architecture::Unknown`. A tuning table keyed on `Architecture` therefore has no entry for a
device it was not measured on, and the caller falls back to a model.

`local_mem_per_compute_unit_of(architecture, local_mem_bytes)` is a closed table of what one
compute unit holds, because `local_mem_size` reports what one work-group may take, which is less
on NVIDIA: 96 KiB for sm_70, 64 KiB for sm_75, 164 KiB for sm_80, 100 KiB for sm_86 and sm_89,
228 KiB for sm_90, 64 KiB for gfx90a and gfx942, 128 KiB for PVC. `Unknown` returns the reported
`local_mem_bytes`. The switch is closed by its meaning and small, so it stays a switch.

## `gemm_tile_pin.hpp`

`gemm_tile_pin()` returns the value of `SB_GEMM_TILE` read once per process, empty when the
variable is unset. It holds a name and knows no shape: the module that owns the shape catalog
validates it. `gemm_tile_pin_from(const char*)` is the pure part, so a test covers it without
touching the environment. The Windows runtime marks `getenv` deprecated in favour of
an allocating variant; the one call is wrapped in a local diagnostic push and pop rather than
a project-wide `_CRT_SECURE_NO_WARNINGS`, because the pointer is copied into a `std::string` at
once and read before any thread can write the environment.

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

Each thread keeps a `thread_local` pointer to the entry it was last handed. A call for the same
device compares one handle and returns without the mutex or the scan; any other call takes the
lock, finds or probes the entry, and repoints. The pointer is safe to follow unlocked because an
entry is written once, under the lock, before any thread can point at it, and a deque never moves
an element it holds. The pointer is a raw pointer, so no thread-exit destructor touches a
`sycl::device`.

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
`<SushiBLAS/reduction/reduce_1d.hpp>` for that.

- `q`: The queue being submitted to, used only to reach the device profile.
- `op`: Callable `(std::size_t i) -> void`, invoked once per index on device.

Warning: `op` is inlined into a SYCL device kernel: it must be device-copyable, capture by value
only, hold no host pointers, allocate nothing and throw nothing.

## `submitted_events.hpp`

A task body that submits one kernel returns that kernel's `sycl::event`, and the runtime times the dispatch from it. A body that submits several returns a `SubmittedEvents`, and the recorder hands every event to the runtime (`SushiRuntime::note_dispatch_event`), so the dispatch runs from the first kernel's start to the last kernel's end. Without it the span would be the last kernel's alone, and the FLOPs of the whole body over that span would state a rate that never happened.

The type converts to the last event, and has `wait()` and `wait_and_throw()` that wait on every event, so a caller that orders against one event, and a test that waits on a result, take it unchanged. The kernels that return it are the ones that submit more than once in a body, and each lists its events in submission order; [the cost README](../cost/README.md) lists them.

## `kernel_variant.hpp`

A kernel that chooses among several variants (the GEMM) calls `note_kernel_variant(variant)` before it launches. With no observer set the call is an atomic load and a branch. The engine sets one on construction that writes the profiler instant of the variant, so no kernel header includes a profiler header and a change of SushiProfiler recompiles no kernel.

## The probe's records

`Detail::probe` runs once per device, on its first sight under the cache's lock, so what it logs is once per device. It logs under `K_LAUNCH_LOG_CATEGORY`:

| Level | When | Says |
| --- | --- | --- |
| `WARN` | The architecture query threw | The device, the exception, and that the tile limits take the generic path |
| `WARN` | The device lists no sub-group size, or only 0 | The device and the width assumed, 1 |
| `WARN` | The device reports no cache size | The device and the size assumed, `K_ASSUMED_CACHE_BYTES` |
| `INFO` | Every probe | The device, its compute units, sub-group width, cache size and architecture, each assumed value marked |

`resolve_sub_group_size` and `resolve_cache_bytes` make the two decisions as pure functions, so they are tested without a device that lacks either.

## Designs

<!-- designs:begin -->
- [GPU_VALIDATION](../../../docs/design/GPU_VALIDATION/README.md): Several changes in this repository and in SushiAI were made for a GPU, justified by argument and by arithmetic, and checked only as far as a CPU can check them.
<!-- designs:end -->

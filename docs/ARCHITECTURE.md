# SushiBLAS Architecture

This document describes how SushiBLAS is built, layer by layer. It is written
for contributors who need an accurate mental model of the library — every
class and file named here exists under `include/SushiBLAS/`, `src/`, or
`kernels/`.

> If you only want to *use* the library, the top-level [README.md](../README.md)
> is enough. This document is about how the inside works.

---

## 1. What SushiBLAS is

SushiBLAS is a C++17 tensor and BLAS library layered on top of
[SushiRuntime](https://github.com/SushiSystems/SushiRuntime)'s task graph. It
owns no execution engine, no scheduler, and no memory pool of its own — every
one of those is SushiRuntime's. What SushiBLAS adds is a **domain layer**: a
`Tensor` abstraction, an `Engine` facade that turns method calls into graph
tasks, and a library of kernels (BLAS levels 1-3, elementwise arithmetic,
non-linear activations, logical/comparison ops, reductions, and random
generation) with zero dependency on a vendor math library.

Two design commitments shape every operation in the library:

1. **Every op is a SushiRuntime task, not a blocking call.** Calling
   `engine.blas().gemm(A, B, C)` registers a task on the engine's `TaskGraph`
   with `A`/`B` as reads and `C` as a write; it does not run anything.
   SushiRuntime's dependency tracker infers ordering against every other
   queued task from those reads/writes, exactly as it would for a hand-written
   `TaskGraph::add_task` call. `engine.execute()` compiles and runs the graph.
2. **No vendor math library, anywhere.** Every kernel is either pure C++
   (`kernels/reference/`) or SushiBLAS's own SYCL (`kernels/sycl/`). GEMM and
   SYRK use a local-memory-tiled kernel derived from portBLAS's tiling
   algorithm (Apache-2.0, attributed in [`NOTICE.md`](../NOTICE.md)); every
   other operation — all of BLAS levels 1/2/3-TRSM, every elementwise/
   nonlinear/logic/reduction op, and the Philox4x32-10 random backend — is
   original SushiBLAS SYCL, written from scratch.

The codebase targets C++17 — the same standard SushiRuntime is built to, and for
a reason beyond tidiness. `SushiRuntime::span` (`core/span.hpp`) is `std::span` at
C++20 and SushiRuntime's own shim below it, chosen on `__cplusplus`, and that choice
reaches the compiled ABI. SushiBLAS at C++20 linked against a SushiRuntime at C++17
put the two halves of one build on opposite sides of that branch. Nothing in
SushiBLAS's own API changed when it moved: the two spans are the same type wherever
`std::span` exists, and every span-taking entry point in these headers is **inline**,
so the branch is re-decided in the consumer's own translation unit and never has to
agree with a symbol somebody else compiled. That is what lets a consumer still on
C++20 compile *and link* against this library unchanged, and `tests/package/` is
built at C++20 to keep proving it. (SushiAI is at C++17 now and does not have that
property — its span-taking exports are out-of-line, so the type reaches the mangled
name; see its `INTEGRATION.md` §3.) The pieces C++20 had been supplying are C++17
stand-ins now: `core/format.hpp` for `std::format` — shared with SushiAI through the
public `Core::format`, so the stack carries one copy and deletes it in one place —
`Core::Detail::SourceLocation` for `std::source_location`, and a value tag
(`Dispatch::ElementTag`) where the dispatch seam used a templated lambda.

---

## 2. The layered architecture

```
   ┌──────────────────────────────────────────────────────────────┐
   │  Engine                 create_tensor() · set_mode()         │  facade: owns the
   │  (engine.hpp)           execute() · blas()/elementwise()/... │  RuntimeContext,
   │                                                                │  TaskGraph, seed/RNG
   ├──────────────────────────────────────────────────────────────┤
   │  Per-domain op views    BLASOps(Level1+Level2+Level3+Sparse)  │  lightweight views —
   │  (engine/*.hpp)         ElementwiseOps · NonLinearOps         │  no state of their
   │                          LogicOps · ReductionOps · RandomOps  │  own, one Engine& each
   ├──────────────────────────────────────────────────────────────┤
   │  Dispatch helpers       execute_blas_task()  — runtime-dtype  │  engine/blas/utils.hpp
   │                          switch over one add_task              │
   │                         dispatch_with_mode() — the             │
   │                          AUTO/FORCE_SYCL/FORCE_REFERENCE/      │
   │                          VERIFY triad                          │
   ├──────────────────────┬───────────────────────────────────────┤
   │  kernels/reference/   │  kernels/sycl/                        │  the two kernels
   │  pure C++, ground     │  the hardware dispatcher, returns     │  every op has
   │  truth, no deps       │  sycl::event                          │
   ├──────────────────────┴───────────────────────────────────────┤
   │  Tensor / Storage      shape+stride+dtype view over a          │  tensor.hpp,
   │                        ref-counted USM allocation              │  storage.hpp
   ├──────────────────────────────────────────────────────────────┤
   │  SushiRuntime           TaskGraph · RuntimeContext ·           │  the backbone —
   │                          IAllocatorStrategy · Execution engine │  everything below
   │                                                                 │  this line is not
   │                                                                 │  SushiBLAS's code
   └──────────────────────────────────────────────────────────────┘
```

Higher layers depend on lower ones; lower layers never depend up. SushiBLAS
itself contributes only the top four boxes — the bottom box is a
`SushiRuntime::SushiRuntime` link, resolved per [§8](#8-the-sushiruntime-dependency-d3).

---

## 3. `ExecutionMode`: the dispatch triad

`Engine::set_mode(ExecutionMode)` (`engine.hpp`) switches every subsequent
operation between four modes:

- **`AUTO`** — the default. Runs the SYCL kernel.
- **`FORCE_SYCL`** — always the SYCL kernel (currently identical to `AUTO`;
  the distinction exists for a future heuristic dispatcher that could pick a
  different backend under `AUTO`).
- **`FORCE_REFERENCE`** — always the pure-C++ reference kernel, run
  synchronously on the host after waiting on the task's dependencies. For
  debugging a suspected SYCL-kernel bug without touching a device at all.
- **`VERIFY`** — runs the SYCL kernel, waits for it, then runs the reference
  kernel and (by convention in the test suite) compares the two. For
  confirming a SYCL kernel agrees with its ground truth.

The triad is implemented once, centrally, in
`Internal::dispatch_with_mode()` (`engine/blas/utils.hpp`):

```cpp
template<typename ReferenceFn, typename SyclFn>
sycl::event dispatch_with_mode(ExecutionMode mode, sycl::queue& q,
                                const std::vector<sycl::event>& deps,
                                ReferenceFn&& reference_fn, SyclFn&& sycl_fn)
{
    if (mode == ExecutionMode::FORCE_REFERENCE)
    {
        sycl::event::wait(deps);
        reference_fn();
        return sycl::event();
    }

    auto ev = sycl_fn(q, deps);

    if (mode == ExecutionMode::VERIFY)
    {
        ev.wait();
        reference_fn();
    }
    return ev;
}
```

Every BLAS-level op's `.cpp` file calls this once per dtype branch instead of
hand-writing the triad — before `utils.hpp` existed, `gemm.cpp` and
`syrk.cpp` each wrote out all three branches five times (once per `DataType`).
Non-BLAS domains (elementwise, nonlinear, logic, reductions, random) follow
the same pattern inline in their own `SushiBLAS::Dispatch` helpers under
`include/SushiBLAS/engine/**/dispatch.hpp`.

---

## 4. `Tensor` and `Storage`

A `Tensor` (`tensor.hpp`) is metadata, not data: shape, strides, rank,
`Core::DataType`, `Core::Layout`, and a `storage_offset` into a shared,
ref-counted `Storage` (`storage.hpp`). Two tensors can point at the same
`Storage` at different offsets — that's what makes `transpose()`, `reshape()`,
and `slice()` zero-copy: each returns a new `Tensor` with adjusted
shape/strides/offset over the *same* `SushiRuntime::sushi_ptr<Storage>`.

A view that is **not** contiguous — a `slice()` along an inner axis — keeps its
parent's strides, and is built through the stride-explicit `Tensor` constructor
rather than the shape-derived one. `Tensor::element_span()` is what the capacity
check and `region()` measure such a view by: `1 + sum((shape[i] - 1) *
strides[i])`, the distance from its first element to its last, which for a
contiguous tensor equals `num_elements` and for a strided one does not.

```cpp
struct Tensor
{
    std::array<int64_t, MAX_TENSOR_RANK> shape{};
    std::array<int64_t, MAX_TENSOR_RANK> strides{};
    int32_t rank = 0;
    Core::DataType dtype = Core::DataType::FLOAT32;
    Core::Layout layout = Core::Layout::ROW_MAJOR;
    int64_t num_elements = 0;
    SushiRuntime::sushi_ptr<Storage> storage;
    int64_t storage_offset = 0;
};
```

`Core::DataType` is one of `HALF`, `FLOAT32`, `FLOAT64`, `COMPLEX32`,
`COMPLEX64`, `INT32`, `INT64` (`core/common.hpp`); `MAX_TENSOR_RANK` is 6;
`Core::bytes_per_element()` is the single dtype-to-width mapping every size
calculation goes through.

**The integer types are not general-purpose arithmetic types.** They exist
because some results are indices rather than numbers: `iamax`, `argmax` and
`argmin` all answer "which element", and answering that into a float tensor
meant either writing eight bytes into a four-byte element (`iamax`, which
corrupted whatever a memory planner had packed next to it, and under-declared
the write in `region()` so nothing ordered against the corruption) or rounding
the index through a mantissa (`argmax`/`argmin`, exact only below 2^24). They
are also the honest element type for a comparison result. So exactly one list of
operations accepts them — `copy`, `constant`, the comparison and logic family,
the index reductions, and `io` — and everything else rejects them by name at
record time. See `Dispatch::Domain` in [§6](#6-dispatch-the-generic-op-seam).
`is_contiguous()` and `is_aligned()` (checked against
`SushiRuntime::Core::DEFAULT_ALIGNMENT`) are used by the BLAS layer to decide
whether a vector operand can be treated as a flat, unit-stride buffer.

**Layout and the batch axis.** `Core::Layout` (`ROW_MAJOR` / `COLUMN_MAJOR`) is
fixed per-tensor at construction and describes the memory order **inside one
matrix**, not across the whole tensor. `Tensor::layout_strides()` is the single
definition — the constructor lays tensors out with it and `is_contiguous()`
checks against it, so the two cannot drift. For rank ≤ 2 it is the familiar
rule (row-major strides grow from the last dimension inward, column-major from
the first). For rank ≥ 3 the trailing two axes are the matrix and take the
layout's order, while every leading batch axis is outermost and slowest-varying
**in either layout** — a rank-3 `{b, m, n}` tensor has batch stride `m * n`
whichever layout it carries. That is the only reading consistent with how the
batch stride is consumed (`A.strides[rank - 3]` is handed to the GEMM/SYRK/TRSM
drivers as the distance between slices) and with every batched BLAS API, which
pairs a `stride_a` with a per-slice `lda`.

**Offsets are element counts.** `storage_offset` is measured in elements, never
bytes; `data()` scales it by `bytes_per_element(dtype)`. A view is therefore
always naturally aligned for its own element type, since the allocation base is
64-byte aligned and the offset is a whole number of elements.

`Storage` (`storage.hpp`) is a `SushiRuntime::Core::RefCounted`, 64-byte
aligned, holding the allocation's base pointer (private; reachable only through
`allocation_base()`), its allocated `size_bytes` (rounded up to
`DEFAULT_ALIGNMENT`), and the `SushiRuntime::Memory::USMAllocator` +
`IAllocatorStrategy` (`Strategies::Shared` / `Device` / `Host`) it was
allocated with. Being `RefCounted` (via `sushi_ptr`, SushiRuntime's intrusive
atomic-refcount smart pointer) is what lets two `Tensor` views share one
allocation safely — the storage is freed only when the last `Tensor`
referencing it is destroyed.

The base pointer is private on purpose. Most op families used to capture
`tensor.storage->data_ptr` directly and so ignored `storage_offset` entirely: a
view at a non-zero offset silently aliased the base of its allocation. Three
accessors, all on `Tensor`, are what an operation may use instead, and between
them they cover every case a driver has:

| Accessor | Gives | Used by |
|----------|-------|---------|
| `data()` / `data_as<T>()` | the view's first element | every kernel pointer |
| `data_or_null()` | the same, or `nullptr` with no storage | drivers tolerating a storage-less tensor |
| `region()` | the view's byte interval, keyed on the allocation base | every `reads`/`writes` entry |

`region()` is what makes an arena work: two disjoint views of one allocation
produce non-overlapping `ResourceRegion`s, so the dependency tracker does not
serialize them, while overlapping views still order correctly.

---

## 5. `Engine`

`Engine` (`engine.hpp`) is the facade every program constructs once:

```cpp
Engine(SushiRuntime::Execution::RuntimeContext& ctx,
       Core::Layout layout = Core::Layout::ROW_MAJOR);
```

It owns a `SushiRuntime::Graph::TaskGraph` (constructed against the caller's
`RuntimeContext`), the default tensor layout, the current `ExecutionMode`, and
a random-number seed + atomic offset counter (`get_and_increment_rng_offset()`
— every distribution draw gets a fresh, non-overlapping stream position, even
under concurrent task submission). `set_seed()` rewinds that counter as well as
setting the seed: the kernels mix `(seed, offset)` into Philox, so the seed
alone does not name a stream position, and re-seeding has to reproduce the
stream or it is not a seed.

- `create_tensor(dims[, dtype], strat)` allocates a `Storage` through the
  given `IAllocatorStrategy` (default `Shared`) and wraps it in a `Tensor`.
- `blas()` / `elementwise()` / `nonlinear()` / `logic()` / `reductions()` /
  `random()` / `io()` each return a small view object constructed with
  `*this` — none of them hold state beyond the `Engine&` reference, so calling
  `engine.blas()` twice is free.
- `execute()` compiles the graph and runs it via
  `context_.get_execution_engine().execute(compiled)`. It treats a graph with
  no queued tasks as an already-complete no-op (returning a default
  `sycl::event`) rather than calling into `IExecutionEngine::execute()`, which
  throws on an empty root set — several `io.cpp` code paths call `execute()`
  defensively (flush any pending device write) even when nothing was queued.

`Engine` is deliberately thin: it has no idea what "GEMM" or "ReLU" means.
Every operation lives in a domain class (`BLASOps`, `ElementwiseOps`, ...)
that takes an `Engine&` in its constructor and reaches back into
`engine.get_graph()` / `engine.get_context()` to do its work.

---

## 6. The dispatch helpers (`engine/blas/utils.hpp`)

Two templates centralize what every BLAS-level `.cpp` file needs:

**`execute_blas_task`** registers one task whose body switches on the
tensor's *runtime* `Core::DataType` and calls a generic, templated closure
with the matching compile-time scalar type:

```cpp
template<typename Func>
sycl::event execute_blas_task(Engine& engine, const TaskMetadata& meta,
                               const std::vector<ResourceRegion>& reads,
                               const std::vector<ResourceRegion>& writes,
                               Core::DataType dtype, Func&& op_func);
```

`op_func` is called as `op_func(scalar_tag{}, queue, deps)`, where
`scalar_tag` is one of `sycl::half{}`, `float{}`, `double{}`,
`std::complex<float>{}`, `std::complex<double>{}`, `int32_t{}`, `int64_t{}` — so
one `op_func` lambda, templated on its first parameter's type via `decltype`,
implements every dtype without a hand-written switch at the call site. An
unsupported dtype throws (`SB_THROW_IF`) rather than silently no-opping —
matching every other "unsupported case" in the codebase (see
[CONTRIBUTING.md](CONTRIBUTING.md)'s error-handling convention).

`execute_blas_task` also takes a leading `Dispatch::Domain` template parameter,
defaulting to `REAL_AND_COMPLEX` — which is what every built-in BLAS *arithmetic*
op wants and has always had. It exists for two callers. `Dispatch::execute_gemm`,
whose functor is written by the *caller*: a real-only epilogue (`value > 0` has no
meaning for `std::complex`) would otherwise be a compile error in the two arms
that can never run. And `copy`, which passes `Domain::ALL` because it moves
elements without doing arithmetic to them, and so is the one BLAS op that serves
the integer dtypes. Same vocabulary, same meaning, as the math and logic
families' `Domain`.

**`dispatch_with_mode`** is the `ExecutionMode` triad from [§3](#3-executionmode-the-dispatch-triad).

Together, a BLAS op's `.cpp` file is typically one `execute_blas_task` call
whose `op_func` body is one `dispatch_with_mode` call per dtype branch,
picking the matching `kernels::Reference::<op><T>` and
`kernels::SYCL::<op><T>` function. This is what let the level1/level2/level3
BLAS ports (17 ops, zero oneMKL) land as from-scratch kernels without
duplicating the graph-registration or mode-dispatch boilerplate 17 times.

---

## 7. Kernels: reference and SYCL

Every operation has exactly two kernel implementations:

- **`kernels/reference/<category>/<op>.hpp`** — pure C++, templated on the
  scalar type, zero external dependencies. This is the mathematical ground
  truth `VERIFY` mode checks against and `FORCE_REFERENCE` mode runs directly.
  It must handle both `Layout` values.
- **`kernels/sycl/<category>/<op>.hpp`** — the SYCL kernel. Takes a
  `sycl::queue&` and a `const std::vector<sycl::event>& deps` it must
  `depends_on(...)`, and returns the `sycl::event` for the submitted work.

`<category>` today is `blas/{level1,level2,level3}` and `random` (the Philox
PRNG backend, `kernels/sycl/random/philox.hpp`). The non-BLAS math domains
(elementwise, nonlinear, logic, reductions) keep their kernels inline in
`include/SushiBLAS/engine/math/*/dispatch.hpp` and
`include/SushiBLAS/engine/logic/dispatch.hpp` rather than under `kernels/`,
since each op there is a short generic lambda rather than a multi-case
tiling kernel — the reference/SYCL split is the same, just colocated with the
dispatch code instead of factored out.

**GEMM's two kernels are the exception to `kernels/` being private.** They live
under `include/SushiBLAS/kernels/{reference,sycl}/blas/level3/gemm.hpp`,
mirroring the private tree's own path, and they are installed — because
`Dispatch::execute_gemm` instantiates them in a *consumer's* translation unit
with the consumer's own epilogue functor, and a template a public seam
instantiates cannot stay private. The rule is exactly that narrow: a kernel a
public seam names moves under `include/SushiBLAS/kernels/`; everything else
stays in `kernels/`. The epilogue contract itself
(`Kernels::GemmIdentityEpilogue`) sits in
`include/SushiBLAS/kernels/blas/level3/gemm_epilogue.hpp`, under neither
`reference/` nor `sycl/`, because both kernels and the seam name it.

Those six headers are public API, not implementation detail: `execute_binary`,
`execute_unary_inplace`, `execute_zip_inplace`, `execute_nonlinear_forward` /
`_backward`, `execute_logic_unary` / `_binary` / `_ternary`, `execute_reduction`,
`execute_folding_reduction`, `execute_folding_axis_reduction`, `execute_random`
and `execute_gemm` all live in
`namespace SushiBLAS::Dispatch` and are installed. A downstream framework
composes several math ops into one functor, hands it to one `execute_*` call,
and gets a single fused SYCL kernel — the reason the seam exists, and the reason
"the consumer never writes SYCL itself" stays true. See
`include/SushiBLAS/engine/dispatch.hpp` for the shared vocabulary (`Domain`) and
the constraints a device-inlined functor has to satisfy.

**`Domain` is a bit set over element-type families**, not a list of named
combinations: `REAL`, `COMPLEX`, `INTEGRAL`, plus the unions `REAL_AND_COMPLEX`,
`REAL_AND_INTEGRAL` and `ALL`. It answers "which element types may this functor
be instantiated for", at compile time, because a generic lambda gives no
SFINAE-visible signal that its body rejects `std::complex` (return-type deduction
instantiates the body, which is not the immediate context). Two functions consume
it. `Dispatch::require_domain<Dom>` rejects an out-of-domain dtype at **record**
time, where the caller can still act on it. `Dispatch::with_element_type<Dom>` is
the dtype switch itself — one switch, shared by the elementwise, non-linear and
logic families and by `execute_index_reduction`, replacing the eight
near-identical seven-arm switches those used to carry. A dtype outside the domain
never instantiates its arm; it throws naming the family, the op and the dtype.

**An arm may be written either way.** `with_element_type` calls it as
`arm(Dispatch::ElementTag<T>{})` if it takes that tag — a plain generic lambda,
`[&](auto tag) { using T = typename decltype(tag)::type; ... }`, which is what the
library itself writes and the only spelling available at C++17 — and otherwise as
`arm.template operator()<T>()`, which is what a C++20 templated lambda
`[&]<typename T>(){ ... }` offers. Both were live before the move to C++17 and both
still are, so the standard change took nothing away from a consumer. An arm that is
neither gets a written-out `static_assert` naming both forms rather than a page of
overload-resolution noise.

`execute_blas_task` spells its switch out by hand rather than calling
`with_element_type`. Its functor is not an arm at all — it takes a *value* of the
element type plus the queue and the dependency list and returns the submitted
`sycl::event`, which is a different shape from the one-tag, void-returning arm this
helper calls — and wrapping it to fit crashes this project's clang during
instantiation. The arms and the domain rule are the same; the duplication is noted
at both sites.

**`execute_gemm`** (`engine/blas/dispatch.hpp`) is the BLAS family's only
executor, and its functor is not a per-element operation but a GEMM *epilogue*:
`T operator()(T value, int64_t row, int64_t col)`, applied to each element of C
in the instant before it is stored. The coordinates are what make it more than an
elementwise pass — with them a bias broadcast is expressible, so a bias-add plus
an activation that would otherwise be two more kernel launches over the whole of
C become zero. `Level3::gemm` *is* this function with
`Kernels::GemmIdentityEpilogue`, so the fused and unfused paths cannot drift.
It carries one warning that has to survive every retelling: **an activation
epilogue in a forward training pass is silently wrong** unless it also spills its
pre-activation, because the backward pass must differentiate at an input that no
longer reached memory. Nothing throws and nothing produces NaN — the model just
trains toward the wrong place. It is why cuBLASLt exposes `GELU_AUX_BIAS` and not
merely `GELU_BIAS`.

**`execute_zip_inplace`** is the variadic member of that set, and the one built
for optimiser steps. It takes a `std::array<Tensor*, N>` of same-shaped tensors
and calls the functor once per element index with an **lvalue reference to each**
— `op(t0[i], t1[i], ..., tN-1[i])` — so one functor can read and update all N in
a single pass over memory. An AdamW update (parameter decay, both moment
updates, both bias corrections, the epsilon, the parameter step, an optional
parameter EMA) is roughly seventeen recorded tasks when spelled as separate ops
and exactly one through this seam. All N tensors are declared as both reads and
writes on the graph, since nothing at this layer can tell which ones the functor
touched; `Tensor::region()`'s precise byte interval keeps that from
over-serialising disjoint views of one arena. N is a compile-time template
parameter because the kernel stages one register per operand per element and
expands the call through an `index_sequence` — a `span` would not do, and a
trailing parameter pack would displace the `name` / `op_id` / `params` arguments
every other executor in the family takes.

### Launch geometry (`engine/launch_1d.hpp`)

Every per-element kernel — elementwise, non-linear and logic — goes through two
primitives in `namespace SushiBLAS::Launch` rather than calling `parallel_for`
itself: `map` (`out[i] = op(in0[i], ...)`) and `zip_inplace`
(`op(t0[i], ..., tN-1[i])`, every slot writable). Both own the launch shape, so
retuning it is one edit instead of thirty-five.

- **Grid-stride `nd_range`, not a bare `sycl::range<1>`.** The geometry comes
  from `Device::plan_1d`, and the loop's `i < n` test *is* the bounds check, so
  there is no padded tail work-group. `Device::flat_work_group_size` picks the
  group width: `work_group_size` on a GPU, and on a CPU the widest group that
  still gives every hardware thread one, because a flat launch pays for
  work-group *count* (each group is a scheduling unit) and never pays for the
  barriers `work_group_size`'s small CPU answer is protecting against.
- **128-bit accesses where the device wants them.** When
  `Device::vector_width` returns more than 1 *and* every operand is 16-byte
  aligned (checked on the host at record time, because a `Tensor` view is only
  guaranteed to be aligned to its own element type), a wide pass loads and
  stores `sycl::vec<T, W>` — but calls the functor once per scalar component, so
  the functor contract stays scalar and every existing caller is unaffected. The
  remainder falls to the scalar loop. Skipped for `std::complex`. Both passes
  live in one kernel: when the wide path is not taken the wide loop's trip count
  is zero and the scalar loop starts at element 0, so there is no second kernel
  and no divergent branch.
- **Determinism is preserved by construction.** The grid, the group width and
  the scalar/wide split are pure functions of the device, the element type, `n`
  and the operands' alignment. Every element is written exactly once by exactly
  one work-item, from the same inputs, in the same arithmetic — which is what
  keeps the byte-for-byte reproducibility `sushiruntime_deterministic_fp`
  promises intact through this layer.

### Notable kernel designs

- **GEMM** (`include/SushiBLAS/kernels/sycl/blas/level3/gemm.hpp`) — two kernel shapes behind one
  entry point, chosen at launch from the `Device::Profile` in
  `include/SushiBLAS/engine/device_policy.hpp` and logged at `SB_LOG_INFO` so
  the choice is observable:
  - `Detail::gemm_local_tiled` — work-group-cooperative A/B staging through
    local memory, for devices with genuine scratchpad memory. The A tile's
    leading dimension is padded off a bank multiple; the arithmetic is spelled
    out at the read site.
  - `Detail::gemm_no_local` — no `local_accessor`, no barrier, a smaller
    register block. What a CPU backend wants, where "local memory" is the same
    cache the global pointer already reads through. A device that declares
    scratchpad memory but cannot fit the tile or host the work-group falls back
    here with a warning rather than failing the launch.

  The register block is derived from `sizeof(T)` against a byte budget
  (`Detail::register_block_dim`), not fixed: 8×8 for 4-byte and narrower types,
  4×4 for the wider ones. The four (layout, transpose) combinations are resolved
  on the host into a row/column stride pair (`Detail::strides_for`), so the
  innermost loop carries no layout branch and no extra instantiation. A trailing
  `Epilogue` template parameter, defaulting to `Kernels::GemmIdentityEpilogue`, is
  applied per element immediately before the store — the library's fusion hook for
  bias-add and activations, reachable from outside the library through
  `Dispatch::execute_gemm` (§7 above). The reference kernel takes the same
  parameter, so `VERIFY` mode has a ground truth for a *fused* GEMM rather than
  declaring every one of them wrong. See the `@warning` on either for why an
  *activation* epilogue in a forward training pass is a silent-wrongness trap.
- **SYRK** (`kernels/sycl/blas/level3/syrk.hpp`) — local-memory-tiled and
  work-group-cooperative, composed from GEMM's operand-addressing primitives
  (`Detail::strides_for`, `Detail::load_operand`) with the second operand set to
  the input itself, plus a triangular output mask. It owns its tile geometry
  (`Detail::SyrkTileConfig`) rather than sharing GEMM's, so retuning one cannot
  silently retune the other. It has not been given GEMM's local/no-local split
  yet — a named follow-up, not a regression.

  Both are `beta==0`-safe: they branch around reading `C` when `beta == T(0)`,
  because `0 * NaN == NaN` in IEEE-754 and a freshly-allocated,
  never-initialized `C` can decode as a HALF-precision NaN. See
  [NOTICE.md](../NOTICE.md) for the full attribution and the deliberate
  simplifications versus upstream portBLAS (no double buffering, no vectorized
  packet loads, no per-backend tuning table).
- **TRSM / TRSV** (`kernels/{reference,sycl}/blas/level{2,3}/trsm.hpp`) — one
  kernel serves both: `Level2::trsv` calls `Kernels::*::trsm<T>` with a
  single-column right-hand side. The substitution direction (forward vs.
  backward) is derived from an "effective triangle" trick: `A_eff(i,k) =
  transA ? A(k,i) : A(i,k)` is exactly `op(A)(i,k)`, and `eff_lower = transA ?
  upper : !upper` picks the walk direction — collapsing all four
  upper/lower × trans/non-trans cases into one code path. Right-side TRSM is
  the same system with the trans flag and the row/column roles swapped once
  more. It runs as one sequential `single_task` (a genuine write-after-read
  dependency between substitution steps rules out a parallel launch).
- **TRMV** (`kernels/*/blas/level2/trmv.hpp`) — in-place (`x = op(A)*x`)
  without a temporary buffer: walking the index from the "open end" of the
  effective triangle toward the "closed end" guarantees every element a step
  reads is still untouched when that step runs. Also a sequential
  `single_task` for the same reason as TRSM.
- **GEMV / GER / SYMV / SYR / SYR2** — embarrassingly parallel, one work-item
  per output element; each output depends only on inputs, never on other
  outputs.
- **Level1 reductions** (`asum`/`nrm2`/`dot`/`iamax`) — serial `single_task`
  kernels rather than a tree reduction. A deliberate correctness-first choice:
  Level1 vectors are small relative to the library's actual
  performance-critical tier (Level3 GEMM/SYRK). Note these are the *BLAS* Level1
  ops (`engine.blas()`), which are separate from the whole-tensor reductions
  below.
- **Whole-tensor folding reductions** (`engine/reduce_1d.hpp`) —
  `reductions().sum`/`mean`/`asum`/`nrm2`/`prod` do **not** use
  `sycl::reduction`. The spec fixes no combination order for it, floating-point
  addition is not associative, and this library links
  `sushiruntime_deterministic_fp` PUBLIC — so byte reproducibility was resting on
  an implementation detail nobody had promised, one that happens to hold on an
  OpenCL CPU and is far likelier to vary on the NVIDIA GPUs the project targets.
  `Reduce::reduce` replaces it with a shape fixed entirely by the geometry: a
  grid-strided serial pass per work-item using **Neumaier** compensation, a
  hand-written halving tree over local memory (not `reduce_over_group`, whose
  order the spec leaves equally free), then one task summing a fixed partial count
  in ascending index order. `max`/`min`/`all`/`any` are left on `sycl::reduction`
  deliberately — each carries a comment explaining why order cannot change its
  value. `argmax`/`argmin` are a serial scan through
  `Dispatch::execute_index_reduction`, which is a different seam from
  `execute_reduction`: their answer is an index, so it goes into an **INT64**
  result tensor rather than one sharing the input's dtype.

  `nrm2` is the one folding reduction whose accumulator is not a plain sum.
  `Reduce::ScaledSumOfSquares` carries a running scale and folds
  `(x / scale)^2`, because `sqrt(sum(x*x))` overflows to infinity for any fp32
  element past ~1.8e19 and underflows to zero below ~1e-22 — the first of which
  is exactly what a gradient-norm clip is called on. Its `value()` is a finished
  partial *norm* and its `combine` is `hypot`, which is what lets it fit
  `Reduce::reduce`'s existing two-stage shape unchanged.

  The per-work-group partials come from a scratch buffer allocated **once, at
  task-record time**, from the engine's allocator, and captured by the task. That
  is the right trade for a compile-once/replay library: a per-launch
  `malloc_device` plus a `host_task` free would add two submissions to every
  reduction on every replay. It is also a declared write region, so the tracker
  can order reductions against it rather than have them race invisibly.

  A CPU below `K_CPU_TWO_STAGE_MIN_ELEMENTS` (16K) takes a single sequential pass
  instead — measured, not assumed: at 1K the single pass is 1.5× faster, at 1M the
  two-stage shape is 11.8× faster, and they cross at 16K. Both are deterministic,
  and the shape is a pure function of the device and the element count, so the
  threshold does not weaken the guarantee.
- **Axis (segmented) reductions** (`engine/reduce_axis.hpp`) — the third launch
  shape, alongside `launch_1d.hpp`'s per-element map and `reduce_1d.hpp`'s
  whole-tensor fold: N independent folds running side by side over one buffer,
  producing a **vector**. `reductions().max_along` and `sum_along` are each one
  call to `Dispatch::execute_folding_axis_reduction`, differing only in the
  accumulation policy; a per-axis `mean` is `CompensatedSum` with `v / n` in the
  finalizer slot and a per-axis `min` is a four-line accumulator, so neither
  re-cuts the seam. An index-valued one (`argmax` along an axis) needs its own
  executor, because its result dtype differs from its input's — exactly the way
  `execute_index_reduction` is a separate twin of `execute_reduction` — but it
  reuses `Dispatch::axis_span_for` for the geometry rather than deriving it
  again.

  It exists because a log-sum-exp shift has to be **per row**. Shifting a batch
  by the batch's global maximum is exact in real arithmetic and wrong in
  binary32: the invariance is destroyed by the exponential, not by the
  subtraction, so a row about 104 below the batch maximum has every
  `exp(x - global_max)` underflow to zero, a row sum of exactly zero, and a
  log-softmax of `+inf`. See [GPU_VALIDATION.md §7](GPU_VALIDATION.md) for the
  before-and-after measurements.

  **`Reduce::AxisSpan` names no axis.** It carries five element counts — segment
  count, extent, and three strides — so the axis, the `Core::Layout` and any
  non-unit stride a view inherited are all resolved once on the host by
  `axis_span_for`. Both axes of both layouts are the same kernel with different
  numbers: no layout branch in the inner loop, no second instantiation.

  **Two shapes, chosen by `Reduce::plan_axis` from where the parallelism is.**
  One work-item per segment when there are already enough segments to fill the
  machine (which is every row-wise reduction over a batch, on every device), and
  one work-group per segment — grid-striding the segment, then the same
  hand-written halving tree stage one of `Reduce::reduce` uses — when there are
  too few segments and each is long enough to be worth the barriers. Both are
  correct at every shape; neither requires the segment length to relate to the
  work-group width. Unlike the whole-tensor fold this needs **no scratch
  buffer**, and structurally so: a work-group's tree *is* its segment's answer,
  so there are no partials to stage between two kernels.

  Determinism holds by the same argument the whole-tensor fold makes, and the
  item-per-segment shape makes it more cheaply: one work-item folds one segment
  in ascending index order, so there is no cross-work-item combination for the
  specification to leave free.
- **Random** (`kernels/sycl/random/philox.hpp`) — a hand-written
  Philox4x32-10 counter-based PRNG (constants verified against the Random123
  reference implementation), replacing what was previously oneMKL's VSL
  backend. Every distribution op (`uniform`, `normal`, `log_normal`,
  `exponential`, `poisson`, `bernoulli`, `discrete_uniform`, the Xavier/He
  initializers) is built on it, keyed by `Engine::get_seed()` and
  `get_and_increment_rng_offset()` so concurrent draws never collide on the
  same stream position.

All five dtypes (including `HALF`) are implemented for every one of the 17
from-scratch BLAS ops — a functional gain over oneMKL, which never supported
`HALF` for several of these routines (SYRK's `HALF` case still throws; that is
a pre-existing kernel-level limitation carried over, not a gap in the port).

---

## 8. The SushiRuntime dependency (D3)

SushiBLAS does not vendor SushiRuntime. `cmake/Runtime.cmake` resolves it
through the same three-mode, layered pattern used stack-wide (sushiengine,
sushiai):

1. **TARGET-guard** — if a superproject already brought in
   `SushiRuntime::SushiRuntime` (e.g. building sushiblas and sushiruntime
   together under a common parent), reuse it.
2. **`find_package(SushiRuntime QUIET)`** — an installed package.
3. **Sibling `add_subdirectory`** — the common case: `../sushiruntime` next to
   this checkout (or `$SUSHISTACK_HOME/sushiruntime` inside a workspace),
   added as a subdirectory.

Whichever mode fires, every consumer downstream links the same namespaced
alias, `SushiRuntime::SushiRuntime` — this is what lets `sushiblas`'s own
`SUSHIRUNTIME_DIR` cache variable, the CLI's `sushiruntime_dir` config field,
and an out-of-tree `find_package(SushiBLAS)` consumer all agree on one link
line regardless of how the dependency was actually found. SushiBLAS also
inherits SushiRuntime's `sushiruntime_deterministic_fp` contract transitively
through this link — it does not need (or own) a copy of that guarantee.

---

## 9. Quarantined surface (`include/SushiBLAS/experimental/`)

`LinalgOps` (SVD/QR/eigendecomposition) and `TransformsOps` (FFT/DFT) are
declared, header-only, under `include/SushiBLAS/experimental/` and are **not**
exposed through the `Engine` facade — there is no `engine.linalg()` or
`engine.signal()` accessor. This is a deliberate scope decision (D1): both
classes were declaration-only with zero `.cpp` implementations and no
callers, so relocating them out of the facade was a pure, zero-risk move.
FFT/LAPACK backend strategy is a named follow-up, not solved here.

`SparseBLAS` (`engine/blas/sparse.hpp`) is exposed through
`engine.blas()` (it composes into `BLASOps` alongside Level1/2/3) but every
method is still a `// TODO:` stub — no `SparseTensor`, no SpMV/SpMM. It exists
as a placed-but-unimplemented extension point, not a quarantined one.

---

## 10. Packaging

SushiBLAS is `find_package`-able: `SushiBLAS::sushiblas` (a `STATIC` library,
aliased in-tree and exported out-of-tree under the same name),
`install(EXPORT)` / `export(EXPORT)`, a generated `SushiBLASConfig.cmake`
(`SameMinorVersion` compatibility, pre-1.0), and a version-drift check between
`include/SushiBLAS/version.hpp` and the CMake project version at configure
time. `tests/package/` is a real out-of-tree consumer exercised in CI-shaped
fashion (see [INTEGRATION.md](INTEGRATION.md)).

Because `sushiblas` is a `STATIC` library, its build-policy targets
(`sushiblas_warnings`, `sushiblas_perf`, `sushiblas_sanitizers`, composed by
`sushiblas_apply_build_policy()` in `cmake/SushiBLASBuildPolicy.cmake`) pull
flag *values* onto the real target via `target_compile_options`/
`target_compile_definitions`/`target_link_options` +
`$<TARGET_PROPERTY:...>`, never `target_link_libraries`. For a static
library, `target_link_libraries(... PRIVATE policyTarget)` — even
`$<BUILD_INTERFACE:...>`-wrapped — records the policy target's name in
`INTERFACE_LINK_LIBRARIES` as `$<LINK_ONLY:...>`, which breaks
`install(EXPORT)`/`export(EXPORT)` with "requires target X that is not in any
export set." Pulling flag values instead never records a target name
anywhere, so nothing leaks and nothing breaks the export.

CMake module files that define SushiBLAS-specific functions
(`SushiBLASBuildPolicy.cmake`, `SushiBLASCompilerWarnings.cmake`,
`SushiBLASOptimization.cmake`, `SushiBLASSanitizers.cmake`) are prefixed with
the program's full name. `Vcpkg.cmake`, `Toolchain.cmake`, and
`SyclTarget.cmake` are not, because their logic is idempotent regardless of
which sibling repo's copy loads first — see
[CONTRIBUTING.md §4](CONTRIBUTING.md#4-coding-style) for the stack-wide naming
rule this follows and the collision it exists to prevent.

## Project layout

See the top-level [README.md § Project layout](../README.md#project-layout)
for the full directory tree.

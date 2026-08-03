# SushiBLAS Architecture

This document describes how SushiBLAS is built, layer by layer. It is written
for contributors who need an accurate mental model of the library — every
class and file named here exists under `include/SushiBLAS/`, `src/`, or
`kernels/`.

> If you only want to *use* the library, the top-level [README.md](../README.md)
> is enough. This document is about how the inside works.

---

## 1. What SushiBLAS is

SushiBLAS is a C++20 tensor and BLAS library layered on top of
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

The codebase targets C++20.

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
`COMPLEX64` (`core/common.hpp`); `MAX_TENSOR_RANK` is 6. `Core::Layout` is
`ROW_MAJOR` or `COLUMN_MAJOR`, fixed per-tensor at construction (row-major
computes strides from the last dimension inward; column-major from the first).
`is_contiguous()` and `is_aligned()` (checked against
`SushiRuntime::Core::DEFAULT_ALIGNMENT`) are used by the BLAS layer to decide
whether a vector operand can be treated as a flat, unit-stride buffer.

`Storage` (`storage.hpp`) is a `SushiRuntime::Core::RefCounted`, 64-byte
aligned, holding a raw `data_ptr`, its allocated `size_bytes` (rounded up to
`DEFAULT_ALIGNMENT`), and the `SushiRuntime::Memory::USMAllocator` +
`IAllocatorStrategy` (`Strategies::Shared` / `Device` / `Host`) it was
allocated with. Being `RefCounted` (via `sushi_ptr`, SushiRuntime's intrusive
atomic-refcount smart pointer) is what lets two `Tensor` views share one
allocation safely — the storage is freed only when the last `Tensor`
referencing it is destroyed.

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
under concurrent task submission).

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
`std::complex<float>{}`, `std::complex<double>{}` — so one `op_func` lambda,
templated on its first parameter's type via `decltype`, implements all five
dtypes without a hand-written switch at the call site. An unsupported dtype
throws (`SB_THROW_IF`) rather than silently no-opping — matching every other
"unsupported case" in the codebase (see [CONTRIBUTING.md](CONTRIBUTING.md)'s
error-handling convention).

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
since each op there is a short templated lambda rather than a multi-case
tiling kernel — the reference/SYCL split is the same, just colocated with the
dispatch code instead of factored out.

Those five headers are public API, not implementation detail: `execute_binary`,
`execute_unary_inplace`, `execute_nonlinear_forward` / `_backward`,
`execute_logic_unary` / `_binary` / `_ternary`, `execute_reduction` and
`execute_random` all live in `namespace SushiBLAS::Dispatch` and are installed.
A downstream framework composes several math ops into one functor, hands it to
one `execute_*` call, and gets a single fused SYCL kernel — the reason the
seam exists, and the reason "the consumer never writes SYCL itself" stays
true. See `include/SushiBLAS/engine/dispatch.hpp` for the shared vocabulary
(`Domain`) and the constraints a device-inlined functor has to satisfy.

### Notable kernel designs

- **GEMM / SYRK** (`kernels/sycl/blas/level3/`) — local-memory-tiled,
  work-group-cooperative kernels derived from portBLAS's tiling algorithm
  (fixed tile config: `item_rows=4, item_cols=4, wg_rows=8, wg_cols=8,
  tile_k=16`). SYRK reuses GEMM's tile-loading primitives with the second
  operand set to the input itself, plus a triangular output mask. Both are
  `beta==0`-safe: they branch around reading `C` when `beta == T(0)`, because
  `0 * NaN == NaN` in IEEE-754 and a freshly-allocated, never-initialized `C`
  can decode as a HALF-precision NaN. See [NOTICE.md](../NOTICE.md) for the
  full attribution and the deliberate simplifications versus upstream
  portBLAS (no double buffering, no vectorized packet loads, a single fixed
  tile size).
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
  performance-critical tier (Level3 GEMM/SYRK).
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

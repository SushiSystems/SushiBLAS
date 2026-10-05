# BLAS headers

`include/SushiBLAS/engine/blas` declares the BLAS surface: `Level1`, `Level2`, `Level3`, the
`SparseBLAS` placeholder, the helpers in `utils.hpp` and the GEMM executor of the dispatch seam in
`dispatch.hpp`.

`SparseBLAS` declares no operation. Its header listed the planned ones as TODO comments: a
`SparseTensor` structure, SpMV, SpMM, CSR as the first format and sparse-sparse operations.
`utils.hpp` likewise noted that `get_vec_params` does not yet accept a row or column slice of a
matrix. Those items belong in `docs/design/REMAINING_WORK.md`.

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `dispatch.hpp`

The BLAS member of SushiBLAS's generic op-dispatch seam: see `<SushiBLAS/engine/dispatch.hpp>` for
the shared vocabulary and for why the family exists at all.

It has exactly one executor, and unlike the five math/logic families the functor it takes is not a
per-element operation but a GEMM *epilogue*: a callable applied to each element of C, with that
element's coordinates, in the instant before it is stored. The five other families fuse a chain of
elementwise ops into one launch; this one fuses the elementwise work into the GEMM that produced its
input, which is one launch fewer still and is where a training step actually spends its time.

`Kernels::SYCL::gemm` has carried the `Epilogue` template parameter since SB-8, but the kernel
header was not installed and `Engine::blas().gemm(...)` has no epilogue argument, so nothing outside
this library could reach it. A fusion hook that exists and is unreachable is a fusion hook that does
not exist. `dispatch.hpp` makes it reachable.

The epilogue is opaque to this layer on purpose (that is what lets a consumer write one without
writing SYCL), so the memory it touches is opaque too. `Dispatch::GemmEpilogue` is how the caller
hands that back, in both directions.

### Why the access set is bundled with the functor

The obvious alternative is a twelfth, defaulted parameter on `execute_gemm` carrying the regions. It
is the smaller diff, it adds no public type and it needs no deduction layer, and it was rejected for
two costs it cannot avoid.

The first is that an omitted defaulted argument is invisible. The failure this whole facility exists
to close is a *silent* one: an undeclared read is not a wrong dependency edge, it is no edge, so the
graph runs the epilogue concurrently with whatever writes its operand and the run merely trains
towards the wrong place. Putting the fix behind an argument the caller can simply not write leaves
it exactly as easy to omit as it was before the fix existed, and leaves the omission invisible in a
diff. Here the operands travel *inside* the epilogue value, so wherever the functor goes, returned
from a helper, chosen in a switch, stored and forwarded, its access set goes with it. A separate
parameter cannot have that property, because it splits one indivisible fact into two
independently-losable ones.

The second is that the access set belongs to the *spelling*, not to the call. A caller choosing
among several epilogues in a switch (SushiAI's lowering does; a GELU_AUX_BIAS arm would add one
whose *write* set differs from every other arm's) would have to hoist the regions out of the arm
that knows them and keep the two in step by hand. Bundled, each arm states its own and they cannot
drift.

What the rejected shape would have bought is given up only partly: a bare functor is still accepted
and still means "declares nothing", so every existing call site, `Level3::gemm` with the identity
epilogue included, compiles and behaves exactly as before. See `Detail::append_epilogue_access`.

### `GemmEpilogue`

`execute_gemm` builds its access set from A, B and C, and it cannot do better on its own: the
epilogue is an opaque callable by design, that is what lets a consumer fuse without writing SYCL, so
a bias vector the functor dereferences is invisible to this layer. The caller always knows, because
the caller built the functor around that pointer. This is where it says so.

It carries **both directions**. An epilogue may also write: the pre-activation spill that makes an
activation safe to fuse into a *training* forward pass, what cuBLASLt's `GELU_AUX_BIAS` exists to
produce, and what the second warning on `execute_gemm` demands, is an auxiliary write, and whatever
reads that buffer afterwards needs an edge for exactly the same reason the bias's producer does.

```cpp
// One launch, and the task that writes `bias` is ordered before it.
Dispatch::execute_gemm(engine, X, W, Y, false, false, 1.0, 0.0,
                       "nn.linear_relu", "nn.linear_relu"_op,
                       Dispatch::gemm_epilogue(BiasRelu{bias.data_as<float>()})
                           .reads(bias));
```

**Operands are named as tensors, never as regions.** A `ResourceRegion` converts implicitly from a
bare pointer, and a bare pointer widens to the whole allocation of whatever it points at, so two
views of one arena become two unrelated resources, `overlaps` says false, and the tracker emits no
edge at all. That is exactly the mistake that cost a training run in the reduction seam (see
`<SushiBLAS/engine/math/reductions/dispatch.hpp>`), and taking a `Tensor` makes it unspellable here:
the region comes from `Tensor::region`, once, inside `GemmEpilogue`. A caller holding memory no
`Tensor` describes needs a new overload rather than a way around this one.

- `Epilogue`: The epilogue functor type; the same contract `execute_gemm` states for a bare one,
  unchanged. Only the functor reaches the device: the regions are read off on the host and left
  behind.

### `GemmEpilogue::writes`

For an auxiliary buffer the epilogue stores to on its way past: a pre-activation spill, an index, a
running statistic. Whatever reads it later is then ordered behind this GEMM, and whatever wrote it
before is ordered in front.

### `GemmEpilogue::declared_region`

A storage-less tensor yields a base-less region, which the tracker would order against nothing: the
declaration would compile, read as a fix, and be a no-op. That is the failure mode this whole class
exists to remove, so it throws instead.

### `gemm_epilogue`

The spelling a call site uses, because it deduces `Epilogue` where naming the class would not (C++17
has class template argument deduction, but a function template is what the rest of this seam is
written in and what an aggregate epilogue deduces from without a deduction guide).

`GemmEpilogue::writes` onto it.

### `Detail::epilogue_functor` for a bundle

More specialised than the bare-functor overload, so partial ordering picks it for a bundle and the
other one for everything else.

### `Detail::append_epilogue_access` for a bare functor

This is what keeps "an epilogue that declares nothing behaves exactly as before" a property of the
code rather than a claim about it: the access set `execute_gemm` builds is byte for byte the one it
built before this facility existed.

### `execute_gemm`

The generic seam behind `Level3::gemm`, which is `execute_gemm` with
`SushiBLAS::Kernels::GemmIdentityEpilogue`. Everything `Level3::gemm` does, shape and
leading-dimension derivation, batch strides, the reads/writes it declares on the graph, the
runtime-dtype switch, the `ExecutionMode` triad, happens here, once, so the fused and unfused paths
cannot drift apart.

The epilogue is applied to the **fully formed** element, after alpha and after `beta * C`,
immediately before the store. Its `row` and `col` arguments are what make it more than an
elementwise pass: with them, a bias broadcast is expressible, and a bias-add plus an activation that
would otherwise be two more kernel launches over the whole of C become zero.

```cpp
// One launch: the product, the row and column biases, and the ReLU.
struct BiasRelu
{
    const float* bias;   // one value per column of C

    template<typename T>
    T operator()(T value, int64_t row, int64_t col) const
    {
        const T y = value + static_cast<T>(bias[col]);
        return y > T(0) ? y : T(0);
    }
};

Dispatch::execute_gemm(engine, X, W, Y, false, false, 1.0, 0.0,
                       "nn.linear_relu", "nn.linear_relu"_op,
                       Dispatch::gemm_epilogue(BiasRelu{bias.data_as<float>()})
                           .reads(bias));
```

The `.reads(bias)` is not decoration. Without it the task declares A, B and C and nothing else,
`bias` is a resource this GEMM never mentions, and the graph is free to run the task that writes it
at the same time. See `GemmEpilogue`.

- `Dom`: Which element types to instantiate `epilogue` for. Defaults to Domain::REAL, exactly as
  `execute_binary` does and for the same reason: a functor written for real scalars always compiles
  under it, while `std::complex` has no ordering, so the `value > 0` at the heart of every
  activation is ill-formed for it. Pass Domain::REAL_AND_COMPLEX only when `epilogue` is well-formed for `std::complex`: `Level3::gemm` does, because the identity epilogue is.
- `Epilogue`: The epilogue functor type, deduced, or a `GemmEpilogue` wrapping one. The functor must
  provide `T operator()(T value, int64_t row, int64_t col) const` for every element type inside
  `Dom`: a template or generic-lambda call operator is the usual spelling, since the dtype is a
  runtime value and every arm inside the domain is instantiated. A bare functor declares no operands
  of its own and produces exactly the access set this seam has always built; a `GemmEpilogue` adds
  whatever the caller declared on it.
- `C`: The output; must not overlap `A` or `B`.
- `beta`: Scale applied to the incoming C; exactly zero skips reading C.
- `epilogue`: The per-element epilogue functor, bare or bundled with its operands by
  `gemm_epilogue`.

Returns: An empty sycl::event; the work is deferred until `Engine::execute()`.

Warning: This is a low-level generic seam, not a convenience wrapper. Same device-kernel constraints
as `execute_binary:` `epilogue` is inlined into a SYCL device kernel, so it must be device-copyable
and trivially copyable, must capture by value only, must not dereference host pointers or touch
host-only state, must not allocate, and must not throw: none of which the compiler will always
diagnose for you. A pointer it carries (a bias vector, say) must be a device-accessible USM pointer,
and it must stay alive until `Engine::execute()` has completed, since the task holds the functor by
value and runs later. Staying alive is not the same as being *ordered*: declare the memory through
`gemm_epilogue` as well, or nothing sequences this task against whatever else touches it.

Warning: **An activation epilogue in a forward *training* pass is silently wrong** unless it also
writes its pre-activation input to an auxiliary buffer, or is invertible. The backward pass has to
differentiate the activation at its input; if the only thing that reached memory is the activation's
output, that input no longer exists and the gradient is computed at the wrong point. Nothing throws,
nothing produces NaN: the model simply trains towards the wrong place, which is the worst failure
mode a numerical library has, because there is no symptom to debug. This is exactly why cuBLASLt
exposes `GELU_AUX_BIAS`, which spills the pre-activation, and not merely `GELU_BIAS`. Inference-only
fusion carries no such constraint, and neither does a bias-only epilogue, which is invertible. If
you fuse an activation into a training forward pass, the epilogue must write `value` out to an
auxiliary buffer before it returns the activated result, and it must declare that buffer through
`GemmEpilogue::writes`, or the backward pass that reads it races the GEMM that filled it and the
spill buys nothing.

Warning: Under `ExecutionMode::FORCE_REFERENCE` and `VERIFY`, `epilogue` is also invoked on the
**host**, by `Kernels::Reference::gemm`. Any pointer it carries must therefore be host-readable in
those modes, which shared USM is and device USM is not. That is the same constraint
`FORCE_REFERENCE` already places on the operands themselves.

Throws `std::runtime_error` if any operand has rank < 2, if the dtypes disagree, if the dtype falls
outside `Dom`, or if the dtype is one `execute_blas_task` does not support.

At `SB_THROW_IF(Core::is_complex_dtype(A.dtype),`: rejected here, at record time, rather than at
execution time: the caller learns from the call that recorded the mistake instead of from a later
Engine::execute() that names a task it did not write.

At `std::vector<SushiRuntime::Core::ResourceRegion> reads = {};`: `C = alpha * op(A) * op(B) + beta
* C` reads C whenever beta is non-zero: the kernels branch around the read on exactly `beta == 0`.
Declaring that read makes the access set true rather than merely survivable via the tracker's WAW
edge.

At `Detail::append_epilogue_access(epilogue, reads, writes);`: anything the epilogue itself touches.
`execute_gemm` still cannot discover it (the functor is an opaque callable and that is the point of
it), but the caller can state it, and `GemmEpilogue` is how. A bare functor appends nothing, so the
reads and writes gathered before the call are the whole of it, exactly as before.

At `const auto kernel_epilogue = Detail::epilogue_functor(epilogue);`: the functor alone, never the
bundle. A GemmEpilogue owns std::vectors and the kernel closure is copied into a SYCL kernel, where
a std::vector is not device-copyable, so the access set is read off here, on the host, and left
behind. Naming this instead of `epilogue` inside the closure is what keeps it out.

## `level1.hpp`

### Scalar parameters of `Level1`

Scalars are taken as `double` at every level of this API and narrowed to the tensor's element type
at the kernel boundary, where that type is finally known. Declaring them `float`, as Level 1 and
Level 2 used to, silently rounded an fp64 computation's scalar to about 1e-7 relative error with no
diagnostic, which made FLOAT64 a partial lie. Templating the public signature instead was rejected:
a tensor's dtype is a runtime value here, so a compile-time scalar type would push a dtype switch
onto every caller.

### `Level1::nrm2`

This operation is more robust against overflow/underflow than a direct implementation.

### `Level1::iamax`

This is often used for pivoting in matrix factorizations.

- `result`: Scalar output tensor (stores the 1-based or 0-based index depending on implementation).

## `level2.hpp`

### Scalar parameters of `Level2`

Scalars (`alpha`, `beta`) are `double` throughout, matching Level 1 and Level 3, and are narrowed to
the tensor's element type at the kernel boundary. See Level1::axpy for why.

## `level3.hpp`

### `Level3`

`Level3` provides an interface for standard BLAS Level 3 operations, which involve matrix-matrix
computations. These operations are typically the most computationally intensive and benefit greatly
from optimized implementations like oneMKL. All operations are executed asynchronously using the
SushiBLAS task graph system.

## `utils.hpp`

### `is_complex`

`is_complex` / `is_complex_v` were declared here too, identically to two copies under src/. They now
have one home, `<SushiBLAS/core/traits.hpp>`, which this header includes so anything that reached
the trait through utils.hpp still compiles: spell it `Core::is_complex_v<T>`.

### `execute_blas_task`

Registers a single generically-dispatched BLAS task (Level 2/3) on the engine's task graph. This is
the level1_internal.hpp::execute_level1 shape (one add_task, one switch(dtype) dispatching a
generically-templated closure) generalized for ops with matrix operands. Unsupported dtypes fail
loudly (P1 corollary), never silently no-op.

The body is `Internal::execute_typed_task`, which this forwards to with the family string "BLAS
operation": the one thing about the recorder that was ever BLAS-specific. This name is kept because
~40 call sites read better for it and because a BLAS op should not have to name its own family at
every site.

- `Dom`: Which element types to instantiate `op_func` for, in exactly the sense `Dispatch::Domain`
  defines for the math and logic families. It defaults to REAL_AND_COMPLEX because that is what
  every built-in BLAS *arithmetic* op wants and has always had; the parameter exists for
  `Dispatch::execute_gemm`, whose functor is written by the *caller* and is therefore no more likely
  to be well-formed for `std::complex` than an elementwise lambda is, and for `copy`, which moves
  elements without doing arithmetic to them and so passes Domain::ALL. Without it, a consumer's
  real-only GEMM epilogue is a compile error in the two arms that will never run.
- `op_func`: A callable `(auto scalar_type, sycl::queue&, const std::vector<sycl::event>&) ->
  sycl::event`, generic over the runtime dtype via `decltype(scalar_type)`.

Returns: An empty sycl::event; the work is deferred until `Engine::execute()`.

Throws `std::runtime_error` at record time if `dtype` falls outside `Dom`.

### `dispatch_with_mode`

`Internal::dispatch_with_mode` used to live here and now lives in `<SushiBLAS/engine/task.hpp>`,
which this header includes, so every call site that reached it through utils.hpp still does,
unqualified and unchanged. It moved for the same reason the recorder did: the spatial domain needs
the FORCE_REFERENCE / normal / VERIFY triad verbatim, and "BLAS utilities" is the wrong home for
something two domains depend on.

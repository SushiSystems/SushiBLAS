# BLAS headers

`modules/operations/blas/include/SushiBLAS/blas` declares the dense BLAS surface and the bricks its operations
check and record with. `BLASOps` in `../blas.hpp` derives from `Level1`, `Level2` and `Level3`,
so one view records all three levels.

| File | Declares |
| --- | --- |
| `level1.hpp` | `Level1`: `axpy`, `scal`, `copy`, `swap`, `dot`, `dotu`, `dotc`, `nrm2`, `asum`, `iamax`, `rot`, `rotg`, `rotm`, `rotmg` |
| `level2.hpp` | `Level2`: the dense routines `gemv`, `ger`, `geru`, `gerc`, `symv`, `hemv`, `trmv`, `trsv`, `syr`, `her`, `syr2`, `her2`; the band routines `gbmv`, `sbmv`, `hbmv`, `tbmv`, `tbsv`; the packed routines `spmv`, `hpmv`, `tpmv`, `tpsv`, `spr`, `hpr`, `spr2`, `hpr2` |
| `level3.hpp` | `Level3`: `gemm`, `symm`, `hemm`, `trsm`, `syrk`, `herk`, `syr2k`, `her2k` |
| `scalar.hpp` | `BLASScalar`, the real or complex scalar argument of a call |
| `argument_check.hpp` | The record-time checks on the flags, scalars and routine domain of a call |
| `matrix_operand.hpp` | `MatrixOperand`, the batch and matrix geometry of one dense matrix operand |
| `vector_operand.hpp` | `VectorOperand`, the length and increment of one vector operand |
| `band_operand.hpp` | `BandOperand`, the geometry of one band-stored matrix operand |
| `packed_operand.hpp` | `PackedOperand`, the geometry of one packed triangle operand |
| `scalar_operand.hpp` | The check on a tensor that stands for one scalar of a call |
| `blas_metadata.hpp` | `BLASMetadata`, which fills a task's metadata slots in one fixed order |
| `blas_task.hpp` | `execute_blas_task`, which records one dtype-dispatched BLAS task |
| `local_tile.hpp` | The local-memory demand of a tiled kernel and its record-time check |
| `packed_product_scratch.hpp` | The helpers that size and lease the panels of a packed product |
| `dispatch.hpp` | The GEMM executor of the dispatch seam and the epilogue bundle it accepts |
| `sparse.hpp` | `SparseBLAS`, a placeholder class that declares no operation |

`SparseBLAS` is not the sparse surface and no class derives from it. The sparse operations are
`SparseOps` in `../sparse.hpp`; see [../sparse/README.md](../sparse/README.md).

Every routine returns a `TaskHandle`. A flag is a `Core::Transpose`, `Core::Uplo`, `Core::Diag` or
`Core::Side`, never a `bool`, and a scalar that may be complex is a `BLASScalar`. Every record-time
refusal of an operand, a flag or a scalar is `std::invalid_argument`; a dtype the operand's
device cannot compute in and a device-only operand under `ForceReference` are
`UnsupportedExecution`.

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
header was not installed and `Level3::gemm` has no epilogue argument, so nothing outside
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
Dispatch::execute_gemm(engine.recorder(), X, W, Y,
                       Core::Transpose::NoTrans, Core::Transpose::NoTrans, 1.0, 0.0,
                       "nn.linear_relu", "nn.linear_relu"_op,
                       Dispatch::gemm_epilogue(BiasReLU{bias.data_as<float>()})
                           .reads(bias));
```

**Operands are named as tensors, never as regions.** A `BufferRegion` converts implicitly from a
bare pointer, and a bare pointer widens to the whole allocation of whatever it points at, so two
views of one arena become two unrelated resources, `overlaps` says false, and the tracker emits no
edge at all. That is exactly the mistake that cost a training run in the reduction seam (see
`<SushiBLAS/reductions/dispatch.hpp>`), and taking a `Tensor` makes it unspellable here:
the region comes from `Tensor::region`, once, inside the `OperandSet` a `GemmEpilogue` holds. A
caller holding memory no `Tensor` describes needs a new overload rather than a way around this
one.

Declaring an operand also retains its storage for the life of the task and lets the recorder see
it: a device-only tensor declared on an epilogue is refused under `ForceReference` as an operand
of the GEMM is. A bare functor that hides a pointer gets neither.

- `Epilogue`: The epilogue functor type; the same contract `execute_gemm` states for a bare one,
  unchanged. Only the functor reaches the device: the regions are read off on the host and left
  behind.

### `GemmEpilogue::writes`

For an auxiliary buffer the epilogue stores to on its way past: a pre-activation spill, an index, a
running statistic. Whatever reads it later is then ordered behind this GEMM, and whatever wrote it
before is ordered in front.

### `GemmEpilogue::require_declared_storage`

A storage-less tensor yields a base-less region, which the tracker would order against nothing: the
declaration would compile, read as a fix, and be a no-op. That is the failure mode this whole class
exists to remove, so `reads` and `writes` throw `std::invalid_argument` instead.

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
code rather than a claim about it: the overload adds nothing to the `OperandSet`, so the set holds
A, B and C alone. The bundle overload merges the bundle's set into it.

### `execute_gemm`

The generic seam behind `Level3::gemm`, which is `execute_gemm` with
`SushiBLAS::Kernels::GemmIdentityEpilogue`. Everything `Level3::gemm` does, shape and
leading-dimension derivation, batch strides, the reads/writes it declares on the graph, the
packed-panel scratch, the runtime-dtype switch, the choice between the reference and the SYCL
kernel, happens here, once, so the fused and unfused paths cannot drift apart.

The epilogue is applied to the **fully formed** element, after alpha and after `beta * C`,
immediately before the store. Its `row` and `col` arguments are what make it more than an
elementwise pass: with them, a bias broadcast is expressible, and a bias-add plus an activation that
would otherwise be two more kernel launches over the whole of C become zero.

```cpp
// One launch: the product, the row and column biases, and the ReLU.
struct BiasReLU
{
    const float* bias;   // one value per column of C

    template<typename T>
    T operator()(T value, int64_t row, int64_t col) const
    {
        const T y = value + static_cast<T>(bias[col]);
        return y > T(0) ? y : T(0);
    }
};

Dispatch::execute_gemm(engine.recorder(), X, W, Y,
                       Core::Transpose::NoTrans, Core::Transpose::NoTrans, 1.0, 0.0,
                       "nn.linear_relu", "nn.linear_relu"_op,
                       Dispatch::gemm_epilogue(BiasReLU{bias.data_as<float>()})
                           .reads(bias));
```

The `.reads(bias)` is not decoration. Without it the task declares A, B and C and nothing else,
`bias` is a resource this GEMM never mentions, and the graph is free to run the task that writes it
at the same time. See `GemmEpilogue`.

- `Dom`: Which element types to instantiate `epilogue` for. Defaults to Domain::Real, exactly as
  `execute_binary` does and for the same reason: a functor written for real scalars always compiles
  under it, while `std::complex` has no ordering, so the `value > 0` at the heart of every
  activation is ill-formed for it. Pass Domain::RealAndComplex only when `epilogue` is well-formed for `std::complex`: `Level3::gemm` does, because the identity epilogue is.
- `Epilogue`: The epilogue functor type, deduced, or a `GemmEpilogue` wrapping one. The functor must
  provide `T operator()(T value, int64_t row, int64_t col) const` for every element type inside
  `Dom`: a template or generic-lambda call operator is the usual spelling, since the dtype is a
  runtime value and every arm inside the domain is instantiated. A bare functor declares no operands
  of its own and produces exactly the access set this seam has always built; a `GemmEpilogue` adds
  whatever the caller declared on it.
- `recorder`: The recorder the task is recorded through; `engine.recorder()` for an engine.
- `transA`, `transB`: `Core::Transpose` values. A value outside the enumeration throws.
- `C`: The output; must not overlap `A` or `B`.
- `alpha`, `beta`: `BLASScalar` values. A non-zero imaginary part throws for a real dtype. `beta`
  exactly zero skips reading C.
- `epilogue`: The per-element epilogue functor, bare or bundled with its operands by
  `gemm_epilogue`.

Returns: The `TaskHandle` of the recorded task; the work is deferred until `Engine::execute()`.

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

Warning: Under `ExecutionMode::ForceReference`, `epilogue` is also invoked on the **host**, by
`Kernels::Reference::gemm`. Any pointer it carries must therefore be host-readable in that mode,
which shared USM is and device USM is not. That is the same constraint
`ForceReference` already places on the operands themselves. The recorder enforces it for a tensor
declared through `gemm_epilogue` and cannot for a pointer a bare functor hides.

Throws `std::invalid_argument` on a flag outside its enumeration, dtypes that disagree, a complex
scalar for a real dtype, a complex dtype under `Domain::Real`, an operand of rank below 2 or
without storage, a non-contiguous operand, mixed layouts, a shape that does not conform or a batch
count that does not match; see "Operand validation" below. Throws `UnsupportedExecution` as
`describe_matrix_operand` and `TaskRecorder::record_pair` state.

At the `if constexpr (Dom == Domain::Real)` block: a complex tensor is rejected here, at record
time, rather than at execution time: the caller learns from the call that recorded the mistake
instead of from a later Engine::execute() that names a task it did not write.

At `if (beta.is_zero())`: `C = alpha * op(A) * op(B) + beta * C` reads C whenever beta is
non-zero: the kernels branch around the read on exactly `beta == 0`. Declaring that read, through
`OperandSet::updates`, makes the access set true rather than merely survivable via the tracker's
WAW edge.

At `Detail::append_epilogue_access(epilogue, operands);`: anything the epilogue itself touches.
`execute_gemm` still cannot discover it (the functor is an opaque callable and that is the point of
it), but the caller can state it, and `GemmEpilogue` is how. A bare functor appends nothing, so
A, B and C are the whole of it.

At `Internal::allocate_packed_product_scratch(...)`: the panels the packed CPU kernel stages are
leased here and declared on the same set. The buffer is empty when the device profile of C does
not select the packed kernel, and under `ForceReference`.

At `const auto kernel_epilogue = Detail::epilogue_functor(epilogue);`: the functor alone, never the
bundle. A GemmEpilogue owns an `OperandSet`, whose vectors are not device-copyable, and the kernel
closure is copied into a SYCL kernel, so the access set is read off here, on the host, and left
behind. Naming this instead of `epilogue` inside the closure is what keeps it out.

## `scalar.hpp`

### `BLASScalar`

One scalar argument of a call, held on the host as two doubles. It converts implicitly from a
`double` and from a `std::complex<double>`, so a call site writes `1.0` or
`std::complex<double>(0.0, 1.0)`. `as<T>()` rounds it once to the element type at the kernel
boundary, where that type is finally known; HALF goes through binary32 first. A routine that takes
a `BLASScalar` checks it with `require_real_scalar`: a non-zero imaginary part given to a real
tensor throws `std::invalid_argument`. A routine whose scalar is real by definition, such as `her`
or `rot`, takes a `double`.

## `argument_check.hpp`

| Function | Refuses |
| --- | --- |
| `require_flag` | A `Transpose`, `Uplo`, `Diag` or `Side` whose value is not an enumerator, such as a value-initialised one |
| `require_real_scalar` | A scalar with a non-zero imaginary part for a real dtype |
| `require_real_dtype` | A complex tensor in a routine defined for real ones; the message names the complex siblings, as `dot` names `dotu` and `dotc` |
| `require_transpose_not` | The one `Transpose` value a routine does not define for a complex tensor; the message names the routine that takes it |

Each throws `std::invalid_argument` through `Detail::throw_operand_error`.

## `vector_operand.hpp`, `band_operand.hpp`, `packed_operand.hpp`, `scalar_operand.hpp`

Each describer checks one operand and returns its geometry, so every routine that takes that kind
of operand derives it the same way. All of them call `require_storage` and
`require_device_dtype`.

`describe_vector_operand` returns a length and an increment of at least 1. It refuses a rank-1
stride of zero or less and a non-contiguous tensor of rank above 1. `require_vector_length` and
`require_unit_increment` are the two follow-up checks; `rotm`, `rotmg` and `trsv` use the second.

`describe_band_operand` takes `kl` and `ku` and returns the stored lines, the leading dimension
and the layout of a band-stored matrix. The tensor is a contiguous rank-2 matrix whose lines hold
at least `kl + ku + 1` elements; a line is a column under `ColumnMajor` and a row under `RowMajor`.
`describe_triangle_band_operand` is the form for one triangle with `k` off-diagonals, used by
`sbmv`, `hbmv`, `tbmv` and `tbsv`.

`describe_packed_operand` returns the order and the layout of a packed triangle. The tensor is
rank 1 and unstrided and holds `n * (n + 1) / 2` elements for some `n`; `largest_packed_order`
finds `n` exactly. `RowMajor` packs by rows and `ColumnMajor` by columns.

`require_scalar_operand` checks a tensor that stands for one scalar, as the four operands of
`rotg` do: storage, the expected dtype and exactly one element.

## `blas_metadata.hpp`

### `BLASMetadata`

Builds the `TaskDescription` of one BLAS task in one fixed order: each scalar as two slots, real part
then imaginary part, then each flag as one slot holding its CBLAS number, then each extent, such
as a band width. Every routine fills its slots through it, so a reader of a recorded graph finds a
flag in the same place for every routine. Writing past `TaskDescription::K_MAX_PARAMETERS` is an
`SB_ASSERT`, which aborts in a build without `NDEBUG` and is empty with it.

## `level1.hpp`

### Scalar parameters of `Level1`

Scalars are taken as `BLASScalar`, or as `double` where the routine defines a real scalar, at
every level of this API and narrowed to the tensor's element type at the kernel boundary, where
that type is finally known. Declaring them `float`, as Level 1 and Level 2 used to, silently
rounded an fp64 computation's scalar to about 1e-7 relative error with no diagnostic, which made
FLOAT64 a partial lie. Templating the public signature instead was rejected: a tensor's dtype is a
runtime value here, so a compile-time scalar type would push a dtype switch onto every caller.

### `Level1::dot`, `dotu`, `dotc`

`dot` is defined for real tensors and refuses a complex one, naming `dotu` and `dotc`. `dotu` does
not conjugate and `dotc` conjugates `x`.

### `Level1::rotg`, `rotm`, `rotmg`

`rotg` builds the Givens rotation that maps (a, b) to (r, 0), from four scalar tensors. `rotm`
applies the modified rotation a five-element `param` describes, and `rotmg` builds one. The
arithmetic of the three is in `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/givens.hpp` and `modified_givens.hpp`, shared by the
reference and the SYCL kernel.

### `Level1::nrm2`

This operation is more robust against overflow/underflow than a direct implementation.

### `Level1::iamax`

This is often used for pivoting in matrix factorizations.

- `result`: Scalar INT64 output tensor. The index is zero-based, and the smallest index wins a
  tie. For a complex tensor the magnitude is `|Re(x_i)| + |Im(x_i)|`.

## `level2.hpp`

### Scalar parameters of `Level2`

Scalars (`alpha`, `beta`) are `BLASScalar`, or `double` where the routine defines a real scalar
(`ger`, `her`, `hpr`), matching Level 1 and Level 3, and are narrowed to the tensor's element type
at the kernel boundary. See "Scalar parameters of `Level1`" for why.

## `level3.hpp`

### `Level3`

`Level3` provides an interface for standard BLAS Level 3 operations, which involve matrix-matrix
computations. These operations are typically the most computationally intensive and benefit greatly
from optimized implementations like oneMKL. Each operation records one task on the graph and
returns its `TaskHandle`; the task runs when `Engine::execute()` runs, and that call blocks until
it finishes.

### `Level3::symm`, `hemm`

`symm` computes `C = alpha * A * B + beta * C` or `C = alpha * B * A + beta * C` for a symmetric
`A` of which one triangle is read, and `hemm` the same for a Hermitian `A`. Both are in
`modules/operations/blas/source/level3/symm.cpp`. The operation builds a `Kernels::SymmetricAccess` for `A` and
calls `gemm_accessed`, so it runs on the GEMM kernels, the packed one included, and leases the
same packed-panel scratch. `C` is declared with `writes` when `beta` is exactly zero and with
`updates` otherwise, as in `execute_gemm`.

### `Level3::syrk`, `herk`, `syr2k`, `her2k`

The rank-k and rank-2k updates write one triangle of `C` and leave the other as it was, so `C` is
declared with `updates` whatever `beta` is. `herk` and `her2k` take a real `beta`, and `herk` a
real `alpha`.

### Operand validation

Every Level 3 operation reads its matrix operands through `matrix_operand.hpp`, so GEMM, SYMM,
SYRK, SYR2K, TRSM and their Hermitian forms derive rank, batch count, rows, columns, leading
dimension and layout the same way. The
kernels take one layout and dense leading dimensions, so `describe_matrix_operand` rejects a
non-contiguous operand (a transposed view, a slice) at record time, and `require_same_layout`
rejects a call whose operands mix row-major and column-major.

The output (C, or B for TRSM) sets the batch count of a call. An input with the same batch count is
stepped matrix by matrix. An input of rank 2, or one whose batch axes hold a single matrix, is
reused for every batch, and `batch_stride` reports a stride of 0 for it. Any other batch count
throws. When an input and the output both hold more than one matrix, `require_same_batch_axes`
also requires every axis before the last two to match: a `{2, 3, m, n}` input against a
`{3, 2, m, n}` output throws, although both hold six matrices. GEMM also checks that op(A) is
m x k and op(B) is k x n against C.

The dense matrix A of a Level 2 operation has rank 2; `require_unbatched` rejects a higher rank,
and `require_square` rejects a matrix that is not square where the routine needs one.

Level 1 and Level 2 vectors must have a positive increment: a stride of 0 would make every write
land on one element, from many work-items at once. The scalar results of `dot`, `asum`, `nrm2` and
`iamax` must have storage, exactly one element and the dtype the kernel writes: the operand dtype
for `dot`, its real component type for `asum` and `nrm2`, and INT64 for `iamax`.

Every check in this section throws `std::invalid_argument`. No record-time check of the BLAS layer
throws `std::runtime_error`.

`syrk`, `herk`, `syr2k` and `her2k` also refuse a device at record time. Outside
`ForceReference`, `require_local_tile` compares the local memory and the work-group size the
tiled SYCL kernel stages for the dtype with the profile of the device of C, and throws
`UnsupportedExecution` when the device cannot host them. `syrk` and `herk` ask only where the
profile does not select the packed kernel.

## The record-time helpers

These were one header, `utils.hpp`, until 2026-10-09. `blas_task.hpp`, `local_tile.hpp` and
`packed_product_scratch.hpp` are here; `scalar_result.hpp` and `scratch_lease.hpp` are in the
`operands` module, because `sparse` uses them too.

### `is_complex`

`is_complex` / `is_complex_v` have one home, `<SushiBLAS/core/traits.hpp>`: spell the trait
`Core::is_complex_v<T>`.

### `allocate_blas_scratch`, `allocate_level1_scratch`, `allocate_packed_product_scratch`

`allocate_blas_scratch` leases device scratch from the recorder's pool on the device of an anchor
operand and declares it on the task's `OperandSet`. It returns an empty buffer when the anchor has
no storage, when the byte count is zero, and under `ForceReference`, where no SYCL kernel reads
the scratch. It reads the mode from the recorder, not from an engine. `allocate_level1_scratch`
sizes the partials of a Level 1 reduction from `Reduce::planned_partials` and forwards;
`asum`, `dot`, `iamax` and `nrm2` use it. `allocate_packed_product_scratch` sizes the panels of the
packed CPU product from `Kernels::plan_packed_product` and forwards. `running_solution_bytes`
gives the element size of the wide solution a triangular substitution carries, or 0 when the
solution is carried in the dtype itself.

### `execute_blas_task`

Registers a single generically-dispatched BLAS task on the recorder's task graph: one add_task
and one switch(dtype) dispatching a generically-templated closure. Unsupported dtypes fail
loudly (P1 corollary), never silently no-op.

The body is `Internal::execute_typed_task`, which this forwards to with the family string "BLAS
operation": the one thing about the recorder that was ever BLAS-specific. This name is kept because
33 call sites read better for it and because a BLAS op should not have to name its own family at
every site.

- `Dom`: Which element types to instantiate `op_func` for, in exactly the sense `Dispatch::Domain`
  defines for the math and logic families. It defaults to `Domain::RealAndComplex` because that
  is what every built-in BLAS *arithmetic* op wants and has always had; the parameter exists for
  `Dispatch::execute_gemm`, whose functor is written by the *caller* and is therefore no more likely
  to be well-formed for `std::complex` than an elementwise lambda is, and for `copy`, which moves
  elements without doing arithmetic to them and so passes Domain::All. Without it, a consumer's
  real-only GEMM epilogue is a compile error in the two arms that will never run.
- `operands`: The `OperandSet` the operation declared: every tensor the task reads or writes, and
  its scratch.
- `op_func`: A callable `(auto scalar_type, const Internal::KernelPairDispatcher&, sycl::queue&,
  const std::vector<sycl::event>&) -> sycl::event`, generic over the runtime dtype via
  `decltype(scalar_type)`.

Returns: The `TaskHandle` of the recorded task; the work is deferred until `Engine::execute()`.

Throws `std::invalid_argument` at record time if `dtype` falls outside `Dom`. Throws
`UnsupportedExecution` at record time if the recorder is in `ForceReference` and a declared tensor
is device-only USM.

Each task body receives a `const Internal::KernelPairDispatcher&` and calls
`dispatcher.run(q, deps, reference_fn, sycl_fn)`; the mode inside the dispatcher is the one the
recorder held at record time. `modules/engine/engine/README.md` describes the recorder, the
dispatcher and the `ForceReference` refusal.

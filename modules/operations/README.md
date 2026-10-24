# The operation sources

The translation units of the operation families, and of the three modules below them that have
one. Each file under `modules/operations/<family>/source/` defines the methods of one operation:
it checks the operands, declares the tensors the task reads and writes in an `OperandSet`, and
records the task through the view's `TaskRecorder`. Every method returns the `TaskHandle` of the
task it recorded, except `RandomOps::set_seed`, which records no task and returns `void`. The
kernels a task runs are in the modules under `modules/kernels/`.

| Path | Defines |
| --- | --- |
| `modules/engine/engine/source/engine.cpp` | The `Engine` constructor |
| `io/source/io.cpp` | `IO`: the `.sushi`, raw binary and `.npy` readers and writers, and the text rendering |
| `modules/data/sparse_storage/source/sparse_builder.cpp` | The members of `SparseBuilder` that are not templates |
| `blas/source/level1/`, `level2/`, `level3/` | `Level1`, `Level2` and `Level3`, one file per BLAS routine; a routine and its Hermitian or conjugated sibling share a file |
| `sparse/source/` | `SparseOps`, one file per routine; `doti.cpp` holds the three dot products and `spsymv.cpp` also `sphemv` |
| `logic/source/` | `LogicOps`, one file per comparison, logical operation or truth reduction |
| `elementwise/source/` | `ElementwiseOps`, one file per operation |
| `nonlinear/source/` | `NonLinearOps`, one file per activation and its gradient |
| `random/source/` | `RandomOps`, one file per distribution or initialiser |
| `reductions/source/` | `ReductionOps`, one file per reduction |
| `spatial/source/` | `SpatialOps`: the window, pooling, window product, channel and broadcast operations |

## What a file holds

A source is named after the method it defines, so the folder is the list of operations. The
files below hold more than one method, or none that records a task.

**Dense BLAS** (`blas/source/level1/`, `level2/`, `level3/`). `dot.cpp` holds `dot`, `dotu` and
`dotc`, and `ger.cpp` holds `ger`, `geru` and `gerc`. A symmetric routine shares its file with
its Hermitian sibling: `sbmv.cpp` with `hbmv`, `spmv.cpp` with `hpmv`, `spr.cpp` with `hpr`,
`spr2.cpp` with `hpr2`, `symv.cpp` with `hemv`, `syr.cpp` with `her`, `syr2.cpp` with `her2`,
`symm.cpp` with `hemm`, `syrk.cpp` with `herk` and `syr2k.cpp` with `her2k`.

**Sparse** (`sparse/source/`). `doti.cpp` holds `doti`, `dotui` and `dotci`, and `spsymv.cpp`
holds `spsymv` and `sphemv`.

**Spatial** (`spatial/source/`). `broadcast.cpp` holds `channel_affine`, `broadcast_spatial` and
`layout_convert`; `channels.cpp` holds `concat_channels` and `slice_channels`; `im2col.cpp` holds
both `im2col` overloads. `pool_avg.cpp`, `upsample_nearest.cpp` and `window_dot.cpp` each hold an
operation and its input gradient. `pool_max_backward.cpp` and `window_dot_weights_backward.cpp`
are gradients with a file of their own.

**Non-linear** (`nonlinear/source/`). Each file holds an activation and its `_backward`.

**Random** (`random/source/`). One file per distribution or initialiser. `set_seed.cpp` records
no task: it reseeds the stream the view draws from.

**Elementwise, logic, reductions** (`elementwise/source/`, `logic/source/`,
`reductions/source/`). One method per file, each named after its method.

## Notes by file

A source comment in this folder states an invariant in one line and cites this page. The reason
behind the invariant is below, under the path of the file that cites it. A reference to another
section of this page names that section's path; a reference to a public header names the README
of its module that holds the header's notes.

### `modules/operations/blas/source/level1/axpy.cpp`

`Level1::axpy` declares `y` in both the read set and the write set. `y = alpha * x + y` reads `y`
before it writes it, so `y` is an accumulator and not only a destination.

### `modules/operations/blas/source/level1/copy.cpp`

`Level1::copy` throws when the two dtypes differ. `?copy` is a BLAS routine, and every BLAS
routine is fixed to one precision. The message names `ElementwiseOps::convert`, the converting
copy, because a caller who wanted a conversion reaches this line.

`copy` is the only BLAS operation recorded with `Domain::All`. It moves elements and does no
arithmetic on them, so it can serve every element type, the integer ones included. An index tensor
that `iamax` or `argmax` produced can then be moved without passing through a float.

### `modules/operations/blas/source/level1/dot.cpp`

`dot.cpp` defines `Level1::dot`, `dotu` and `dotc`. `dot` is defined for real tensors and throws
`std::invalid_argument` for a complex one, naming `dotu` and `dotc`. `dotu` does not conjugate
and `dotc` conjugates `x`. All three record through one helper in the file.

### `modules/operations/blas/source/level1/iamax.cpp`

`Level1::iamax` requires its result tensor to be INT64, because the result is an index. The
integer dtypes exist so that this check can be made.

Before they existed, `iamax` wrote an `int64_t` into whatever tensor the caller passed. In every
test and every consumer that was a FLOAT32 tensor, so eight bytes went into a four-byte element.
`tests/operations/blas/test_blas/test_iamax.cpp` worked around it by allocating two floats, and left a
comment reading "Wait, different sizes!". The overrun was survivable while a tensor owned its
allocation. It stopped being survivable when SushiAI began packing tensors into arenas, because
the four bytes past the result then belong to a different tensor. `Tensor::region()` reported a
four-byte write, so the dependency tracker ordered nothing against the corruption either.

### `modules/operations/blas/source/level1/rot.cpp`

`Level1::rot` declares both operands as read and written. Each new `x_i` is built from the old
`x_i` and `y_i`, and each new `y_i` likewise.

### `modules/operations/blas/source/level1/swap.cpp`

`Level1::swap` declares both operands with `OperandSet::updates`: each is read and written, so
each sits in both access sets.

### `modules/operations/blas/source/level2/gemv.cpp`

`Level2::gemv` declares a read of `y` whenever `beta` is non-zero, because `y = alpha * op(A) * x +
beta * y` reads it then. The test is `BLASScalar::is_zero` on the caller's value, not on the value
narrowed to the element type. A `beta` that is non-zero as a double and flushes to zero in the
element type declares an edge the kernel does not need. It can never drop an edge the kernel does
need.

### `modules/operations/blas/source/level2/ger.cpp`

`Level2::ger` declares `A` as read and written, because the rank-1 update reads every element it
writes. The earlier write-only declaration still ordered tasks correctly, since the tracker
inserts a write-after-write edge anyway. Any pass that trusts the declared access set (aliasing
analysis, arena reuse, parallel scheduling) would have reasoned from a false premise.

### `modules/operations/blas/source/level2/syr.cpp`

`Level2::syr` declares `A` as read-modify-write for two reasons. `A` is an accumulator, and the
kernel touches only the selected triangle, so the other triangle has to survive the task
unchanged.

### `modules/operations/blas/source/level2/syr2.cpp`

`Level2::syr2` declares `A` as read-modify-write for the reasons given under
`modules/operations/blas/source/level2/syr.cpp` above: the update accumulates into `A` and touches one triangle.

### `modules/operations/blas/source/level2/trsv.cpp`

`Level2::trsv` dispatches into the TRSM kernels, `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/reference/level3/trsm.hpp` and
`modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level3/trsm.hpp`. TRSV is a TRSM with one right-hand-side column (n = 1), so a
second copy of the substitution would add nothing.

`trsv` passes `b` as an n x 1 matrix. A row-major row of that matrix has one element, so
`ldb = 1`; a column-major column needs `ldb >= n`.

`b` is declared with `updates`. Where `RunningSolution` needs a wide copy of the solution, the
operation leases it with `Internal::allocate_blas_scratch` and hands it to the SYCL kernel.

### `modules/operations/blas/source/level3/gemm.cpp`

`Level3::gemm` calls the fused GEMM with nothing fused into it. That avoids a second copy of the
shape derivation, the access set, the scratch lease, the dtype switch and the choice between the
reference and the SYCL kernel.
`GemmIdentityEpilogue` inlines to its argument, so the kernel this file records is byte for byte
the one it recorded when it called the kernel directly; see
`modules/operations/blas/README.md`, under `execute_gemm`.

`gemm` passes `RealAndComplex` explicitly. The seam defaults to `Real`, because an epilogue a
caller writes is usually real only. Plain GEMM has supported all five dtypes since SB-2, and the
identity epilogue is well-formed for each of them.

### `modules/operations/blas/source/level3/syrk.cpp`

`Level3::syrk` declares `C` as read-modify-write unconditionally, for two independent reasons. The
beta term reads `C` when beta != 0, and the triangular mask leaves the opposite triangle untouched
even when beta == 0. A byte interval cannot express "one triangle", so the declaration that is
true of the whole region is read and write.

### `modules/operations/blas/source/level3/syr2k.cpp`

`Level3::syr2k` and `her2k` declare `C` with `updates` whatever `beta` is, for the reason given
under `modules/operations/blas/source/level3/syrk.cpp` above: one triangle of `C` is kept.

### `modules/operations/logic/source/all.cpp`

`LogicOps::all` uses `sycl::reduction` on its SYCL path, which a floating-point sum in this library
must not. Under `ForceReference` it runs `all_reference`, a serial loop in the same file. The
SYCL specification leaves the combination order of a reduction unspecified, and that matters only
for a combiner whose result depends on the order. `all` reduces over two values, T(0) and T(1),
under `minimum`, which is associative, commutative and exact on that domain, so every order gives
the same bits. With `+` on floats the order shows in the result and can change with the launch
configuration.

The reduction is created with `initialize_to_identity`. Without that property a SYCL reduction
seeds itself from whatever the result tensor already holds and folds it into the answer. `all`
over a tensor of ones returned 0 whenever the caller reused a result that had held 0, and returned
whatever the allocator left behind when the result had never been written.

`all` is recorded with `Domain::RealAndIntegral`. A truth test means as much on an integer
tensor as on a float one, and an integer element carries the 1 or 0 this family produces without
a conversion. Complex is excluded: `logic_not_zero` would work on it, but `sycl::minimum` and
`sycl::maximum` over `std::complex` do not, and a reduction needs an ordering.

The dtype switch is `Dispatch::with_element_type`, the one the logic dispatch seam uses, so the
arms and the rejection by name are written once.

### `modules/operations/logic/source/any.cpp`

`LogicOps::any` uses `sycl::reduction` for the reason given under `modules/operations/logic/source/all.cpp`
above, with `maximum` in place of `minimum`. The combiner sees only T(0) and T(1), on which
`maximum` is associative, commutative and exact, so the unspecified combination order cannot
change a bit of the answer.

The reduction is created with `initialize_to_identity`, as in `all`. Without it, `any` over a
tensor of zeros returned 1 whenever the caller reused a result that had held 1.

`any` is recorded with `Domain::RealAndIntegral` and switches on the dtype through
`Dispatch::with_element_type`, both for the reasons given under `modules/operations/logic/source/all.cpp`.

### `modules/operations/elementwise/source/convert.cpp`

`convert.cpp` defines the copy that converts between dtypes. Three questions are answered in
`modules/operations/elementwise/README.md`, under `ElementwiseOps::convert`: why the operation
lives in the elementwise family, why the dtype pair is bounded to REAL x REAL, and why a
conversion to the same dtype is legal. This section records the mechanics.

`SushiBLAS::Dispatch` has no public seam for `convert`. Every other executor in `Dispatch` is
public because it takes a functor the caller writes. That lets a consumer fuse several math
operations into one kernel without writing SYCL, and it is why those headers were promoted out of
`source/`. `convert` has no functor: the conversion is `static_cast`, fixed by the pair of dtypes, so
a seam would expose a parameter that does not exist. Keeping the implementation in this
translation unit also confines the nine instantiations to one object file, which is the bound the
header's notes argue for.

`record_convert` is one of the nine arms. It is an ordinary function template and not a lambda,
because the dispatch in `dispatch_destination` and `ElementwiseOps::convert` has to be two plain
`switch` statements. A generic lambda nested inside a generic lambda crashes this project's clang
during instantiation (see `modules/engine/engine/README.md`, under `execute_typed_task`), and
an operation over two types is where that nesting would otherwise be written.

`record_convert` launches through `Launch::indexed` on the SYCL path, and runs the same functor
through `Serial::indexed` under `ForceReference`. `Launch::map` has a `static_assert` that
every operand shares one element type, and the assertion is correct: the wide pass of `map` loads
and stores `sycl::vec<T, W>` at one width, and two operands of different widths have no common W.
`convert` therefore gets the device split, flat on a CPU and grid-stride on a GPU, and no vector
path. A vector path would read half of every second element.

The conversion is `static_cast`, which rounds to nearest even under IEEE 754 for every pair here.
An fp32 value past 65504 therefore arrives as an fp16 infinity, not as a wrapped or clamped
number, and an infinity is what a loss-scaling backoff is built to detect.

FLOAT64 to HALF may round twice. `sycl::half` has no direct double constructor on every
toolchain, so that pair may go double to float to half, and a double rounding differs from one
correctly rounded step for the few values that land exactly halfway after the first step. This
was not measured. Nothing relies on it, and it is off the mixed-precision path, which converts
between fp32 and fp16.

`dispatch_destination` has three arms and is instantiated once per source type, nine
instantiations in total. The arms are written out and not generated, so a reader can count them.

The `switch` in `dispatch_destination` ends in a throw that cannot be reached, because the caller
rejected every non-REAL dtype when it recorded the task. A switch that can fall out of the bottom
is how a silent no-op gets in; `modules/operations/blas/README.md`, under
`execute_blas_task`, gives the same reason for its own.

The dtype check in `ElementwiseOps::convert` names both dtypes in its message. The caller's first
question is which one was wrong, and the pair is what was rejected. It throws
`std::invalid_argument`, as every record-time check in this file does.

`convert` refuses a non-contiguous operand by name. The kernel walks a flat index, so it would
read a view whose elements are not laid out flatly at the wrong addresses, and nothing would
report the wrong answer.

`convert` refuses operands that overlap in memory. Each work-item reads one element and writes
another of a different width, so an overlap is a race between work-items. It is not the
well-defined same-index aliasing that the in-place callers of `Launch::map` rely on. The
same-dtype case is refused too: equal widths do not make converting a tensor onto itself useful,
and a rule with an exception is one more thing for a caller to remember.

The access sets hold `Tensor::region()`, never `data()`. A `BufferRegion` converts implicitly
from a bare pointer, and a bare pointer widens to the whole allocation. Two views of one arena
would become two unrelated resources, `overlaps` would answer false, and the tracker would emit no
edge. `modules/README.md`, under `Tensor::region`, describes the region, and
`docs/reference/MAINPAGE.md` records that this defect has cost the stack two training runs.

The `switch` in `ElementwiseOps::convert` is the outer half of the dispatch and picks the source
type. Together with `dispatch_destination` it makes two plain switches over function templates,
nine instantiations, and no lambda nested in a lambda.

### `modules/operations/random/source/constant.cpp`

`RandomOps::constant` has integer arms because integer tensors need known values: an index tensor
gets one before an operation writes over it, and a comparison operand gets a threshold. The
double parameter is truncated to the integer type, by the same convention the HALF and FLOAT32
arms follow: the caller states a value, and the tensor's dtype decides how it is stored.

### `modules/operations/random/source/normal.cpp`

`RandomOps::normal` draws the real and imaginary halves of a complex tensor as one flat run of 2n
reals. The position of a component in that run fixes the Philox block and word it is drawn from,
which keeps a complex fill reproducible element for element. Both Box-Muller outputs of a uniform
pair are used: a block yields four FLOAT32 or two FLOAT64 normals.

### `modules/operations/random/source/orthogonal.cpp`

`RandomOps::orthogonal` first fills the tensor with N(0,1). The Gaussian ensemble is orthogonally
invariant, so orthonormalising a Gaussian matrix yields a uniformly distributed orthogonal matrix
and not a biased one. The fill is recorded as its own task, and the graph orders the sweep after
it because both declare the same region.

`orthogonal` orthonormalises along the shorter axis: the rows of a wide matrix, the columns of a
tall one. At most min(rows, cols) vectors can be mutually orthogonal. The result satisfies
Q Q^T = gain^2 I in the first case and Q^T Q = gain^2 I in the second, the same convention as
PyTorch's `orthogonal_`.

The sweep itself is `Kernels::SYCL::Random::orthonormalize`, one work-group per vector, or
`Kernels::Reference::Random::orthonormalize` under `ForceReference`; see `kernels/README.md`.
`orthogonal` calls `TaskRecorder::require_host_reachable` before it records the fill, so a refused
call records nothing and draws nothing. It takes its steps from `t.strides`, not from arithmetic on
rows and columns, so it is correct for `RowMajor` and `ColumnMajor` with no special case. A
column-major matrix walks the two axes in the other order, and the transpose of an orthogonal matrix
is orthogonal.

### `modules/operations/random/source/poisson.cpp`

`RandomOps::poisson` checks lambda against [0, `K_MAX_POISSON_LAMBDA`] = [0, 1e7] at record
time, computes the per-lambda constants once on the host with
`Kernels::SYCL::Random::make_poisson_params`, and fills one element per work-item with
`draw_poisson`. Below lambda 10 the draw is Knuth's multiplication loop; from 10 up it is
Hoermann's PTRS. The math module README explains the bound and `kernels/README.md` the two
methods. It throws `UnsupportedExecution` when `t` has elements and its device lacks the fp64
aspect.

### `modules/operations/random/source/shuffle.cpp`

`RandomOps::shuffle` runs Fisher-Yates sequentially. Each step draws its swap partner from a range
that depends on the step index, so the permutation is uniform only if the steps happen in order. A
`single_task` keeps that order on the device and leaves the tensor's data where it is; under
`ForceReference` the same `FisherYates` functor runs on the host. The
parallel alternative tags every element with a random key and sorts, which needs a second buffer
of the full size and a device sort this library does not have.

The swap is instantiated per dtype and does not swap raw bytes, so a `std::complex<double>` moves
as one value and cannot tear into two halves.

`shuffle` declares its tensor as read and written, unlike the distributions. A shuffle permutes
what is already there, so the task depends on whatever produced the contents and not only on
earlier writers.

### `modules/operations/random/source/uniform.cpp`

`RandomOps::uniform` fills a complex tensor as `modules/operations/random/source/normal.cpp` above
describes: one flat run of 2n reals, read four FLOAT32 or two FLOAT64 values per Philox block.

### `modules/operations/reductions/source/argmax.cpp`

`ReductionOps::argmax` folds (value, index) pairs through `Dispatch::ArgReductionDispatch` over
`Reduce::RunningArgMax`, in the fixed-order shape of `Reduce::reduce_indexed`. The accumulator's
order is total: the larger value wins, then the lower index, and its NaN rule is stated on
`RunningArgMax::combine`. Because the order is total the combination order cannot change the
answer, so the lowest-index maximum wins a tie on every device and launch shape. Under
`ForceReference` the same accumulator folds in index order on the host.

It does not go through `sycl::reduction`: custom reductions behave differently on the OpenCL and
Level-Zero backends, and the tie-breaking rule must not change with the backend.

The result goes into an INT64 tensor through `execute_index_reduction`, not into a tensor of the
input's dtype through `execute_reduction`. It was once stored as `static_cast<T>(idx)`, a float. A
float is exact below 2^24. Above that, two adjacent indices round to the same value, so the
operation returned a wrong index on the tensor sizes a training run calls it on. In HALF the
ceiling is 2048.

### `modules/operations/reductions/source/argmin.cpp`

`ReductionOps::argmin` is `argmax` with `Reduce::RunningArgMin`: the lowest-index minimum always
wins a tie, and the result does not depend on the combination order, for the reasons given under
`modules/operations/reductions/source/argmax.cpp` above.

The result goes into an INT64 tensor through `execute_index_reduction`. The same section explains
why a result stored in the input's dtype was wrong above 2^24, and above 2048 in HALF.

### `modules/operations/reductions/source/count_nonfinite.cpp`

`modules/operations/reductions/README.md`, under `ReductionOps::count_nonfinite`, answers two
questions about this operation. The first is why the result is an INT64 count and not a flag in
the input's dtype. The second is why `asum`, which would almost have served, reports false
positives on the mixed-precision input this operation exists for.

The operation is one call to the counting seam with a dispatcher whose transform is a predicate.
`Reduce::reduce` gives its SYCL path the library's fixed-order two-stage fold, and
`Serial::reduce` gives its host path the same fold in index order, both over `Reduce::ExactSum`.
The accumulation type is in those functions' signatures so that no operation has to write a
second copy of the fold.

The scratch is sized from the device the data lives on, as in `execute_folding_reduction`.
`Reduce::plan` clamps to whatever scratch exists at launch, so a disagreement between the two
costs performance and never correctness.

The predicate is `sycl::isfinite`, not `x != x`. `x != x` is the usual NaN test, and it is half of
a non-finite test: an infinity compares equal to itself, so `x != x` would pass every overflowed
fp16 gradient. `isfinite` is false for both. Substituting `x == x` here fails 13 of this
operation's 23 tests: every infinity case at every dtype, and none of the NaN cases. That is why
the suite tests the two separately.

### `modules/operations/reductions/source/max.cpp`

`ReductionOps::max` propagates NaN, as NumPy's `max` does: if any element is NaN, the result is
NaN, whatever the element's position and whatever launch shape the device gets. It folds through
`Dispatch::execute_folding_reduction` with `Reduce::NaNPropagatingMax` from
`modules/foundation/reduction/include/SushiBLAS/reduction/reduce_1d.hpp`. A plain `<` comparison is false against a NaN, so
under it a NaN would survive only when it arrives first. `NaNPropagatingMax` returns NaN from
`combine` when either side is NaN and stops changing once `accumulate` has seen one, so every fold
order gives NaN.

An empty tensor throws `std::invalid_argument` when the call is recorded.

### `modules/operations/reductions/source/max_along.cpp`

`ReductionOps::max_along` goes through the folding seam with `Reduce::NaNPropagatingMax`, the
accumulator the whole-tensor `max` uses, so a NaN in a segment makes that segment's result NaN
and a per-axis maximum agrees with a whole-tensor one. The axis family has one shape that serves
every accumulator, because `sycl::reduction` has no segmented form. `min_along.cpp` is the same
call with `Reduce::NaNPropagatingMin`. See `docs/architecture/FIXED_ORDER_FOLD.md`, under
`NaNPropagatingMax`.

### `modules/operations/reductions/source/mean.cpp`

`ReductionOps::mean` divides in the finalizer, so it costs the same number of kernel launches as
`sum`. An earlier version submitted a second `single_task` only to divide one scalar by one
integer.

### `modules/operations/reductions/source/mean_spatial.cpp`

`ReductionOps::mean_spatial` is built from the same three pieces as `mean`: a compensated sum,
no transform, and `v / n` as the finalizer. It runs them over the segmented seam, where `mean`
uses the whole-tensor one. The new part is the span. A fold is identified by a pair `(n, c)`,
which `spatial_span_for` builds and the second segment axis of `AxisSpan` carries.

### `modules/operations/reductions/source/min.cpp`

`ReductionOps::min` mirrors `max` above, with `Reduce::NaNPropagatingMin`: any NaN element makes
the result NaN in every fold order, and an empty tensor throws `std::invalid_argument`.

### `modules/operations/reductions/source/nrm2.cpp`

`ReductionOps::nrm2` computes the scaling 2-norm that BLAS's xNRM2 specifies. An earlier version
computed `sqrt(sum(x*x))`. The call has no transform and no finalizer, because the squaring and
the square root are safe only inside a running scale. `docs/architecture/FIXED_ORDER_FOLD.md`, under
`ScaledSumOfSquares`, describes the algorithm and says why it belongs in the accumulator. The
launch count did not change: one launch on the single-pass shape and two on the two-stage one.

### `modules/operations/reductions/source/prod.cpp`

`ReductionOps::prod` accumulates with `RunningProduct`, not `CompensatedSum`. Neumaier's
correction recovers the bits an addition rounds off and does nothing for a multiplication. What
`prod` shares with the additive reductions is the fixed combination order, which is why it left
`sycl::reduction` as they did: floating-point multiplication is no more associative than addition.

### `modules/operations/reductions/source/sum.cpp`

`ReductionOps::sum` depends on the combination order, so it goes through the folding seam and not
`sycl::reduction`. `docs/architecture/FIXED_ORDER_FOLD.md` says why the
unspecified order is a correctness problem for `+` and not for `max`. `sum` has no transform and
no finalizer. The other reductions on the folding seam are this call with a transform, a
finalizer or a different accumulator added.

### `modules/operations/reductions/source/sum_along.cpp`

`ReductionOps::sum_along` depends on the combination order, so it takes the same
Neumaier-compensated fixed-order fold as the whole-tensor `sum`, in its segmented shape. It has no
transform and no finalizer. A per-axis `mean` is this call with `v / n` as the finalizer and costs
no extra launch.

### `modules/operations/spatial/source/broadcast.cpp`

None of the three operations in `broadcast.cpp` has a sliding window, so none of them calls
`Spatial::window_span_for`. That helper validates every window origin against a `ConvGeometry`.
The geometry these operations would have to invent to satisfy it, a 1x1 window at stride 1, is six
fields a reader would have to rule out before trusting the call. Each operation builds its span
from the operands' `shape[]` and `strides[]`, and sets `out_h` and `out_w` equal to `in_h` and
`in_w`, because these are copies and not reductions over a window.

`SpatialOps::channel_affine` reads `gamma` and `beta` through their own strides and does not
assume they are contiguous. A parameter vector is routinely a view into a packed parameter arena.

The input of `SpatialOps::broadcast_spatial` is rank-2, so it has no row or column stride to
record. The `in_*` extents are filled in only so that the span describes one shape, and the
input's two real strides travel to the kernels as arguments.

`broadcast_spatial` narrows `scale` inside the dispatch closure and not at the seam. The factor
has to be a `T` for both kernels, and `T` is not known until the closure is instantiated.

`SpatialOps::layout_convert` requires the same logical shape on both sides. What differs is which
stride is largest. A caller expresses the transpose by handing in an output tensor with different
strides, never by a direction flag, so an output whose shape is permuted is a caller error and not
a request for NCHW.

### `modules/operations/spatial/source/channels.cpp`

`channel_copy_span` builds its span by hand and does not call `Spatial::window_span_for`, which
validates window origins against a `Spatial::ConvGeometry` that a plain copy does not have. For
the same reason it sets `out_h` and `out_w` to the input extents: the operation visits every pixel
of one shared grid, so there is no second extent derived from a window.

`span.channels` stays at the source's channel count and neither kernel reads it. The width of the
copy travels as an explicit argument.

The channel copies are recorded with `Domain::All`. They move elements and do no arithmetic on
them, so every element type the library has can be copied.

### `modules/operations/spatial/source/col2im.cpp`

`SpatialOps::col2im` calls the extents overload of `window_span_for`. `image` takes the input role
in the span: it is the rank-4 operand whose H, W and strides the window walks. The other operand
is a rank-2 column matrix with no extents to read.

### `modules/operations/spatial/source/im2col.cpp`

`SpatialOps::im2col` calls the extents overload of `window_span_for`, not the two-tensor one. The
other operand is a rank-2 column matrix, so no output image exists whose shape could supply H_o
and W_o.

The overload of `im2col` that derives the extents assumes symmetric padding, and can assume
nothing else. `ConvGeometry` carries `pad_left` and no `pad_right`, so W_o can be recovered only
by taking the leading pad for both sides. H_o then follows from the row count. A caller whose
padding is asymmetric must call the overload with explicit extents, which exists for that case.
The 2x2 stride-1 pool of YOLOv3-tiny is the one in scope.

### `modules/operations/spatial/source/pool_avg.cpp`

In `SpatialOps::pool_avg_backward`, `dx` takes the input role and `dy` the output role, the
assignment the forward made. The geometry must therefore be the forward's.

### `modules/operations/spatial/source/pool_max.cpp`

`SpatialOps::pool_max` requires its index buffer to be INT64 and checks that by name; the kernel
never reinterprets another dtype. fp32 represents integers exactly only below 2^24, and a
224x224x64 feature map passes that at batch 6.

The dtype that drives the switch is the value tensor's. The index pointer is captured as
`int64_t*` outside the switch, because the operation fixes its type and the operands do not.

### `modules/operations/spatial/source/pool_max_backward.cpp`

In `SpatialOps::pool_max_backward`, `dx` takes the input role in the span, since it has the
image's H, W and strides, and `dy` takes the output role. That is the assignment the forward made,
so the geometry must be the forward's exactly.

### `modules/operations/spatial/source/window_dot.cpp`, `window_dot_weights_backward.cpp`

`SpatialOps::window_dot` builds its span with the two-operand `window_span_for` and its taps with
`tap_span_for`, and refuses an output that shares a byte with an input, because the kernel is a
gather that reads while it writes.

In `window_dot_backward`, `dx` takes the input role in the span and `dy` the output role, the
assignment the forward made, so the geometry must be the forward's. In
`window_dot_weights_backward`, `in` takes the input role and `dy` the output role for the same
reason, and `dweights` is described by `tap_span_for` as `weights` is in the forward.

All three are recorded with `Domain::Real`.

### `modules/operations/spatial/source/upsample_nearest.cpp`

`upsample_span_for` does not call `Spatial::window_span_for`. That function needs a
`Spatial::ConvGeometry`, and the only one that fits here is the default 1x1 window at stride 1.
Under that geometry the window-origin check reads "output row `out_h - 1` starts at input row
`out_h - 1`" and rejects every output larger than the input, which is every upsampling.
`upsample_span_for` reads the strides off the tensors and checks the rule this operation has: each
output extent is an exact `scale` multiple of the input's. `span.geometry` stays default, and no
upsample kernel reads it.

In `SpatialOps::upsample_nearest_backward`, `dx` takes the input role and `dy` the output role,
the assignment the forward made. The scales must therefore be the forward's.

### `modules/operations/io/source/io.cpp`

`IO::save_npy` writes a `descr` for every dtype it knows and throws for any other. The chain once
fell through for HALF and wrote a header with an empty `descr` and a payload numpy could not read:
a corrupt file with no error at either end.

The text rendering decodes every element to double before printing it. INT32 survives that
exactly; INT64 loses precision past 2^53. That is acceptable for a dump a person reads
and nowhere else. `IO::save_npy` writes the raw bytes, so no saved value passes through the text
rendering.

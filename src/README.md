# src

The translation units of the SushiBLAS library. Each file under `engine/` defines the methods of
one operation: it checks the operands, declares the tensors the task reads and writes in an
`OperandSet`, and records the task through the view's `TaskRecorder`. Every method returns the
`TaskHandle` of the task it recorded. The kernels a task runs are in `kernels/` and under
`include/SushiBLAS/kernels/`.

| Path | Defines |
| --- | --- |
| `engine.cpp` | The `Engine` constructor and `Engine::execute` |
| `io.cpp` | `IO`: the `.sushi`, raw binary and `.npy` readers and writers, and the text rendering |
| `sparse_builder.cpp` | The members of `SparseBuilder` that are not templates |
| `engine/blas/level1/`, `level2/`, `level3/` | `Level1`, `Level2` and `Level3`, one file per BLAS routine; a routine and its Hermitian or conjugated sibling share a file |
| `engine/sparse/` | `SparseOps`, one file per routine; `doti.cpp` holds the three dot products and `spsymv.cpp` also `sphemv` |
| `engine/logic/` | `LogicOps`, one file per comparison, logical operation or truth reduction |
| `engine/math/elementwise/` | `ElementwiseOps`, one file per operation |
| `engine/math/nonlinear/` | `NonLinearOps`, one file per activation and its gradient |
| `engine/math/random/` | `RandomOps`, one file per distribution or initialiser |
| `engine/math/reductions/` | `ReductionOps`, one file per reduction |
| `engine/spatial/` | `SpatialOps`: the window, pooling, window product, channel and broadcast operations |

## Files

| File | Defines |
| --- | --- |
| `engine.cpp` | Defines the Engine constructor and Engine::execute |
| `io.cpp` | Defines IO, which saves and loads tensors as .sushi, raw binary and .npy files and renders them as text |
| `sparse_builder.cpp` | Defines the allocation, the host transfers and the dense compression of SparseBuilder |

BLAS level 1:

| File | Defines |
| --- | --- |
| `engine/blas/level1/asum.cpp` | Defines Level1::asum, which records the BLAS ASUM task |
| `engine/blas/level1/axpy.cpp` | Defines Level1::axpy, which records the BLAS AXPY task |
| `engine/blas/level1/copy.cpp` | Defines Level1::copy, which records the BLAS COPY task |
| `engine/blas/level1/dot.cpp` | Defines Level1::dot, dotu and dotc, which record the BLAS DOT, DOTU and DOTC tasks |
| `engine/blas/level1/iamax.cpp` | Defines Level1::iamax, which records the BLAS IAMAX task |
| `engine/blas/level1/nrm2.cpp` | Defines Level1::nrm2, which records the BLAS NRM2 task |
| `engine/blas/level1/rot.cpp` | Defines Level1::rot, which records the BLAS ROT task |
| `engine/blas/level1/rotg.cpp` | Defines Level1::rotg, which records the BLAS ROTG task |
| `engine/blas/level1/rotm.cpp` | Defines Level1::rotm, which records the BLAS ROTM task |
| `engine/blas/level1/rotmg.cpp` | Defines Level1::rotmg, which records the BLAS ROTMG task |
| `engine/blas/level1/scal.cpp` | Defines Level1::scal, which records the BLAS SCAL task |
| `engine/blas/level1/swap.cpp` | Defines Level1::swap, which records the BLAS SWAP task |

BLAS level 2:

| File | Defines |
| --- | --- |
| `engine/blas/level2/gbmv.cpp` | Defines Level2::gbmv, which records the BLAS GBMV task |
| `engine/blas/level2/gemv.cpp` | Defines Level2::gemv, which records the BLAS GEMV task |
| `engine/blas/level2/ger.cpp` | Defines Level2::ger, Level2::geru and Level2::gerc, which record the BLAS GER, GERU and GERC tasks |
| `engine/blas/level2/sbmv.cpp` | Defines Level2::sbmv and Level2::hbmv, which record the BLAS SBMV and HBMV tasks |
| `engine/blas/level2/spmv.cpp` | Defines Level2::spmv and Level2::hpmv, which record the BLAS SPMV and HPMV tasks |
| `engine/blas/level2/spr.cpp` | Defines Level2::spr and Level2::hpr, which record the BLAS SPR and HPR tasks |
| `engine/blas/level2/spr2.cpp` | Defines Level2::spr2 and Level2::hpr2, which record the BLAS SPR2 and HPR2 tasks |
| `engine/blas/level2/symv.cpp` | Defines Level2::symv and Level2::hemv, which record the BLAS SYMV and HEMV tasks |
| `engine/blas/level2/syr.cpp` | Defines Level2::syr and Level2::her, which record the BLAS SYR and HER tasks |
| `engine/blas/level2/syr2.cpp` | Defines Level2::syr2 and Level2::her2, which record the BLAS SYR2 and HER2 tasks |
| `engine/blas/level2/tbmv.cpp` | Defines Level2::tbmv, which records the BLAS TBMV task |
| `engine/blas/level2/tbsv.cpp` | Defines Level2::tbsv, which records the BLAS TBSV task |
| `engine/blas/level2/tpmv.cpp` | Defines Level2::tpmv, which records the BLAS TPMV task |
| `engine/blas/level2/tpsv.cpp` | Defines Level2::tpsv, which records the BLAS TPSV task |
| `engine/blas/level2/trmv.cpp` | Defines Level2::trmv, which records the BLAS TRMV task |
| `engine/blas/level2/trsv.cpp` | Defines Level2::trsv, which records the BLAS TRSV task |

BLAS level 3:

| File | Defines |
| --- | --- |
| `engine/blas/level3/gemm.cpp` | Defines Level3::gemm, which records the BLAS GEMM task |
| `engine/blas/level3/symm.cpp` | Defines Level3::symm and Level3::hemm, which record the BLAS SYMM and HEMM tasks |
| `engine/blas/level3/syr2k.cpp` | Defines Level3::syr2k and Level3::her2k, which record the BLAS SYR2K and HER2K tasks |
| `engine/blas/level3/syrk.cpp` | Defines Level3::syrk and Level3::herk, which record the BLAS SYRK and HERK tasks |
| `engine/blas/level3/trsm.cpp` | Defines Level3::trsm, which records the BLAS TRSM task |

Sparse:

| File | Defines |
| --- | --- |
| `engine/sparse/axpyi.cpp` | Defines SparseOps::axpyi, which records the sparse AXPYI task |
| `engine/sparse/doti.cpp` | Defines SparseOps::doti, dotui and dotci, which record the sparse dot product tasks |
| `engine/sparse/gthr.cpp` | Defines SparseOps::gthr, which records the sparse gather task |
| `engine/sparse/gthrz.cpp` | Defines SparseOps::gthrz, which records the sparse gather-and-zero task |
| `engine/sparse/roti.cpp` | Defines SparseOps::roti, which records the sparse plane rotation task |
| `engine/sparse/sctr.cpp` | Defines SparseOps::sctr, which records the sparse scatter task |
| `engine/sparse/spmm.cpp` | Defines SparseOps::spmm, which records the sparse-times-dense matrix product task |
| `engine/sparse/spmv.cpp` | Defines SparseOps::spmv, which records the sparse matrix-vector product task |
| `engine/sparse/spsymv.cpp` | Defines SparseOps::spsymv and sphemv, which record the one-triangle product tasks |
| `engine/sparse/sptrsv.cpp` | Defines SparseOps::sptrsv, which records the sparse triangular solve task |

Spatial:

| File | Defines |
| --- | --- |
| `engine/spatial/broadcast.cpp` | Defines SpatialOps::channel_affine, broadcast_spatial and layout_convert, which record the three broadcast copies |
| `engine/spatial/channels.cpp` | Defines SpatialOps::concat_channels and slice_channels, which record the copy of a run of channels |
| `engine/spatial/col2im.cpp` | Defines SpatialOps::col2im, which records the scatter of a column matrix back onto an image |
| `engine/spatial/im2col.cpp` | Defines the two SpatialOps::im2col overloads, which record the gather of image windows into a column matrix |
| `engine/spatial/pool_avg.cpp` | Defines SpatialOps::pool_avg and pool_avg_backward, which record average pooling and its gradient |
| `engine/spatial/pool_max.cpp` | Defines SpatialOps::pool_max, which records max pooling with its argmax indices |
| `engine/spatial/pool_max_backward.cpp` | Defines SpatialOps::pool_max_backward, which records the gradient of max pooling |
| `engine/spatial/upsample_nearest.cpp` | Defines SpatialOps::upsample_nearest and upsample_nearest_backward, which record nearest-neighbour upsampling and its gradient |
| `engine/spatial/window_dot.cpp` | Defines SpatialOps::window_dot and window_dot_backward, which record the per-channel weighted window sum and its input gradient |
| `engine/spatial/window_dot_weights_backward.cpp` | Defines SpatialOps::window_dot_weights_backward, which records the weight gradient of the per-channel weighted window sum |

Logic:

| File | Defines |
| --- | --- |
| `engine/logic/all.cpp` | Defines LogicOps::all, which records the reduction that tests whether every element is non-zero |
| `engine/logic/any.cpp` | Defines LogicOps::any, which records the reduction that tests whether any element is non-zero |
| `engine/logic/equal.cpp` | Defines LogicOps::equal, which records the elementwise equal task over two tensors |
| `engine/logic/greater.cpp` | Defines LogicOps::greater, which records the elementwise greater task over two tensors |
| `engine/logic/greater_equal.cpp` | Defines LogicOps::greater_equal, which records the elementwise greater_equal task over two tensors |
| `engine/logic/less.cpp` | Defines LogicOps::less, which records the elementwise less task over two tensors |
| `engine/logic/less_equal.cpp` | Defines LogicOps::less_equal, which records the elementwise less_equal task over two tensors |
| `engine/logic/logical_and.cpp` | Defines LogicOps::logical_and, which records the elementwise logical_and task over two tensors |
| `engine/logic/logical_not.cpp` | Defines LogicOps::logical_not, which records the elementwise logical_not task |
| `engine/logic/logical_or.cpp` | Defines LogicOps::logical_or, which records the elementwise logical_or task over two tensors |
| `engine/logic/logical_xor.cpp` | Defines LogicOps::logical_xor, which records the elementwise logical_xor task over two tensors |
| `engine/logic/not_equal.cpp` | Defines LogicOps::not_equal, which records the elementwise not_equal task over two tensors |
| `engine/logic/where.cpp` | Defines LogicOps::where, which records the elementwise choice between two tensors by a condition |

Elementwise:

| File | Defines |
| --- | --- |
| `engine/math/elementwise/abs.cpp` | Defines ElementwiseOps::abs, which records the in-place elementwise abs task |
| `engine/math/elementwise/acos.cpp` | Defines ElementwiseOps::acos, which records the in-place elementwise acos task |
| `engine/math/elementwise/acosh.cpp` | Defines ElementwiseOps::acosh, which records the in-place elementwise acosh task |
| `engine/math/elementwise/add.cpp` | Defines ElementwiseOps::add, which records the elementwise add task over two tensors |
| `engine/math/elementwise/asin.cpp` | Defines ElementwiseOps::asin, which records the in-place elementwise asin task |
| `engine/math/elementwise/asinh.cpp` | Defines ElementwiseOps::asinh, which records the in-place elementwise asinh task |
| `engine/math/elementwise/atan.cpp` | Defines ElementwiseOps::atan, which records the in-place elementwise atan task |
| `engine/math/elementwise/atanh.cpp` | Defines ElementwiseOps::atanh, which records the in-place elementwise atanh task |
| `engine/math/elementwise/ceil.cpp` | Defines ElementwiseOps::ceil, which records the in-place elementwise ceil task |
| `engine/math/elementwise/clamp.cpp` | Defines ElementwiseOps::clamp, which records the in-place elementwise clamp task |
| `engine/math/elementwise/convert.cpp` | Defines ElementwiseOps::convert, which records the dtype-converting copy between real tensors |
| `engine/math/elementwise/cos.cpp` | Defines ElementwiseOps::cos, which records the in-place elementwise cos task |
| `engine/math/elementwise/cosh.cpp` | Defines ElementwiseOps::cosh, which records the in-place elementwise cosh task |
| `engine/math/elementwise/div.cpp` | Defines ElementwiseOps::div, which records the elementwise div task over two tensors |
| `engine/math/elementwise/exp.cpp` | Defines ElementwiseOps::exp, which records the in-place elementwise exp task |
| `engine/math/elementwise/floor.cpp` | Defines ElementwiseOps::floor, which records the in-place elementwise floor task |
| `engine/math/elementwise/fmod.cpp` | Defines ElementwiseOps::fmod, which records the elementwise fmod task over two tensors |
| `engine/math/elementwise/log.cpp` | Defines ElementwiseOps::log, which records the in-place elementwise log task |
| `engine/math/elementwise/max.cpp` | Defines ElementwiseOps::max, which records the elementwise max task over two tensors |
| `engine/math/elementwise/min.cpp` | Defines ElementwiseOps::min, which records the elementwise min task over two tensors |
| `engine/math/elementwise/mul.cpp` | Defines ElementwiseOps::mul, which records the elementwise mul task over two tensors |
| `engine/math/elementwise/neg.cpp` | Defines ElementwiseOps::neg, which records the in-place elementwise neg task |
| `engine/math/elementwise/pow.cpp` | Defines ElementwiseOps::pow, which records the in-place elementwise pow task |
| `engine/math/elementwise/reciprocal.cpp` | Defines ElementwiseOps::reciprocal, which records the in-place elementwise reciprocal task |
| `engine/math/elementwise/remainder.cpp` | Defines ElementwiseOps::remainder, which records the elementwise remainder task over two tensors |
| `engine/math/elementwise/round.cpp` | Defines ElementwiseOps::round, which records the in-place elementwise round task |
| `engine/math/elementwise/sin.cpp` | Defines ElementwiseOps::sin, which records the in-place elementwise sin task |
| `engine/math/elementwise/sinh.cpp` | Defines ElementwiseOps::sinh, which records the in-place elementwise sinh task |
| `engine/math/elementwise/sqrt.cpp` | Defines ElementwiseOps::sqrt, which records the in-place elementwise sqrt task |
| `engine/math/elementwise/square.cpp` | Defines ElementwiseOps::square, which records the in-place elementwise square task |
| `engine/math/elementwise/sub.cpp` | Defines ElementwiseOps::sub, which records the elementwise sub task over two tensors |
| `engine/math/elementwise/tan.cpp` | Defines ElementwiseOps::tan, which records the in-place elementwise tan task |

Non-linear:

| File | Defines |
| --- | --- |
| `engine/math/nonlinear/elu.cpp` | Defines NonLinearOps::elu and NonLinearOps::elu_backward, which record the elu activation and its gradient |
| `engine/math/nonlinear/gelu.cpp` | Defines NonLinearOps::gelu and NonLinearOps::gelu_backward, which record the gelu activation and its gradient |
| `engine/math/nonlinear/leaky_relu.cpp` | Defines NonLinearOps::leaky_relu and NonLinearOps::leaky_relu_backward, which record the leaky_relu activation and its gradient |
| `engine/math/nonlinear/relu.cpp` | Defines NonLinearOps::relu and NonLinearOps::relu_backward, which record the relu activation and its gradient |
| `engine/math/nonlinear/sigmoid.cpp` | Defines NonLinearOps::sigmoid and NonLinearOps::sigmoid_backward, which record the sigmoid activation and its gradient |
| `engine/math/nonlinear/silu.cpp` | Defines NonLinearOps::silu and NonLinearOps::silu_backward, which record the silu activation and its gradient |
| `engine/math/nonlinear/softplus.cpp` | Defines NonLinearOps::softplus and NonLinearOps::softplus_backward, which record the softplus activation and its gradient |
| `engine/math/nonlinear/tanh.cpp` | Defines NonLinearOps::tanh and NonLinearOps::tanh_backward, which record the tanh activation and its gradient |

Random:

| File | Defines |
| --- | --- |
| `engine/math/random/bernoulli.cpp` | Defines RandomOps::bernoulli, which records the Bernoulli fill of a tensor |
| `engine/math/random/constant.cpp` | Defines RandomOps::constant, which records the constant fill of a tensor |
| `engine/math/random/discrete_uniform.cpp` | Defines RandomOps::discrete_uniform, which records the discrete uniform fill of a tensor |
| `engine/math/random/exponential.cpp` | Defines RandomOps::exponential, which records the exponential fill of a tensor |
| `engine/math/random/he_normal.cpp` | Defines RandomOps::he_normal, which records the He normal fill of a tensor |
| `engine/math/random/he_uniform.cpp` | Defines RandomOps::he_uniform, which records the He uniform fill of a tensor |
| `engine/math/random/log_normal.cpp` | Defines RandomOps::log_normal, which records the log-normal fill of a tensor |
| `engine/math/random/normal.cpp` | Defines RandomOps::normal, which records the normal fill of a tensor |
| `engine/math/random/orthogonal.cpp` | Defines RandomOps::orthogonal, which records the orthogonal initialisation of a matrix |
| `engine/math/random/poisson.cpp` | Defines RandomOps::poisson, which records the Poisson fill of a tensor |
| `engine/math/random/set_seed.cpp` | Defines RandomOps::set_seed, which reseeds the stream the view draws from |
| `engine/math/random/shuffle.cpp` | Defines RandomOps::shuffle, which records the in-place shuffle of a tensor's elements |
| `engine/math/random/truncated_normal.cpp` | Defines RandomOps::truncated_normal, which records the truncated normal fill of a tensor |
| `engine/math/random/uniform.cpp` | Defines RandomOps::uniform, which records the uniform fill of a tensor |
| `engine/math/random/xavier_normal.cpp` | Defines RandomOps::xavier_normal, which records the Xavier normal fill of a tensor |
| `engine/math/random/xavier_uniform.cpp` | Defines RandomOps::xavier_uniform, which records the Xavier uniform fill of a tensor |

Reductions:

| File | Defines |
| --- | --- |
| `engine/math/reductions/argmax.cpp` | Defines ReductionOps::argmax, which records the argmax reduction |
| `engine/math/reductions/argmin.cpp` | Defines ReductionOps::argmin, which records the argmin reduction |
| `engine/math/reductions/asum.cpp` | Defines ReductionOps::asum, which records the sum-of-magnitudes reduction |
| `engine/math/reductions/count_nonfinite.cpp` | Defines ReductionOps::count_nonfinite, which records the non-finite count reduction |
| `engine/math/reductions/logsumexp.cpp` | Defines ReductionOps::logsumexp, which records the log-sum-exp reduction |
| `engine/math/reductions/max.cpp` | Defines ReductionOps::max, which records the maximum reduction |
| `engine/math/reductions/max_along.cpp` | Defines ReductionOps::max_along, which records the per-axis maximum reduction |
| `engine/math/reductions/mean.cpp` | Defines ReductionOps::mean, which records the mean reduction |
| `engine/math/reductions/mean_spatial.cpp` | Defines ReductionOps::mean_spatial, which records the spatial mean reduction |
| `engine/math/reductions/min.cpp` | Defines ReductionOps::min, which records the minimum reduction |
| `engine/math/reductions/min_along.cpp` | Defines ReductionOps::min_along, which records the minimum along one axis |
| `engine/math/reductions/nrm2.cpp` | Defines ReductionOps::nrm2, which records the Euclidean norm reduction |
| `engine/math/reductions/prod.cpp` | Defines ReductionOps::prod, which records the product reduction |
| `engine/math/reductions/std.cpp` | Defines ReductionOps::std, which records the standard deviation reduction |
| `engine/math/reductions/sum.cpp` | Defines ReductionOps::sum, which records the sum reduction |
| `engine/math/reductions/sum_along.cpp` | Defines ReductionOps::sum_along, which records the per-axis sum reduction |
| `engine/math/reductions/var.cpp` | Defines ReductionOps::var, which records the variance reduction |

## Notes by file

A source comment in this folder states an invariant in one line and cites this page. The reason
behind the invariant is below, under the path of the file that cites it. A reference to another
section of this page names that section's path; a reference to a public header names the README
under `include/SushiBLAS/` that holds the header's notes.

### `src/engine/blas/level1/axpy.cpp`

`Level1::axpy` declares `y` in both the read set and the write set. `y = alpha * x + y` reads `y`
before it writes it, so `y` is an accumulator and not only a destination.

### `src/engine/blas/level1/copy.cpp`

`Level1::copy` throws when the two dtypes differ. `?copy` is a BLAS routine, and every BLAS
routine is fixed to one precision. The message names `ElementwiseOps::convert`, the converting
copy, because a caller who wanted a conversion reaches this line.

`copy` is the only BLAS operation recorded with `Domain::ALL`. It moves elements and does no
arithmetic on them, so it can serve every element type, the integer ones included. An index tensor
that `iamax` or `argmax` produced can then be moved without passing through a float.

### `src/engine/blas/level1/dot.cpp`

`dot.cpp` defines `Level1::dot`, `dotu` and `dotc`. `dot` is defined for real tensors and throws
`std::invalid_argument` for a complex one, naming `dotu` and `dotc`. `dotu` does not conjugate
and `dotc` conjugates `x`. All three record through one helper in the file.

### `src/engine/blas/level1/iamax.cpp`

`Level1::iamax` requires its result tensor to be INT64, because the result is an index. The
integer dtypes exist so that this check can be made.

Before they existed, `iamax` wrote an `int64_t` into whatever tensor the caller passed. In every
test and every consumer that was a FLOAT32 tensor, so eight bytes went into a four-byte element.
`tests/functional/unit/test_iamax.cpp` worked around it by allocating two floats, and left a
comment reading "Wait, different sizes!". The overrun was survivable while a tensor owned its
allocation. It stopped being survivable when SushiAI began packing tensors into arenas, because
the four bytes past the result then belong to a different tensor. `Tensor::region()` reported a
four-byte write, so the dependency tracker ordered nothing against the corruption either.

### `src/engine/blas/level1/rot.cpp`

`Level1::rot` declares both operands as read and written. Each new `x_i` is built from the old
`x_i` and `y_i`, and each new `y_i` likewise.

### `src/engine/blas/level1/swap.cpp`

`Level1::swap` declares both operands with `OperandSet::updates`: each is read and written, so
each sits in both access sets.

### `src/engine/blas/level2/gemv.cpp`

`Level2::gemv` declares a read of `y` whenever `beta` is non-zero, because `y = alpha * op(A) * x +
beta * y` reads it then. The test is `BlasScalar::is_zero` on the caller's value, not on the value
narrowed to the element type. A `beta` that is non-zero as a double and flushes to zero in the
element type declares an edge the kernel does not need. It can never drop an edge the kernel does
need.

### `src/engine/blas/level2/ger.cpp`

`Level2::ger` declares `A` as read and written, because the rank-1 update reads every element it
writes. The earlier write-only declaration still ordered tasks correctly, since the tracker
inserts a write-after-write edge anyway. Any pass that trusts the declared access set (aliasing
analysis, arena reuse, parallel scheduling) would have reasoned from a false premise.

### `src/engine/blas/level2/syr.cpp`

`Level2::syr` declares `A` as read-modify-write for two reasons. `A` is an accumulator, and the
kernel touches only the selected triangle, so the other triangle has to survive the task
unchanged.

### `src/engine/blas/level2/syr2.cpp`

`Level2::syr2` declares `A` as read-modify-write for the reasons given under
`src/engine/blas/level2/syr.cpp` above: the update accumulates into `A` and touches one triangle.

### `src/engine/blas/level2/trsv.cpp`

`Level2::trsv` dispatches into the TRSM kernels, `kernels/reference/blas/level3/trsm.hpp` and
`kernels/sycl/blas/level3/trsm.hpp`. TRSV is a TRSM with one right-hand-side column (n = 1), so a
second copy of the substitution would add nothing.

`trsv` passes `b` as an n x 1 matrix. A row-major row of that matrix has one element, so
`ldb = 1`; a column-major column needs `ldb >= n`.

`b` is declared with `updates`. Where `RunningSolution` needs a wide copy of the solution, the
operation leases it with `Internal::allocate_blas_scratch` and hands it to the SYCL kernel.

### `src/engine/blas/level3/gemm.cpp`

`Level3::gemm` calls the fused GEMM with nothing fused into it. That avoids a second copy of the
shape derivation, the access set, the scratch lease, the dtype switch and the choice between the
reference and the SYCL kernel.
`GemmIdentityEpilogue` inlines to its argument, so the kernel this file records is byte for byte
the one it recorded when it called the kernel directly; see
`include/SushiBLAS/engine/blas/README.md`, under `execute_gemm`.

`gemm` passes `REAL_AND_COMPLEX` explicitly. The seam defaults to `REAL`, because an epilogue a
caller writes is usually real only. Plain GEMM has supported all five dtypes since SB-2, and the
identity epilogue is well-formed for each of them.

### `src/engine/blas/level3/syrk.cpp`

`Level3::syrk` declares `C` as read-modify-write unconditionally, for two independent reasons. The
beta term reads `C` when beta != 0, and the triangular mask leaves the opposite triangle untouched
even when beta == 0. A byte interval cannot express "one triangle", so the declaration that is
true of the whole region is read and write.

### `src/engine/blas/level3/syr2k.cpp`

`Level3::syr2k` and `her2k` declare `C` with `updates` whatever `beta` is, for the reason given
under `src/engine/blas/level3/syrk.cpp` above: one triangle of `C` is kept.

### `src/engine/logic/all.cpp`

`LogicOps::all` uses `sycl::reduction` on its SYCL path, which a floating-point sum in this library
must not. Under `FORCE_REFERENCE` it runs `all_reference`, a serial loop in the same file. The
SYCL specification leaves the combination order of a reduction unspecified, and that matters only
for a combiner whose result depends on the order. `all` reduces over two values, T(0) and T(1),
under `minimum`, which is associative, commutative and exact on that domain, so every order gives
the same bits. With `+` on floats the order shows in the result and can change with the launch
configuration.

The reduction is created with `initialize_to_identity`. Without that property a SYCL reduction
seeds itself from whatever the result tensor already holds and folds it into the answer. `all`
over a tensor of ones returned 0 whenever the caller reused a result that had held 0, and returned
whatever the allocator left behind when the result had never been written.

`all` is recorded with `Domain::REAL_AND_INTEGRAL`. A truth test means as much on an integer
tensor as on a float one, and an integer element carries the 1 or 0 this family produces without
a conversion. Complex is excluded: `logic_not_zero` would work on it, but `sycl::minimum` and
`sycl::maximum` over `std::complex` do not, and a reduction needs an ordering.

The dtype switch is `Dispatch::with_element_type`, the one the logic dispatch seam uses, so the
arms and the rejection by name are written once.

### `src/engine/logic/any.cpp`

`LogicOps::any` uses `sycl::reduction` for the reason given under `src/engine/logic/all.cpp`
above, with `maximum` in place of `minimum`. The combiner sees only T(0) and T(1), on which
`maximum` is associative, commutative and exact, so the unspecified combination order cannot
change a bit of the answer.

The reduction is created with `initialize_to_identity`, as in `all`. Without it, `any` over a
tensor of zeros returned 1 whenever the caller reused a result that had held 1.

`any` is recorded with `Domain::REAL_AND_INTEGRAL` and switches on the dtype through
`Dispatch::with_element_type`, both for the reasons given under `src/engine/logic/all.cpp`.

### `src/engine/math/elementwise/convert.cpp`

`convert.cpp` defines the copy that converts between dtypes. Three questions are answered in
`include/SushiBLAS/engine/math/README.md`, under `ElementwiseOps::convert`: why the operation
lives in the elementwise family, why the dtype pair is bounded to REAL x REAL, and why a
conversion to the same dtype is legal. This section records the mechanics.

`SushiBLAS::Dispatch` has no public `execute_convert` seam. Every other executor in `Dispatch` is
public because it takes a functor the caller writes. That lets a consumer fuse several math
operations into one kernel without writing SYCL, and it is why those headers were promoted out of
`src/`. `convert` has no functor: the conversion is `static_cast`, fixed by the pair of dtypes, so
a seam would expose a parameter that does not exist. Keeping the implementation in this
translation unit also confines the nine instantiations to one object file, which is the bound the
header's notes argue for.

`record_convert` is one of the nine arms. It is an ordinary function template and not a lambda,
because the dispatch in `dispatch_destination` and `ElementwiseOps::convert` has to be two plain
`switch` statements. A generic lambda nested inside a generic lambda crashes this project's clang
during instantiation (see `include/SushiBLAS/engine/README.md`, under `execute_typed_task`), and
an operation over two types is where that nesting would otherwise be written.

`record_convert` launches through `Launch::indexed` on the SYCL path, and runs the same functor
through `Serial::indexed` under `FORCE_REFERENCE`. `Launch::map` has a `static_assert` that
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
is how a silent no-op gets in; `include/SushiBLAS/engine/blas/README.md`, under
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

The access sets hold `Tensor::region()`, never `data()`. A `ResourceRegion` converts implicitly
from a bare pointer, and a bare pointer widens to the whole allocation. Two views of one arena
would become two unrelated resources, `overlaps` would answer false, and the tracker would emit no
edge. `include/SushiBLAS/README.md`, under `Tensor::region`, describes the region, and
`docs/api/mainpage.md` records that this defect has cost the stack two training runs.

The `switch` in `ElementwiseOps::convert` is the outer half of the dispatch and picks the source
type. Together with `dispatch_destination` it makes two plain switches over function templates,
nine instantiations, and no lambda nested in a lambda.

### `src/engine/math/random/constant.cpp`

`RandomOps::constant` has integer arms because integer tensors need known values: an index tensor
gets one before an operation writes over it, and a comparison operand gets a threshold. The
double parameter is truncated to the integer type, by the same convention the HALF and FLOAT32
arms follow: the caller states a value, and the tensor's dtype decides how it is stored.

### `src/engine/math/random/normal.cpp`

`RandomOps::normal` draws the real and imaginary halves of a complex tensor as one flat run of 2n
reals. The position of a component in that run fixes the Philox block and word it is drawn from,
which keeps a complex fill reproducible element for element. Both Box-Muller outputs of a uniform
pair are used: a block yields four FLOAT32 or two FLOAT64 normals.

### `src/engine/math/random/orthogonal.cpp`

`RandomOps::orthogonal` first fills the tensor with N(0,1). The Gaussian ensemble is orthogonally
invariant, so orthonormalising a Gaussian matrix yields a uniformly distributed orthogonal matrix
and not a biased one. The fill is recorded as its own task, and the graph orders the sweep after
it because both declare the same region.

`orthogonal` orthonormalises along the shorter axis: the rows of a wide matrix, the columns of a
tall one. At most min(rows, cols) vectors can be mutually orthogonal. The result satisfies
Q Q^T = gain^2 I in the first case and Q^T Q = gain^2 I in the second, the same convention as
PyTorch's `orthogonal_`.

The sweep itself is `Kernels::SYCL::Random::orthonormalize`, one work-group per vector, or
`Kernels::Reference::Random::orthonormalize` under `FORCE_REFERENCE`; see `kernels/README.md`.
`orthogonal` calls `TaskRecorder::require_host_reachable` before it records the fill, so a refused
call records nothing and draws nothing. It takes its steps from `t.strides`, not from arithmetic on
rows and columns, so it is correct for `ROW_MAJOR` and `COLUMN_MAJOR` with no special case. A
column-major matrix walks the two axes in the other order, and the transpose of an orthogonal matrix
is orthogonal.

### `src/engine/math/random/poisson.cpp`

`RandomOps::poisson` checks lambda against [0, `K_MAX_POISSON_LAMBDA`] = [0, 1e7] at record
time, computes the per-lambda constants once on the host with
`Kernels::SYCL::Random::make_poisson_params`, and fills one element per work-item with
`draw_poisson`. Below lambda 10 the draw is Knuth's multiplication loop; from 10 up it is
Hoermann's PTRS. The math module README explains the bound and `kernels/README.md` the two
methods. It throws `UnsupportedExecution` when `t` has elements and its device lacks the fp64
aspect.

### `src/engine/math/random/shuffle.cpp`

`RandomOps::shuffle` runs Fisher-Yates sequentially. Each step draws its swap partner from a range
that depends on the step index, so the permutation is uniform only if the steps happen in order. A
`single_task` keeps that order on the device and leaves the tensor's data where it is; under
`FORCE_REFERENCE` the same `FisherYates` functor runs on the host. The
parallel alternative tags every element with a random key and sorts, which needs a second buffer
of the full size and a device sort this library does not have.

The swap is instantiated per dtype and does not swap raw bytes, so a `std::complex<double>` moves
as one value and cannot tear into two halves.

`shuffle` declares its tensor as read and written, unlike the distributions. A shuffle permutes
what is already there, so the task depends on whatever produced the contents and not only on
earlier writers.

### `src/engine/math/random/uniform.cpp`

`RandomOps::uniform` fills a complex tensor as `src/engine/math/random/normal.cpp` above
describes: one flat run of 2n reals, read four FLOAT32 or two FLOAT64 values per Philox block.

### `src/engine/math/reductions/argmax.cpp`

`ReductionOps::argmax` folds (value, index) pairs through `Dispatch::ArgReductionDispatch` over
`Reduce::RunningArgMax`, in the fixed-order shape of `Reduce::reduce_indexed`. The accumulator's
order is total: the larger value wins, then the lower index, and its NaN rule is stated on
`RunningArgMax::combine`. Because the order is total the combination order cannot change the
answer, so the lowest-index maximum wins a tie on every device and launch shape. Under
`FORCE_REFERENCE` the same accumulator folds in index order on the host.

It does not go through `sycl::reduction`: custom reductions behave differently on the OpenCL and
Level-Zero backends, and the tie-breaking rule must not change with the backend.

The result goes into an INT64 tensor through `execute_index_reduction`, not into a tensor of the
input's dtype through `execute_reduction`. It was once stored as `static_cast<T>(idx)`, a float. A
float is exact below 2^24. Above that, two adjacent indices round to the same value, so the
operation returned a wrong index on the tensor sizes a training run calls it on. In HALF the
ceiling is 2048.

### `src/engine/math/reductions/argmin.cpp`

`ReductionOps::argmin` is `argmax` with `Reduce::RunningArgMin`: the lowest-index minimum always
wins a tie, and the result does not depend on the combination order, for the reasons given under
`src/engine/math/reductions/argmax.cpp` above.

The result goes into an INT64 tensor through `execute_index_reduction`. The same section explains
why a result stored in the input's dtype was wrong above 2^24, and above 2048 in HALF.

### `src/engine/math/reductions/count_nonfinite.cpp`

`include/SushiBLAS/engine/math/README.md`, under `ReductionOps::count_nonfinite`, answers two
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

### `src/engine/math/reductions/max.cpp`

`ReductionOps::max` propagates NaN, as NumPy's `max` does: if any element is NaN, the result is
NaN, whatever the element's position and whatever launch shape the device gets. It folds through
`Dispatch::execute_folding_reduction` with `Reduce::NaNPropagatingMax` from
`include/SushiBLAS/support/reduce_1d.hpp`. A plain `<` comparison is false against a NaN, so
under it a NaN would survive only when it arrives first. `NaNPropagatingMax` returns NaN from
`combine` when either side is NaN and stops changing once `accumulate` has seen one, so every fold
order gives NaN.

An empty tensor throws `std::invalid_argument` when the call is recorded.

### `src/engine/math/reductions/max_along.cpp`

`ReductionOps::max_along` goes through the folding seam with `Reduce::NaNPropagatingMax`, the
accumulator the whole-tensor `max` uses, so a NaN in a segment makes that segment's result NaN
and a per-axis maximum agrees with a whole-tensor one. The axis family has one shape that serves
every accumulator, because `sycl::reduction` has no segmented form. `min_along.cpp` is the same
call with `Reduce::NaNPropagatingMin`. See `include/SushiBLAS/support/README.md`, under
`NaNPropagatingMax`.

### `src/engine/math/reductions/mean.cpp`

`ReductionOps::mean` divides in the finalizer, so it costs the same number of kernel launches as
`sum`. An earlier version submitted a second `single_task` only to divide one scalar by one
integer.

### `src/engine/math/reductions/mean_spatial.cpp`

`ReductionOps::mean_spatial` is built from the same three pieces as `mean`: a compensated sum,
no transform, and `v / n` as the finalizer. It runs them over the segmented seam, where `mean`
uses the whole-tensor one. The new part is the span. A fold is identified by a pair `(n, c)`,
which `spatial_span_for` builds and the second segment axis of `AxisSpan` carries.

### `src/engine/math/reductions/min.cpp`

`ReductionOps::min` mirrors `max` above, with `Reduce::NaNPropagatingMin`: any NaN element makes
the result NaN in every fold order, and an empty tensor throws `std::invalid_argument`.

### `src/engine/math/reductions/nrm2.cpp`

`ReductionOps::nrm2` computes the scaling 2-norm that BLAS's xNRM2 specifies. An earlier version
computed `sqrt(sum(x*x))`. The call has no transform and no finalizer, because the squaring and
the square root are safe only inside a running scale. `include/SushiBLAS/support/README.md`, under
`ScaledSumOfSquares`, describes the algorithm and says why it belongs in the accumulator. The
launch count did not change: one launch on the single-pass shape and two on the two-stage one.

### `src/engine/math/reductions/prod.cpp`

`ReductionOps::prod` accumulates with `RunningProduct`, not `CompensatedSum`. Neumaier's
correction recovers the bits an addition rounds off and does nothing for a multiplication. What
`prod` shares with the additive reductions is the fixed combination order, which is why it left
`sycl::reduction` as they did: floating-point multiplication is no more associative than addition.

### `src/engine/math/reductions/sum.cpp`

`ReductionOps::sum` depends on the combination order, so it goes through the folding seam and not
`sycl::reduction`. `include/SushiBLAS/support/README.md`, under `reduce_1d.hpp`, says why the
unspecified order is a correctness problem for `+` and not for `max`. `sum` has no transform and
no finalizer. The other reductions on the folding seam are this call with a transform, a
finalizer or a different accumulator added.

### `src/engine/math/reductions/sum_along.cpp`

`ReductionOps::sum_along` depends on the combination order, so it takes the same
Neumaier-compensated fixed-order fold as the whole-tensor `sum`, in its segmented shape. It has no
transform and no finalizer. A per-axis `mean` is this call with `v / n` as the finalizer and costs
no extra launch.

### `src/engine/spatial/broadcast.cpp`

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

### `src/engine/spatial/channels.cpp`

`channel_copy_span` builds its span by hand and does not call `Spatial::window_span_for`, which
validates window origins against a `Spatial::ConvGeometry` that a plain copy does not have. For
the same reason it sets `out_h` and `out_w` to the input extents: the operation visits every pixel
of one shared grid, so there is no second extent derived from a window.

`span.channels` stays at the source's channel count and neither kernel reads it. The width of the
copy travels as an explicit argument.

The channel copies are recorded with `Domain::ALL`. They move elements and do no arithmetic on
them, so every element type the library has can be copied.

### `src/engine/spatial/col2im.cpp`

`SpatialOps::col2im` calls the extents overload of `window_span_for`. `image` takes the input role
in the span: it is the rank-4 operand whose H, W and strides the window walks. The other operand
is a rank-2 column matrix with no extents to read.

### `src/engine/spatial/im2col.cpp`

`SpatialOps::im2col` calls the extents overload of `window_span_for`, not the two-tensor one. The
other operand is a rank-2 column matrix, so no output image exists whose shape could supply H_o
and W_o.

The overload of `im2col` that derives the extents assumes symmetric padding, and can assume
nothing else. `ConvGeometry` carries `pad_left` and no `pad_right`, so W_o can be recovered only
by taking the leading pad for both sides. H_o then follows from the row count. A caller whose
padding is asymmetric must call the overload with explicit extents, which exists for that case.
The 2x2 stride-1 pool of YOLOv3-tiny is the one in scope.

### `src/engine/spatial/pool_avg.cpp`

In `SpatialOps::pool_avg_backward`, `dx` takes the input role and `dy` the output role, the
assignment the forward made. The geometry must therefore be the forward's.

### `src/engine/spatial/pool_max.cpp`

`SpatialOps::pool_max` requires its index buffer to be INT64 and checks that by name; the kernel
never reinterprets another dtype. fp32 represents integers exactly only below 2^24, and a
224x224x64 feature map passes that at batch 6.

The dtype that drives the switch is the value tensor's. The index pointer is captured as
`int64_t*` outside the switch, because the operation fixes its type and the operands do not.

### `src/engine/spatial/pool_max_backward.cpp`

In `SpatialOps::pool_max_backward`, `dx` takes the input role in the span, since it has the
image's H, W and strides, and `dy` takes the output role. That is the assignment the forward made,
so the geometry must be the forward's exactly.

### `src/engine/spatial/window_dot.cpp`, `window_dot_weights_backward.cpp`

`SpatialOps::window_dot` builds its span with the two-operand `window_span_for` and its taps with
`tap_span_for`, and refuses an output that shares a byte with an input, because the kernel is a
gather that reads while it writes.

In `window_dot_backward`, `dx` takes the input role in the span and `dy` the output role, the
assignment the forward made, so the geometry must be the forward's. In
`window_dot_weights_backward`, `in` takes the input role and `dy` the output role for the same
reason, and `dweights` is described by `tap_span_for` as `weights` is in the forward.

All three are recorded with `Domain::REAL`.

### `src/engine/spatial/upsample_nearest.cpp`

`upsample_span_for` does not call `Spatial::window_span_for`. That function needs a
`Spatial::ConvGeometry`, and the only one that fits here is the default 1x1 window at stride 1.
Under that geometry the window-origin check reads "output row `out_h - 1` starts at input row
`out_h - 1`" and rejects every output larger than the input, which is every upsampling.
`upsample_span_for` reads the strides off the tensors and checks the rule this operation has: each
output extent is an exact `scale` multiple of the input's. `span.geometry` stays default, and no
upsample kernel reads it.

In `SpatialOps::upsample_nearest_backward`, `dx` takes the input role and `dy` the output role,
the assignment the forward made. The scales must therefore be the forward's.

### `src/io.cpp`

`IO::save_npy` writes a `descr` for every dtype it knows and throws for any other. The chain once
fell through for HALF and wrote a header with an empty `descr` and a payload numpy could not read:
a corrupt file with no error at either end.

The text rendering decodes every element to double before printing it. INT32 survives that
exactly; INT64 loses precision past 2^53. That is acceptable for a dump a person reads
and nowhere else. `IO::save_npy` writes the raw bytes, so no saved value passes through the text
rendering.

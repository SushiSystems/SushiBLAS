# kernels

The kernels that are private to the library. An operation has a plain C++ kernel under
`reference/`, which is the mathematical ground truth, and a SYCL kernel under `sycl/`, which runs
on the device. The random draws exist only under `sycl/`; the orthonormalisation sweep has both.
GEMM's pair, the packed product and the access bricks the BLAS kernels read their operands through
are not here: they are under `include/SushiBLAS/kernels/`, for the reason given under
`kernels/blas.hpp` below.

| Path | Holds |
| --- | --- |
| `blas.hpp`, `sparse.hpp`, `spatial.hpp`, `random.hpp` | The include of every kernel of one family |
| `math.hpp` | No include |
| `reference/blas/`, `sycl/blas/` | Levels 1 to 3. `level1.hpp` holds the vector routines of one side and the three rotation builders have a file each; levels 2 and 3 have one file per routine, and one per shared loop |
| `sparse/` | `CsrPattern` and `IndexList`, how a kernel of either side reads a sparse operand |
| `reference/sparse/`, `sycl/sparse/` | The sparse-dense level 1 routines, the CSR products and the triangular solve |
| `reference/spatial/`, `sycl/spatial/` | im2col, col2im, pooling, the window product, upsampling, channel and broadcast copies |
| `reference/random/` | The reference orthonormalisation sweep |
| `sycl/random/` | Philox4x32-10, the Poisson and truncated normal draws, and the Gram-Schmidt sweep |

## Files

The include points:

| File | Holds |
| --- | --- |
| `blas.hpp` | Includes every BLAS kernel, reference and SYCL |
| `math.hpp` | Reserves the include point of the math kernels and includes nothing |
| `random.hpp` | Includes the random kernels: the SYCL ones and the reference orthonormalisation |
| `sparse.hpp` | Includes every sparse kernel, reference and SYCL |
| `spatial.hpp` | Includes every spatial kernel, reference and SYCL |

Dense BLAS, reference:

| File | Holds |
| --- | --- |
| `reference/blas/level1/level1.hpp` | Defines the reference kernels of the BLAS level 1 routines |
| `reference/blas/level1/rotg.hpp` | Defines the reference kernel of ROTG |
| `reference/blas/level1/rotm.hpp` | Defines the reference kernel of ROTM |
| `reference/blas/level1/rotmg.hpp` | Defines the reference kernel of ROTMG |
| `reference/blas/level2/gbmv.hpp` | Defines the reference kernel of GBMV |
| `reference/blas/level2/gemv.hpp` | Defines the reference kernel of GEMV |
| `reference/blas/level2/ger.hpp` | Defines the reference kernel of GER, GERU and GERC |
| `reference/blas/level2/sbmv.hpp` | Defines the reference kernel of SBMV and HBMV |
| `reference/blas/level2/spmv.hpp` | Defines the reference kernel of SPMV and HPMV |
| `reference/blas/level2/spr.hpp` | Defines the reference kernel of SPR and HPR |
| `reference/blas/level2/spr2.hpp` | Defines the reference kernel of SPR2 and HPR2 |
| `reference/blas/level2/symmetric_product.hpp` | Defines the reference symmetric matrix-vector product SBMV and SPMV share |
| `reference/blas/level2/symv.hpp` | Defines the reference kernel of SYMV and HEMV |
| `reference/blas/level2/syr.hpp` | Defines the reference kernel of SYR and HER |
| `reference/blas/level2/syr2.hpp` | Defines the reference kernel of SYR2 and HER2 |
| `reference/blas/level2/tbmv.hpp` | Defines the reference kernel of TBMV |
| `reference/blas/level2/tbsv.hpp` | Defines the reference kernel of TBSV |
| `reference/blas/level2/tpmv.hpp` | Defines the reference kernel of TPMV |
| `reference/blas/level2/tpsv.hpp` | Defines the reference kernel of TPSV |
| `reference/blas/level2/triangular_product.hpp` | Defines the reference triangular matrix-vector product TBMV and TPMV share |
| `reference/blas/level2/triangular_solve.hpp` | Defines the reference triangular substitution TBSV and TPSV share |
| `reference/blas/level2/trmv.hpp` | Defines the reference kernel of TRMV |
| `reference/blas/level3/syr2k.hpp` | Defines the reference kernel of SYR2K and, under a conjugation policy, of HER2K |
| `reference/blas/level3/syrk.hpp` | Defines the reference kernels of SYRK, single and batched |
| `reference/blas/level3/trsm.hpp` | Defines the reference kernels of TRSM, single and batched |

Dense BLAS, SYCL:

| File | Holds |
| --- | --- |
| `sycl/blas/level1/level1.hpp` | Defines the SYCL kernels of the BLAS level 1 routines |
| `sycl/blas/level1/rotg.hpp` | Defines the SYCL kernel of ROTG |
| `sycl/blas/level1/rotm.hpp` | Defines the SYCL kernel of ROTM |
| `sycl/blas/level1/rotmg.hpp` | Defines the SYCL kernel of ROTMG |
| `sycl/blas/level2/gbmv.hpp` | Defines the SYCL kernel of GBMV |
| `sycl/blas/level2/gemv.hpp` | Defines the SYCL kernel of GEMV |
| `sycl/blas/level2/ger.hpp` | Defines the SYCL kernel of GER, GERU and GERC |
| `sycl/blas/level2/row_reduction.hpp` | Defines the work-group-per-row reduction GEMV, SYMV and TRMV share |
| `sycl/blas/level2/sbmv.hpp` | Defines the SYCL kernel of SBMV and HBMV |
| `sycl/blas/level2/spmv.hpp` | Defines the SYCL kernel of SPMV and HPMV |
| `sycl/blas/level2/spr.hpp` | Defines the SYCL kernel of SPR and HPR |
| `sycl/blas/level2/spr2.hpp` | Defines the SYCL kernel of SPR2 and HPR2 |
| `sycl/blas/level2/symmetric_product.hpp` | Defines the SYCL symmetric matrix-vector product SBMV and SPMV share |
| `sycl/blas/level2/symv.hpp` | Defines the SYCL kernel of SYMV and HEMV |
| `sycl/blas/level2/syr.hpp` | Defines the SYCL kernel of SYR and HER |
| `sycl/blas/level2/syr2.hpp` | Defines the SYCL kernel of SYR2 and HER2 |
| `sycl/blas/level2/tbmv.hpp` | Defines the SYCL kernel of TBMV |
| `sycl/blas/level2/tbsv.hpp` | Defines the SYCL kernel of TBSV |
| `sycl/blas/level2/tpmv.hpp` | Defines the SYCL kernel of TPMV |
| `sycl/blas/level2/tpsv.hpp` | Defines the SYCL kernel of TPSV |
| `sycl/blas/level2/triangular_product.hpp` | Defines the SYCL triangular matrix-vector product TBMV and TPMV share |
| `sycl/blas/level2/vector_snapshot.hpp` | Defines the copy of x the in-place triangular products read |
| `sycl/blas/level2/triangular_solve.hpp` | Defines the SYCL triangular substitution TBSV and TPSV share |
| `sycl/blas/level2/trmv.hpp` | Defines the SYCL kernel of TRMV: a snapshot of x, then one row product per element |
| `sycl/blas/level3/syr2k.hpp` | Defines the SYCL kernel of SYR2K and, under a conjugation policy, of HER2K |
| `sycl/blas/level3/syrk.hpp` | Defines the SYCL kernels of SYRK and HERK, their tile geometry and their packed store |
| `sycl/blas/level3/triangular_plan.hpp` | Declares TriangularSolvePlan, the diagonal block size of a triangular solve on a device |
| `sycl/blas/level3/triangular_system.hpp` | Declares TriangularSystem, the one substitution every side and flag of TRSM decodes to |
| `sycl/blas/level3/trsm.hpp` | Defines the SYCL kernel of TRSM: a blocked substitution with parallel updates |

Sparse:

| File | Holds |
| --- | --- |
| `reference/sparse/level1.hpp` | Defines the reference kernels of the sparse-dense level 1 routines |
| `reference/sparse/spmm.hpp` | Defines the reference kernel of the product of a sparse matrix and a dense matrix |
| `reference/sparse/spmv.hpp` | Defines the reference kernel of the sparse matrix-vector product |
| `reference/sparse/spsymv.hpp` | Defines the reference kernel of the symmetric and Hermitian sparse matrix-vector product |
| `reference/sparse/sptrsv.hpp` | Defines the reference kernel of the sparse triangular solve |
| `reference/sparse/structure.hpp` | Defines the host checks of the structure a sparse kernel takes on trust |
| `sparse/csr_pattern.hpp` | Declares CsrPattern, how a kernel reads the rows and columns of a CSR matrix |
| `sparse/index_list.hpp` | Declares IndexList, how a kernel reads the stored positions of a sparse vector |
| `sycl/sparse/level1.hpp` | Defines the SYCL kernels of the sparse-dense level 1 routines |
| `sycl/sparse/row_blocks.hpp` | Defines the row-block scatter that the transposed and the symmetric product share |
| `sycl/sparse/spmm.hpp` | Defines the SYCL kernel of the product of a sparse matrix and a dense matrix |
| `sycl/sparse/spmv.hpp` | Defines the SYCL kernel of the sparse matrix-vector product |
| `sycl/sparse/spsymv.hpp` | Defines the SYCL kernel of the symmetric and Hermitian sparse matrix-vector product |
| `sycl/sparse/sptrsv.hpp` | Defines the SYCL kernel of the sparse triangular solve |

Spatial:

| File | Holds |
| --- | --- |
| `reference/spatial/broadcast.hpp` | Defines the reference kernels of the three broadcast copies |
| `reference/spatial/channels.hpp` | Defines the reference kernel that copies a run of channels |
| `reference/spatial/col2im.hpp` | Defines the reference kernel of col2im |
| `reference/spatial/im2col.hpp` | Defines the reference kernel of im2col |
| `reference/spatial/pool_avg.hpp` | Defines the reference kernels of average pooling and its gradient |
| `reference/spatial/pool_max.hpp` | Defines the reference kernels of max pooling and its gradient |
| `reference/spatial/upsample.hpp` | Defines the reference kernels of nearest-neighbour upsampling and its gradient |
| `reference/spatial/window_dot.hpp` | Defines the reference kernels of the per-channel weighted window sum and its input gradient |
| `reference/spatial/window_dot_weights.hpp` | Defines the reference kernel of the weight gradient of the per-channel weighted window sum |
| `sycl/spatial/broadcast.hpp` | Defines the SYCL kernels of the three broadcast copies |
| `sycl/spatial/channels.hpp` | Defines the SYCL kernel that copies a run of channels |
| `sycl/spatial/col2im.hpp` | Defines the SYCL kernel of col2im |
| `sycl/spatial/im2col.hpp` | Defines the SYCL kernel of im2col |
| `sycl/spatial/pool_avg.hpp` | Defines the SYCL kernels of average pooling and its gradient |
| `sycl/spatial/pool_max.hpp` | Defines the SYCL kernels of max pooling and its gradient |
| `sycl/spatial/upsample.hpp` | Defines the SYCL kernels of nearest-neighbour upsampling and its gradient |
| `sycl/spatial/window_dot.hpp` | Defines the SYCL kernels of the per-channel weighted window sum and its input gradient |
| `sycl/spatial/window_dot_weights.hpp` | Defines the SYCL kernel of the weight gradient of the per-channel weighted window sum |

Random:

| File | Holds |
| --- | --- |
| `reference/random/orthonormalize.hpp` | Defines the reference kernel of the orthonormalisation sweep |
| `sycl/random/orthonormalize.hpp` | Defines the work-group parallel Gram-Schmidt sweep behind RandomOps::orthogonal |
| `sycl/random/philox.hpp` | Defines Philox4x32-10 and the uniform, normal and index draws built on it |
| `sycl/random/poisson.hpp` | Defines the Poisson draw: Knuth's product below lambda 10, Hoermann's PTRS above |
| `sycl/random/truncated_normal.hpp` | Defines the truncated standard normal draw by bisection of the distribution function |

`math.hpp` includes nothing. Until 2026-10-05 it carried, commented out and marked as planned, the
includes of `kernels/reference/math/reductions/sum.hpp` and `kernels/sycl/math/reductions/sum.hpp`;
neither file exists.

## Notes by file

A source comment in this folder states an invariant in one line, or a `@see` in a symbol block,
and cites this page. The reason behind it is below, under the path of the file that cites it.
A reference to another section of this page names that section's path; a reference to a public
header names the README under `include/SushiBLAS/` that holds the header's notes.

### `kernels/blas.hpp`

`blas.hpp` includes most kernels by a quoted path relative to the repository root, because they
are private to this tree. It includes GEMM's two kernels as `<SushiBLAS/...>`. Those two live
under `include/` and are installed, because `Dispatch::execute_gemm` in
`<SushiBLAS/engine/blas/dispatch.hpp>` instantiates them in a consumer's translation unit with the
consumer's own epilogue functor. A template that a public seam instantiates cannot stay private.
The rule is: a kernel that a public seam names lives under `include/SushiBLAS/kernels/`, split
into `reference/` and `sycl/` the same way, and every other kernel stays here.

### `kernels/sparse.hpp`

`sparse.hpp` includes every sparse kernel by a quoted path relative to the repository root. They
are private to this tree for the reason given under `kernels/blas.hpp` above: `SparseOps` is
compiled into the library, and no public seam instantiates a sparse kernel.

### `kernels/reference/blas/level1/level1.hpp`

The reference level 1 kernels are unblocked, single-threaded loops. Level 1 is not the tier this
library is tuned for (level 3 is, in `gemm.hpp` and `syrk.hpp`), so a serial loop that is easy to
check is the right trade.

`nrm2` computes the Euclidean norm in the scaling form xNRM2 specifies. An earlier version computed
`sqrt(sum(x*x))`, and that one line held two bugs.

The squaring overflowed and underflowed. An fp32 element past about 1.8e19 squares to infinity, so
the norm of a vector whose largest entry is 1e20 came back `inf` where the true answer is about
1e20. An element below about 1e-22 squares to zero, so a vector of denormals came back exactly 0.
`nrm2` now carries a running `scale` and sums `(x / scale)^2`, which squares only values near 1.
The result overflows only when the true norm does.

`abs1` was the wrong magnitude for a complex element. It is BLAS's `|Re| + |Im|`, which xASUM and
IxAMAX are defined on, and the 2-norm needs `Re^2 + Im^2`. Using `abs1` added `2|Re||Im|` per
element, so a complex `nrm2` was `sqrt(2)` times too large for x = 1+i. `nrm2` folds the two
components separately, which gives the correct magnitude, stays safe from overflow, and is what
LAPACK's `classq` does.

The reference `nrm2` uses the running maximum as its scale and applies no compensation. The SYCL
kernel rounds its scale to an octave and applies Neumaier's correction, and the two share no code.
A ground truth that shares an implementation with the kernel it checks does not check it.

`nrm2` folds an infinity or a NaN in unscaled, because such an element has no magnitude to be
relative to. That saturates the sum, which is the answer BLAS gives.

Every sum in this file is formed in `Kernels::ReferenceSum`: in `double` for HALF, FLOAT32 and
COMPLEX32, and in a compensated `double` sum for FLOAT64 and COMPLEX64. The result is rounded to
the element type once, on the store. `include/SushiBLAS/support/README.md`, under
`reference_accumulator.hpp`, says why a reference sums wider than the kernel it judges. The other
reference kernels that sum (`gemv`, `symv`, `trmv`, `gbmv`, the shared symmetric and triangular
loops, `syrk`, `syr2k`, `trsm`, the sparse kernels, the orthonormalisation sweep and
`window_dot_weights`) do the same.

### `kernels/reference/blas/level1/rotg.hpp`, `rotm.hpp`, `rotmg.hpp`

The three rotation builders have one file each on each side. The arithmetic is not in them: it is
`givens_rotation` and `modified_givens` under `include/SushiBLAS/kernels/blas/`, shared with the
SYCL kernels. The reference `rotg` passes a host square root, `Detail::HostRoot`, where the SYCL
kernel passes `Detail::DeviceRoot`. `rotm` applies the transformation H that a five-element
parameter array describes to every pair `(x_i, y_i)`; the SYCL kernel runs one work-item per pair.

### `kernels/reference/blas/level2/symmetric_product.hpp`, `triangular_product.hpp`, `triangular_solve.hpp`

Three loops written once over an access brick and shared by the band and the packed routines.
`symmetric_product` is the matrix-vector product of a symmetric or Hermitian matrix of which one
triangle is stored: `sbmv` and `hbmv` call it with a `BandAccess`, `spmv` and `hpmv` with a
`PackedAccess`. `triangular_product` is `x = op(A) * x` in place, shared by `tbmv` and `tpmv`, and
`triangular_solve` is the substitution shared by `tbsv` and `tpsv`. A brick answers `holds(r, c)`
and bounds the stored part of a row, so each loop visits only what the storage holds. The dense
routines `symv`, `trmv` and `trsv` keep their own kernels.

`gbmv`, `spr` and `spr2` have a kernel each: `gbmv` reads a general band through `BandAccess`,
and `spr`, `hpr`, `spr2` and `hpr2` write a packed triangle through `PackedAccess`. The
conjugation policy, `Unconjugated` or `Conjugated`, is the only difference between a symmetric
routine and its Hermitian sibling in every one of these files.

### `kernels/reference/blas/level2/trmv.hpp`

`trmv` runs in place without a temporary copy of `x`. The effective triangle of op(A) sets the
direction of the walk: upper if the support after the transpose is k >= i, lower otherwise. The
walk visits `i` in the order that reads each `x[i]` before the kernel overwrites it and never
after. The section on `kernels/sycl/blas/level2/trmv.hpp` below shows why both directions are
safe.

### `kernels/reference/blas/level3/syrk.hpp`

`syrk` addresses its operand by the same two-stride rule as `Reference::gemm`. It once carried the
same defect: the `ColumnMajor` arm was inverted, which made `layout` a no-op for A and read past
its end for some shapes. The identity that catches the defect is in
`include/SushiBLAS/kernels/README.md`, under `reference/blas/level3/gemm.hpp`.

### `kernels/reference/blas/level3/syr2k.hpp`

`syr2k` computes `C = alpha * op(A) * Conj(op(B))^T + Conj(alpha) * op(B) * Conj(op(A))^T +
beta * C` on one triangle of C. Under `Unconjugated` that is SYR2K and under `Conjugated` HER2K,
whose `beta` is real and whose diagonal is stored real. A zero `beta` does not read the prior C.

### `kernels/reference/blas/level3/trsm.hpp`

`trsm` also backs `Level2::trsv`, which is a TRSM with one right-hand-side column (n = 1); see
`src/engine/blas/level2/trsv.cpp`.

The left side is solved one column of B at a time, by forward substitution when the effective
triangle of A is lower and by backward substitution when it is upper. The section on
`kernels/sycl/blas/level3/trsm.hpp` below derives the effective triangle. The right side is the
same system transposed once more, row by row: X\*op(A) = alpha\*B is equivalent to
op(A)^T \* X^T = alpha\*B^T.

The solved elements are carried between steps in a `RunningSolution`, so a HALF solve does not
round each element to HALF before a later row reads it. Where that needs a wide copy, the
reference allocates it on the host, indexed like B.

### `kernels/reference/random/orthonormalize.hpp`

The reference of the sweep behind `RandomOps::orthogonal`: modified Gram-Schmidt in place over a
`VectorSet`, vector `i` projected out of every later vector in ascending order,
`K_ORTHONORMAL_SWEEPS` times, then each vector scaled to norm `gain`. The layout, the sweep count
and `tiny_norm_sq` come from `<SushiBLAS/support/orthonormal_sweep.hpp>`, which the SYCL kernel
reads too.

### `kernels/sparse/csr_pattern.hpp`, `index_list.hpp`

`CsrPattern` holds the shape, the index base and the two index arrays of a CSR matrix, and
`row_begin`, `row_end` and `column` return zero-based values whatever the base and whether the
indices are INT32 or INT64. `IndexList` does the same for the stored positions of a sparse
vector. Both are in neither `reference/` nor `sycl/` because both sides read through them.

### `kernels/reference/sparse/structure.hpp`

`find_defect` walks the index arrays of a CSR matrix or a sparse vector and names the first rule
they break as a `StructureDefect`: a first offset that is not the index base, a row that ends
before it starts, a last offset that does not name the entry count, an index outside the operand,
a position stored twice. A sparse kernel takes its structure on trust, because the record-time
checks in `include/SushiBLAS/engine/sparse/` cannot read the arrays. The reference kernels call
`find_defect` inside an `SB_ASSERT`, which aborts in a build without `NDEBUG` and is empty with
it.

### `kernels/reference/sparse/level1.hpp`, `spmv.hpp`, `spmm.hpp`, `spsymv.hpp`, `sptrsv.hpp`

Plain loops in stored order. `spmv` and `spmm` go row by row; `spsymv` reads the one stored
triangle and applies each off-diagonal entry on both sides, conjugated under `Conjugated` for
`sphemv`; `sptrsv` solves a lower triangle from row 0 down and an upper one upwards, carrying the
solution in a `RunningSolution`. The dot products sum in a `ReferenceSum`.

### `kernels/reference/spatial/broadcast.hpp`

No kernel in `broadcast.hpp` accumulates. Each one writes one output element per iteration from
one input element and at most one per-channel scalar. There is no running total, no reduction,
and no operand whose combination order could differ between two runs or between the reference and
the device. The other spatial kernels do accumulate, and they are written as gathers, never as
atomic scatters, to keep that order fixed. Here there is no order to fix, so the reference and the
SYCL kernel agree bit for bit, and the test that compares them checks indexing arithmetic.

`channel_affine` computes `out[n,h,w,c] = in[n,h,w,c] * gamma[c] + beta[c]`. That is the apply
half of BatchNorm, and all of BatchNorm in evaluation mode. `gamma` and `beta` are rank-1 `[C]`
operands addressed through their own strides, so a parameter vector that is a view into a packed
parameter arena works without a compacting copy.

`broadcast_spatial` computes `out[n,h,w,c] = in[n,c] * scale`. With `scale = 1/(H*W)` it is the
backward of global average pooling. Its input is rank-2, so the input's two strides are parameters
and not fields of `span`. A `Spatial::WindowSpan` has four input strides, and only two of them
would mean anything here; a reader would have to rule out the other two.

`layout_convert` is the boundary transpose between NCHW and NHWC. It has no direction flag. Both
sides are indexed through their own four strides, so the caller chooses the direction by handing
in an output with different strides, and one kernel serves both directions.

### `kernels/reference/spatial/channels.hpp`

`copy_channel_range` is all of channel concatenation and channel slicing (design decision D-A6).
Concatenation is this copy with the source's offset at zero, and slicing is the same copy with the
destination's offset at zero. Once the offsets are named nothing distinguishes the two directions,
so there is one kernel and two public names.

`span` carries the shared `[N, H, W]` grid and both operands' strides: `in_*` describes the
source and `out_*` the destination. The kernel does not read `span.channels`. The width of the
copy is `count`, which differs from either tensor's channel count and has to be passed.
`span.out_h` and `span.out_w` hold the input extents, because a copy has no window and so no
output extent of its own.

### `kernels/reference/spatial/col2im.hpp`

`col2im` is the transpose of `im2col` and its gradient. It is written as a gather: the loop nest
runs over output pixels, and each pixel finds the column entries that read it and sums them. The
scatter form walks the column matrix and adds each entry into the pixel it came from. On
overlapping windows that needs an atomic add, and a floating-point atomic add combines in an order
that varies between runs. SushiBLAS links `sushiruntime_deterministic_fp` PUBLIC, so the library
cannot accept that.

The terms `col2im` sums and the order it sums them in depend only on the geometry, so two runs
perform the same additions on the same values in the same order.

The divisibility test inverts the stride. Window row `h_o` reads pixel `h` through tap `r` exactly
when `h_o * stride_h - pad_top + r * dilation_h == h`, so `h_o` exists only if
`h + pad_top - r * dilation_h` is non-negative and divisible by `stride_h`. Without that test
every pixel looks as though every tap reads it.

`col2im` writes every pixel exactly once. A pixel that no window reads gets an explicit zero, so
the destination needs no prior clear.

### `kernels/reference/spatial/im2col.hpp`

The reference `im2col` is the specification of `im2col` and the answer
`ExecutionMode::ForceReference` produces. It is a plain nest of loops in index order, because a
reference kernel with an optimisation in it is a second implementation that can be wrong.

`ExecutionMode` has no mode that runs both kernels. The comparison is in
`tests/functional/unit/test_spatial_im2col.cpp`, which runs the two forced modes into two buffers
and checks them bit for bit.

An out-of-bounds tap writes zero. Padding never exists as a padded copy of the input, which saves
a pass over the whole tensor here and a crop in `col2im`.

### `kernels/reference/spatial/pool_avg.hpp`

`pool_avg` divides by `kernel_h * kernel_w`, which counts padded taps that contributed nothing.
That is PyTorch's `count_include_pad=True`, its default, and so what a network ported from PyTorch
expects. Dividing by the number of in-bounds taps is defensible and gives different numbers at
every border pixel of every padded pool. A test that does not pad cannot see the choice, so it is
recorded here and pinned by a test named for it.

`pool_avg` sums in ascending `(r, s)` order, so the additions depend only on the geometry and two
runs perform them identically.

`pool_avg_backward` uses the same divisibility gather as `pool_max_backward`. It is a separate
function, not the same one behind a mode flag. It adds `dy / area` for every window that covers
the pixel and tests no index. The two share a loop shape and nothing else: one selects through a
recorded index, the other broadcasts a constant. Folding them together would put a branch in the
inner loop. The divisor is the window area, as in `pool_avg`.

### `kernels/reference/spatial/pool_max.hpp`

The running maximum in `pool_max` starts at the lowest finite value of `T`, never at zero. Padding
is the absence of an element, and a maximum that starts at zero turns every all-negative border
window into 0. That is a plausible number, and no shape or dtype assertion catches it.

The first in-bounds tap is taken whatever its value, so a window whose every tap is -inf outputs
-inf and records that tap's index.

A tie keeps the first element in scan order (`>`, not `>=`). The recorded index is then
single-valued, and the equality test in `pool_max_backward` cannot match two taps of one window.
PyTorch follows the same convention.

The recorded index is the logical flat position `((n*H + h)*W + w)*C + c`. A strided offset into
this particular view would stop being comparable when the gradient arrives through an alias of the
same image with different strides.

A window with no in-bounds tap cannot occur, because `pooling_span_for` refuses any geometry that
leaves one window entirely in the padding. Such a window would record index -1, which matches no
pixel, and its gradient would be dropped and not misrouted.

A NaN tap wins its window: the first NaN in scan order is the output and its index is recorded, and
later taps do not replace it. The reference and SYCL kernels apply the same rule.

`pool_max_backward` is a gather over the windows that could have selected each input pixel. It
uses the same divisibility walk as `col2im`, with an equality test on the recorded index. The
scatter form walks the output and adds each gradient into the pixel its index names. Wherever
windows overlap that needs a floating-point atomic add, which combines in an order that varies
between runs. `include/SushiBLAS/engine/README.md`, under `spatial.hpp`, says why the library
cannot accept that.

A pixel that several overlapping windows selected accumulates all their gradients, in ascending
`(r, s)` order.

### `kernels/reference/spatial/upsample.hpp`

`upsample_nearest` computes `out[n, h_o, w_o, c] = in[n, h_o / scale_h, w_o / scale_w, c]`. It
copies and does no arithmetic, so the output's bit patterns are the input's.

Its span carries the two operands' extents and strides and a default-constructed
`Spatial::ConvGeometry`, which the kernel does not use. Nearest-neighbour upsampling needs two
numbers, and a `ConvGeometry` holds eight.

`upsample_nearest_backward` makes one pass per input pixel and gathers the fixed
`scale_h * scale_w` output elements that read it, in ascending `(dh, dw)` order. It scatters
nothing and adds nothing twice, so it uses no atomic and two runs sum identically.

### `kernels/reference/spatial/window_dot.hpp`, `window_dot_weights.hpp`

`window_dot` is a per-channel weighted window sum: output element `(n, h_o, w_o, c)` sums
`in * weights` over the taps of its window, each channel against its own taps, in ascending
`(r, s)` order. A tap outside the input is skipped, which is what a zero padding would have
contributed. The sum is formed in `Accumulator<T>` and rounded once on the store.

`window_dot_backward` is a gather, like `col2im`: each input element finds the windows that read
it by the divisibility test described under `kernels/reference/spatial/col2im.hpp` above, and
sums `dy * weights` over them. Every element of `dx` is written.

`window_dot_weights_backward` sums, for each tap of each channel, every output gradient times the
input element that tap read. The terms are added in ascending `(n, h_o, w_o)` order, so the order
depends only on the geometry. It is a fold of many terms into one value per tap, which is why the
reference sums it in a `ReferenceSum` and the SYCL kernel runs it through
`Reduce::reduce_segments`.

### `kernels/spatial.hpp`

`spatial.hpp` includes every spatial kernel by a quoted path relative to the repository root. They
are private to this tree for the reason given under `kernels/blas.hpp` above: only a kernel that a
public seam instantiates in a consumer's translation unit has to live under `include/`, and no
spatial kernel is one. The spatial operations are reached only through `SpatialOps`, whose methods
are compiled into this library.

### `kernels/sycl/blas/level1/level1.hpp`

The elementwise level 1 kernels (`axpy`, `copy`, `scal`, `swap`, `rot`) run one work-item per
element. The reductions (`asum`, `nrm2`, `dot`, `dotc`, `iamax`) run through
`Reduce::reduce_indexed` in `<SushiBLAS/support/reduce_1d.hpp>`, so they take the same single-pass or
two-stage shape, and the same fixed combination order, as the whole-tensor reductions. `asum` and
`dot` fold with `CompensatedSum`, `dotc` with `CompensatedComplexSum`, `nrm2` with
`ScaledSumOfSquares` and `iamax` with `RunningArgMax`. A HALF input accumulates in FLOAT32 and is
rounded once when the result is stored. The reference `asum`, `nrm2` and `dot` accumulate in
`double`, through `Kernels::ReferenceSum`, so on a long vector the two modes agree to the
precision of the SYCL sum, not bit for bit. The serial loops these kernels replaced summed without
compensation, so a result can differ from theirs in the last bits.

Each kernel takes a scratch pointer and its size in bytes. The caller in `src/engine/blas/level1/`
leases it through `Internal::allocate_level1_scratch` (`engine/blas/utils.hpp`), from the recorder's
scratch pool on the device of x, sized as `Reduce::planned_partials` over the folded count (2n for a
complex `nrm2`) times `K_LEVEL1_PARTIAL_BYTES`, the largest partial any of the five kernels stores
(a `std::complex<double>` or an `IndexedValue<double>`, 16 bytes). Each kernel divides the bytes by
its own partial size, and `partial_capacity` refuses at compile time a partial larger than the
constant.

The parallel kernels launch through `Launch::indexed`, not a bare `sycl::range<1>`, so they get
the same device-dependent geometry as every elementwise operation: flat on a CPU, grid-stride on a
GPU. `Launch::map` would also give them the 128-bit wide path, but it addresses its operands as
`in[i]`. A level 1 kernel reads `x[i * incx]`, and the increments are part of the BLAS signature,
so its access pattern is not the contiguous one `map` assumes. Specialising each kernel on
`incx == incy == 1` would reach the wide path. Nobody has written that, because the wide path only
engages on a GPU and this project has no GPU device to measure it on.

`nrm2` computes the scaling 2-norm through the library's own accumulator,
`Reduce::ScaledSumOfSquares`, and holds no second copy of the algorithm.
`include/SushiBLAS/support/README.md`, under `ScaledSumOfSquares`, describes the accumulator and
says why `sqrt(sum(x*x))`, which this kernel once computed, is wrong at both ends of the range.
The reference kernel computes the same norm a different way on purpose; see
`kernels/reference/blas/level1/level1.hpp` above.

`nrm2` folds a complex element as two reals. `abs1` (`|Re| + |Im|`) is the magnitude xASUM and
IxAMAX are defined on. A 2-norm needs a different one, and the earlier kernel, which used `abs1`,
returned a wrong answer for complex input.

### `kernels/sycl/blas/level2/gemv.hpp`

`gemv` and `symv` have two shapes, and `trmv` has the same two. The per-item shape launches through
`Launch::indexed`, so its grid is the one `<SushiBLAS/support/launch_1d.hpp>` plans for the device:
flat on a CPU, and on a GPU a grid-stride loop capped at what the machine can hold resident. Each
work-item reduces one whole row of `op(A)`. That reduction is why neither kernel can use
`Launch::map`: the output at index `o` is a function of `in_len` inputs, not of `in[o]`.

The work-group shape, in `kernels/sycl/blas/level2/row_reduction.hpp`, gives each output element
one work-group. Its work-items read the row at consecutive addresses, keep strided partial sums,
and fold them in a halving tree of fixed order, so the same device repeats a result to the bit.
`row_reduction_width` picks the shape: it takes the work-group shape when
`Reduce::planned_partials` would split a one-dimensional reduction of the row's length into two
stages on that device. On a CPU that means a row of at least `Reduce::K_CPU_TWO_STAGE_MIN_ELEMENTS`
elements. `gemv` takes the work-group shape only when the row of `op(A)` is contiguous: row-major
without a transpose, or column-major with one. In the other two cases the per-item shape already
reads neighbouring addresses across neighbouring work-items. `symv` reads half of each row
contiguously and half with stride `lda` in either shape, so it takes the work-group shape whenever
the row is long enough.

### `kernels/sycl/blas/level2/ger.hpp`

`ger` stays on a `sycl::range<2>`, unlike `gemv` and `symv`. `<SushiBLAS/support/launch_1d.hpp>`
plans one-dimensional geometries, and the index of `ger` has two dimensions: `A` is addressed by
`(i, j)` under a runtime layout, and the two axes have different strides. Flattening the index to
`i * n + j` to pass it through `Launch::indexed` would cost a division and a modulo per work-item,
and the backend could no longer pick a two-dimensional work-group. It would also buy the
grid-stride shape for a kernel whose launch is already sized by the output matrix, not by a vector
length. If `ger` ever needs the device split, the fix is a `plan_2d` beside `plan_1d`.

### `kernels/sycl/blas/level2/syr.hpp`

`syr` launches on a `sycl::range<2>` for the reason given under `kernels/sycl/blas/level2/ger.hpp`
above: its index has two dimensions and `launch_1d.hpp` plans one-dimensional grids. Half the
launched work-items return immediately. That inefficiency comes from which indices exist, and the
launch geometry does not affect it.

### `kernels/sycl/blas/level2/symv.hpp`, `syr2.hpp`

`symv` serves SYMV and HEMV and `syr2` serves SYR2 and HER2: each takes the conjugation policy
as a template parameter, as the reference kernels do. `symv` reads one triangle of A and has the
two row shapes described under `kernels/sycl/blas/level2/gemv.hpp` above. `syr2` writes only the
triangle `uplo` names and launches on a `sycl::range<2>` for the reason given under
`kernels/sycl/blas/level2/ger.hpp` above.

### `kernels/sycl/blas/level2/trmv.hpp`

`trmv` is two launches. The first copies `x` into a contiguous device scratch of `n` elements,
one work-item per element (`Detail::VectorSnapshot`, in `vector_snapshot.hpp`). The second computes each row product
`sum_k op(A)(i, k) * x(k)` from the snapshot and stores it into `x`, rounded once to `T`
(`Detail::TrmvTerm`, `Detail::TrmvStore`). Every row reads the snapshot and no row reads `x`, so
the rows are independent and run in parallel; the in-place hazard the reference kernel orders its
walk around does not exist here. The scratch is leased at record time by `Level2::trmv` and no
other task touches it.

The row product has the two shapes `gemv` has. A row of op(A) is contiguous exactly when GEMV's
row is, row-major without a transpose or column-major with one, and then
`row_reduction_width` may give each row a work-group that folds its stored columns in a
fixed-order tree (`Detail::reduce_row_windows_by_group` over a `Detail::TriangleRow`). Otherwise
one work-item computes one row in ascending column order. The overload that takes `group_width`
is there so a test can run both shapes on one input.

### `kernels/sycl/blas/level2/symmetric_product.hpp`, `triangular_product.hpp`, `triangular_solve.hpp`

The SYCL forms of the three shared loops described under
`kernels/reference/blas/level2/symmetric_product.hpp` above. `symmetric_product` runs per
work-item or per work-group by row, as `symv` does.

`triangular_product` is two launches, as `trmv` is: `Detail::VectorSnapshot` copies `x` into a
device scratch of `n` elements, then `Detail::TriangularProductRow` computes one row per
work-item from the snapshot, in ascending column order. `Level2::tbmv` and `Level2::tpmv` lease
the scratch at record time. The product has the per-item shape only; it has no work-group
shape.

`triangular_solve` is the blocked substitution of TRSM for one vector, read through a
`BandAccess` or a `PackedAccess`. One work-item solves the rows of a diagonal block in order
(`Detail::VectorSolveSteps`), then one work-item per row subtracts the block from the rows
still to be solved (`Detail::VectorSolveUpdate`). A solved row is subtracted from at most
`reach` later rows: the order for a packed triangle, `k` for a band, so a band update launches
`k` work-items and not `n`. Every row subtracts its terms in one column order at every block
width, ascending in a forward solve and descending in a backward one, so the block width does
not change a bit of the result. When `RunningSolution` needs a workspace, a first launch copies
`x` into it (`Detail::VectorSolveOpen`).

`plan_vector_solve` in `triangular_plan.hpp` picks the width. It asks `plan_triangular_solve`
whether `min(n, reach)` rows repay the launches of a block, and returns one block of the whole
order when they do not. A band narrower than the break-even therefore still runs as one
sequential task: a row of it depends on the row before, and it has no work to spread.

### `kernels/sycl/blas/level3/syr2k.hpp`

`syr2k` computes the two products of SYR2K, or of HER2K under `Conjugated`, on one triangle of C
in a tiled kernel of its own. `Detail::Syr2kTileConfig` fits two register blocks, one per
product, in the byte budget GEMM gives one. It stages its tiles in local memory and throws
`std::runtime_error` when the device cannot host the work-group and the tiles. `Syr2kTile`
names that demand per element type, and the recorder refuses such a device at record time, so a
call through `Level3` does not reach the throw.
A zero `beta` does not read the prior C.

### `kernels/sycl/blas/level3/syrk.hpp`

No SYRK tiling kernel exists upstream: portBLAS's `doc/MissingFeatures.md` lists SYRK as
unimplemented (see `NOTICE.md`). `syrk` composes SushiBLAS's ported GEMM tiling, the way
portBLAS's SYMM composes onto its GEMM kernel. C = alpha\*op(A)\*op(A)^T is
GEMM(A, A, transA=trans, transB=!trans) restricted to one triangle of C, so `syrk` reuses the
operand-addressing helpers of `gemm.hpp` and adds the triangular output mask. The mask and the
composition are SushiBLAS work, not a port.

`syrk` does not reuse GEMM's tile geometry. It owns `Detail::SyrkTileConfig`, so retuning GEMM's
register block cannot retune SYRK without anyone seeing it. `include/SushiBLAS/support/README.md`,
under `device_policy.hpp`, makes the same argument for keeping tile sizes out of the device
profile. SYRK does not have GEMM's split between a local-memory and a no-local kernel: its tiled
kernel always takes the local-memory shape, and throws `std::runtime_error` when the device
cannot host the tile's work-group. `SyrkTile` names that demand per element type, and the
recorder refuses such a device at record time.

`syrk` has a second path. It plans the product with `plan_packed_product` and, when
`packed_product_is_ready` holds for the scratch it was given, runs the packed product of
`include/SushiBLAS/kernels/sycl/blas/level3/packed_product.hpp` with op(A) as the first factor
and its transpose, conjugated under `Conjugated`, as the second. `Detail::SyrkStore` is the store
of that launch: its `wants` skips a macro-block that lies wholly in the triangle C does not keep,
and it writes `alpha * sum + beta * C` only where the triangle holds the element. HERK is the
same kernel under `Conjugated`.

`Detail::SyrkTileConfig` holds the fixed shape SYRK has always used: a 4x4 register block in an
8x8 work-group. It is written out and not borrowed from `GemmLocalTile`, so GEMM's per-type
derivation of its register block stays inside GEMM. A shared config would make a change to one
kernel's tuning an unseen change to the other's.

### `kernels/sycl/blas/level3/trsm.hpp`

`trsm` also backs `Level2::trsv`, which is a TRSM with one right-hand-side column (n = 1); see
`src/engine/blas/level2/trsv.cpp`.

The substitution in TRSM is sequential along one matrix dimension. Each of the `n` columns (left
side) or `m` rows (right side) is an independent triangular solve against the same A, with no data
hazard between them. `trsm` runs those in parallel, and it also cuts each substitution into
diagonal blocks so that the work between blocks is parallel too.

For each diagonal block the kernel makes two launches. `Detail::TriangularBlockSolve` runs one
work-item per right-hand-side vector and batch entry, and solves the block's steps for that
vector in order. `Detail::TriangularBlockUpdate` then runs one work-item per unknown that is not
yet solved, and subtracts the block's solved terms from its running value. The update of one
unknown does not depend on the update of another, which is where a solve with few right-hand
sides gets its parallelism. Every launch is ordered behind the one before it. With one block of
the whole order the kernel is the unblocked form: one launch, one substitution per work-item.

The solved elements are carried in a `RunningSolution`. For HALF that needs a workspace of
`batch_size * m * n` wide elements, which the operation leases at record time; for every other
type the solution lives in B and the workspace is never read.

### `kernels/sycl/blas/level3/triangular_system.hpp`

`triangular_system` decodes the side, the triangle, the transpose and the diagonal flag of a TRSM
call into one description, so the kernel has one substitution to write. A `TriangularSystem`
names unknown `(s, v)` as step `s` of the substitution for vector `v`. `coefficient` reads the
factor of unknown `t` in step `s`, `unknown` gives the offset of `(s, v)` in B, and `forward`
says whether step `s` needs the steps below it.

The left side solves op(A)\*X = alpha\*B column by column, so the order is `m`, the vectors are
the `n` columns, and the coefficient access is that of op(A). op(A) is upper exactly when the
transpose flag and `uplo == Upper` differ in truth, and a lower op(A) is solved forward. The
right side solves X\*op(A) = alpha\*B, which is op(A)^T \* X^T = alpha\*B^T: the same system
with both accesses transposed, the order `n`, the vectors the `m` rows, and the direction
reversed.

### `kernels/sycl/blas/level3/triangular_plan.hpp`

`plan_triangular_solve` chooses the block size on the host from the device profile, the order of
the system and the number of independent substitutions, `streams` (vectors times batch size). A
block is `K_CPU_SOLVE_BLOCK`, 256 steps, on a CPU and `K_GPU_SOLVE_BLOCK`, 128, elsewhere. The
plan is one block of the whole order in two cases: when the substitutions alone fill the device
(`streams` is at least the work-items the device keeps running at once), and when the system is
too small to repay the extra launches, which is
`order * (slots - streams) <= K_SOLVE_BREAK_EVEN_BLOCKS * block * slots`.

### `kernels/sycl/sparse/row_blocks.hpp`

A CSR product that is transposed, or symmetric from one stored triangle, scatters: one stored
entry adds into an output element chosen by its column, and two rows can choose the same one.
Giving each row a work-item would need an atomic add, whose combination order varies between
runs. The kernels split the rows into blocks instead. `scatter_blocks` picks the count: the
smallest of the work-items the device keeps running at once, `K_MAX_SCATTER_BLOCKS` = 256, and
the count at which the partial sums reach `K_SCATTER_PARTIALS_PER_ENTRY` = 2 per stored entry,
and never below 1. `block_first_row` cuts the rows so the blocks hold about the same number of
entries. Block `b` alone writes its own slice of the scratch,
`partials[b * outputs, (b + 1) * outputs)`, walking its rows in order. A second launch, one
work-item per output, folds the blocks' partials in ascending block order and stores
`alpha * sum + beta * y`. Ascending block order is ascending row order, so the sum has one fixed
order on a given device.

### `kernels/sycl/sparse/level1.hpp`, `spmv.hpp`, `spmm.hpp`, `spsymv.hpp`, `sptrsv.hpp`

The level 1 routines run one work-item per stored entry. `axpyi`, `gthrz`, `sctr` and `roti`
write the dense vector at stored positions, and their precondition is that no position is stored
twice, so no two work-items write one element. The dot products fold through a compensated sum,
as the dense `dot` does.

`spmv` without a transpose gives each row of the output one work-item. With a transpose it
scatters through `row_blocks.hpp`, and so does `spsymv` in every case. `spmm` gives each column
of C one work-item, so no element has two writers. `sptrsv` is one work-item that solves the
whole system in order, a lower triangle from row 0 down and an upper one upwards. It computes
no dependency levels of the rows.

### `kernels/sycl/spatial/window_dot.hpp`, `window_dot_weights.hpp`

`window_dot` runs one work-item per output element and `window_dot_backward` one per input
element, each summing its own terms in the order the reference kernel does. Neither file contains
a `sycl::atomic_ref`. `window_dot_weights_backward` reduces, for each tap of each channel, the
`N * H_o * W_o` terms of that tap's gradient through `Reduce::reduce_segments`: the taps are the
segments, `Detail` functors open a tap's terms and store its sum, and the fold has the fixed
order of the segmented reduction in `<SushiBLAS/support/reduce_axis.hpp>`.

### `kernels/sycl/random/orthonormalize.hpp`

`orthonormalize` backs `RandomOps::orthogonal`. Orthogonal weight initialisation is usually
written as "fill with N(0,1), take the Q of a QR decomposition". SushiBLAS has no QR (`LinalgOps`
is quarantined and unimplemented), so `orthonormalize` computes the same Q by modified
Gram-Schmidt. It is not a general QR: it returns Q, discards R, and has no pivoting or rank
detection, which is why it is not exported as a linear-algebra primitive.

The work is spread over work-groups, not over one work-item. Step i is one kernel with one
work-group per vector m > i. Each group computes `v_i . v_i` and `v_i . v_m` together and
subtracts `(v_i . v_m) / (v_i . v_i) * v_i` from `v_m`. Vector i is not normalised during the
sweep, so no group writes a vector another group reads in the same step. After the sweeps, one
kernel with one group per vector scales each vector to norm `gain`.

Every sum is deterministic. A work-item accumulates elements `lid, lid + G, lid + 2G, ...` in
that order, and the group combines the G partial sums by a fixed pairwise tree in local memory.
G is 64, or the largest power of two the device accepts below that, so on one device the result
is the same bits on every run.

The sweep runs twice before the final scaling (`K_ORTHONORMAL_SWEEPS`, in
`<SushiBLAS/support/orthonormal_sweep.hpp>`). One modified Gram-Schmidt
sweep loses orthogonality in proportion to the matrix's condition number times the rounding
unit; a second sweep over the already nearly orthogonal vectors brings the loss down to the
rounding unit. That keeps a 64x64 FLOAT32 matrix inside the 1e-4 Gram tolerance of
`tests/functional/unit/test_orthogonal.cpp`. The cost is 2 * (count - 1) project kernels and one
scaling kernel per call.

`tiny_norm_sq` is (1e-12)^2. A vector whose squared norm is at or below it is treated as zero: it
is projected out of nothing and left unscaled except by `gain`, since dividing by its norm would
produce infinities. The Gaussian input this kernel is used with reaches that case with
probability zero.

### `kernels/sycl/random/philox.hpp`

`philox.hpp` implements Philox4x32-10, a counter-based generator (Salmon, Moraes, Dror, Shaw,
"Parallel Random Numbers: As Easy as 1, 2, 3", SC'11). It is stateless: the same (key, counter)
pair always produces the same four words, so any block can be drawn independently and in
parallel with no shared state. The round constants were checked against the reference Random123
implementation (DEShawResearch/random123, `include/Random123/philox.h`).

The file replaces oneMKL's `philox4x32x10` (SB-2). The algorithm is unchanged; the backend is
hand-written SYCL with no oneMKL dependency.

`philox_block` computes one block with key = (seed_lo, seed_hi) and counter = (offset_lo,
offset_hi, block_lo, block_hi); offset and block occupy separate words, so two tasks never share a
(key, counter) pair, whatever their tensor sizes.

`uniform01_float` takes the top 24 bits of a word and `uniform01_double` the top 53 bits of two
words, so each result lies in [0,1) and is exactly representable: 1 is never returned, 0 can be.
`uniform01_block` applies them to every word of a block, giving `K_DRAWS_PER_BLOCK` values: four
FLOAT32 draws or two FLOAT64 draws. Callers that take a logarithm take it of 1 - u: Box-Muller in
`normal_block` uses log(1 - u1), and `exponential` uses -log1p(-u). `uniform_block` returns the
value just below hi when rounding of lo + u(hi - lo) would reach hi.

`normal_block` uses Box-Muller on consecutive uniform pairs and keeps both outputs, `r cos(theta)`
and `r sin(theta)`. A block gives four FLOAT32 normals or two FLOAT64 normals.
The file spells 2 pi out as a literal, because MSVC defines `M_PI` only under
`_USE_MATH_DEFINES`.

`draw_index` returns a uniformly distributed integer in [0, bound), drawn from the same
(key, counter) scheme as the float generators. It uses integer arithmetic only. Fisher-Yates needs
an index, and routing the index through a float would make the result of a shuffle depend on the
device's fp64 support. The bias of the modulo is bounded by bound / 2^64, below one part in 10^13
for any tensor that fits in memory, so a rejection loop would change nothing measurable.

### `kernels/sycl/random/poisson.hpp`

`draw_poisson` returns one Poisson count for one element. Each element owns
`K_POISSON_BLOCKS_PER_ELEMENT` = 64 consecutive Philox blocks, starting at element * 64, so no two
elements share a uniform draw.

Below `K_POISSON_PTRS_MIN_LAMBDA` = 10, `poisson_knuth` multiplies FLOAT64 uniforms, two per
block, until the product reaches exp(-lambda), and returns how many it multiplied before that. It
stops after 128 uniforms; at lambda 10 a count of 128 has probability below 1e-80.

From lambda 10 up, `poisson_ptrs` runs Hoermann's transformed rejection with squeeze (PTRS,
Insurance: Mathematics and Economics 12, 1993), the method NumPy and TensorFlow use for large
means. Each attempt reads one block for its two uniforms u and v. The constants a, b, 1/alpha and
v_r depend only on lambda, so `make_poisson_params` computes them on the host once per call. The
hat is tight for lambda >= 10, so an attempt is accepted with probability well above one half
and 64 attempts all fail with probability below 2^-64. If they do, the draw returns
floor(lambda).

Both methods compute in FLOAT64 whatever the tensor's dtype, so `RandomOps::poisson` throws
`UnsupportedExecution` at record time when the tensor's device lacks `aspect::fp64`.

### `kernels/sycl/random/truncated_normal.hpp`

`draw_truncated_standard_normal` draws standard normal values inside [lo, hi] by inverting the
distribution function. It maps one Philox block to uniforms in [Phi(lo), Phi(hi)) with
`uniform_block`, then bisects [lo, hi] for each one, `std::numeric_limits<T>::digits` steps, until
`standard_normal_cdf` meets the uniform. The bisection only narrows its interval, so a result
never leaves [lo, hi].

The caller keeps lo <= 0, mirroring an interval above zero, because Phi loses its precision near
1. It also clamps both bounds to `K_TRUNCATED_NORMAL_MAX_BOUND` = 40 on the host: Phi(-40) is
below the smallest double, so the clamp removes no mass, and a fixed step count over a bound such
as -1e30 would otherwise end tens of orders of magnitude away from the mass.

### `kernels/sycl/spatial/broadcast.hpp`

No kernel in the SYCL `broadcast.hpp` accumulates. Each work-item writes one output element from
one input element and at most one per-channel scalar. There is no reduction, no running total, and
nothing whose combination order could vary between two runs or differ from the reference kernel's.
The file contains no `sycl::reduction`, no `sycl::reduce_over_group` and no floating-point
`sycl::atomic_ref`, and none may be added. The pooling backwards avoid those constructs as a
constraint; these operations have no overlap to combine at all.

`channel_affine` runs one work-item per output element and addresses `gamma` and `beta` through
their own strides. `broadcast_spatial` runs one work-item per output element, each reading the
single `(n, c)` entry its position selects. `layout_convert` runs one work-item per element and
indexes each side through its own four strides; it has no direction flag. The reasons for all
three are under `kernels/reference/spatial/broadcast.hpp` above.

### `kernels/sycl/spatial/channels.hpp`

`copy_channel_range` runs one work-item per copied element, `N * H * W * count` of them. The
copied-channel index is innermost, so an NHWC copy of a contiguous range reads and writes
consecutive addresses. `kernels/reference/spatial/channels.hpp` above says why one kernel serves
both concatenation and slicing, and what `span` carries.

### `kernels/sycl/spatial/col2im.hpp`

`col2im` runs one work-item per input-image element, each summing its own contributions in
ascending `(r, s)` order. The file contains no `sycl::atomic_ref` and none may be added. The
gather form is there so that overlapping windows need no atomic add, whose combination order
would vary between runs and break the library's guarantee of bit-reproducible results.

`kernels/reference/spatial/col2im.hpp` above explains how the divisibility test inverts the
stride.

### `kernels/sycl/spatial/im2col.hpp`

`im2col` runs one work-item per output element, which makes it a gather: every item writes exactly
one entry and reads exactly one input element, or none for a padded tap. No two items write the
same address, so there is nothing to synchronise and nothing whose combination order could vary
between runs.

Consecutive items differ in `c` first, so a sub-group reads a contiguous run of channels. That is
part of the case for the NHWC layout.

### `kernels/sycl/spatial/pool_avg.hpp`

`pool_avg` runs one work-item per output element. `kernels/reference/spatial/pool_avg.hpp` above
says why the divisor counts padded taps (`count_include_pad=True`).

`pool_avg_backward` runs one work-item per input element, each gathering its own share. The file
contains no `sycl::atomic_ref` and none may be added.

### `kernels/sycl/spatial/pool_max.hpp`

`pool_max` runs one work-item per output element. It must match two conventions of the reference
kernel, both explained under `kernels/reference/spatial/pool_max.hpp` above: the running maximum
starts at `lowest()` and never at zero, and a tie keeps the first element in scan order.

`pool_max_backward` runs one work-item per input element, each gathering its own contributions.
The file contains no `sycl::atomic_ref` and none may be added, for the reason given in the same
section.

### `kernels/sycl/spatial/upsample.hpp`

`upsample_nearest` runs one work-item per output element, addressed as
`kernels/reference/spatial/upsample.hpp` above describes.

`upsample_nearest_backward` runs one work-item per input element, each gathering its own fixed
`scale_h * scale_w` block in ascending order. The file contains no `sycl::atomic_ref` and none may
be added.

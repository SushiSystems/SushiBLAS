# kernels

The kernels that are private to the library. An operation has a plain C++ kernel under
`reference/`, which is the mathematical ground truth, and a SYCL kernel under `sycl/`, which runs
on the device. The random kernels exist only under `sycl/`. GEMM's pair is not here: it is under
`include/SushiBLAS/kernels/`, for the reason given under `kernels/blas.hpp` below.

| Path | Holds |
| --- | --- |
| `blas.hpp` | The include of every BLAS kernel |
| `math.hpp` | No include |
| `random.hpp` | The include of the SYCL random kernels |
| `spatial.hpp` | The include of every spatial kernel |
| `reference/blas/`, `sycl/blas/` | Levels 1 to 3; level 1 is one file per side, levels 2 and 3 one file per routine |
| `reference/spatial/`, `sycl/spatial/` | im2col, col2im, pooling, upsampling, channel and broadcast copies |
| `sycl/random/` | Philox4x32-10 and the modified Gram-Schmidt sweep |

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

### `kernels/reference/blas/level2/trmv.hpp`

`trmv` runs in place without a temporary copy of `x`. The effective triangle of op(A) sets the
direction of the walk: upper if the support after the transpose is k >= i, lower otherwise. The
walk visits `i` in the order that reads each `x[i]` before the kernel overwrites it and never
after. The section on `kernels/sycl/blas/level2/trmv.hpp` below shows why both directions are
safe.

### `kernels/reference/blas/level3/syrk.hpp`

`syrk` addresses its operand by the same two-stride rule as `Reference::gemm`. It once carried the
same defect: the `COLUMN_MAJOR` arm was inverted, which made `layout` a no-op for A and read past
its end for some shapes. The identity that catches the defect is in
`include/SushiBLAS/kernels/README.md`, under `reference/blas/level3/gemm.hpp`.

### `kernels/reference/blas/level3/trsm.hpp`

`trsm` also backs `Level2::trsv`, which is a TRSM with one right-hand-side column (n = 1); see
`src/engine/blas/level2/trsv.cpp`.

The left side is solved one column of B at a time, by forward substitution when the effective
triangle of A is lower and by backward substitution when it is upper. The section on
`kernels/sycl/blas/level3/trsm.hpp` below derives the effective triangle. The right side is the
same system transposed once more, row by row: X\*op(A) = alpha\*B is equivalent to
op(A)^T \* X^T = alpha\*B^T.

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
`ExecutionMode::FORCE_REFERENCE` produces. It is a plain nest of loops in index order, because a
reference kernel with an optimisation in it is a second implementation that can be wrong.

`ExecutionMode::VERIFY` does not compare the two kernels, whatever its name suggests. It runs the
device kernel and then overwrites the result with the reference one. The comparison is in
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
`Reduce::RunningMax` uses the same identity for the same reason.

A tie keeps the first element in scan order (`>`, not `>=`). The recorded index is then
single-valued, and the equality test in `pool_max_backward` cannot match two taps of one window.
PyTorch follows the same convention.

The recorded index is the logical flat position `((n*H + h)*W + w)*C + c`. A strided offset into
this particular view would stop being comparable when the gradient arrives through an alias of the
same image with different strides.

A window with no in-bounds tap cannot occur, because `window_span_for` refuses an output extent
whose last window starts past the input. Such a window would record index -1, which matches no
pixel, and its gradient would be dropped and not misrouted.

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

### `kernels/spatial.hpp`

`spatial.hpp` includes every spatial kernel by a quoted path relative to the repository root. They
are private to this tree for the reason given under `kernels/blas.hpp` above: only a kernel that a
public seam instantiates in a consumer's translation unit has to live under `include/`, and no
spatial kernel is one. The spatial operations are reached only through `SpatialOps`, whose methods
are compiled into this library.

### `kernels/sycl/blas/level1/level1.hpp`

The elementwise level 1 kernels (`axpy`, `copy`, `scal`, `swap`, `rot`) run one work-item per
element. The reductions (`asum`, `nrm2`, `dot`, `iamax`) each run as one serial device task. Level
1 vectors are small next to this library's level 3 workloads, so a tree reduction is not written
yet. A profile that shows large-vector level 1 reductions would be the reason to write one.

The parallel kernels launch through `Launch::indexed`, not a bare `sycl::range<1>`, so they get
the same device-dependent geometry as every elementwise operation: flat on a CPU, grid-stride on a
GPU. `Launch::map` would also give them the 128-bit wide path, but it addresses its operands as
`in[i]`. A level 1 kernel reads `x[i * incx]`, and the increments are part of the BLAS signature,
so its access pattern is not the contiguous one `map` assumes. Specialising each kernel on
`incx == incy == 1` would reach the wide path. Nobody has written that, because the wide path only
engages on a GPU and this project has no GPU device to measure it on.

`nrm2` computes the scaling 2-norm through the library's own accumulator,
`Reduce::ScaledSumOfSquares`, and holds no second copy of the algorithm.
`include/SushiBLAS/engine/README.md`, under `ScaledSumOfSquares`, describes the accumulator and
says why `sqrt(sum(x*x))`, which this kernel once computed, is wrong at both ends of the range.
The reference kernel computes the same norm a different way on purpose; see
`kernels/reference/blas/level1/level1.hpp` above.

`nrm2` folds a complex element as two reals. `abs1` (`|Re| + |Im|`) is the magnitude xASUM and
IxAMAX are defined on. A 2-norm needs a different one, and the earlier kernel, which used `abs1`,
returned a wrong answer for complex input.

### `kernels/sycl/blas/level2/gemv.hpp`

`gemv` launches through `Launch::indexed`, so its grid is the one
`<SushiBLAS/engine/launch_1d.hpp>` plans for the device: flat on a CPU, and on a GPU a grid-stride
loop capped at what the machine can hold resident. Each work-item still reduces one whole row of
`op(A)`. That reduction is why `gemv` cannot use `Launch::map`: the output at index `o` is a
function of `in_len` inputs, not of `in[o]`.

The parallelism is one work-item per output element. That shape is right while `m` is large and
wrong once `m` is small relative to the device: a 1xN GEMV is one work-item. The standard fix is a
cooperative reduction with one row per work-group. It changes what each work-item computes, so it
is a separate change from the launch geometry.

### `kernels/sycl/blas/level2/ger.hpp`

`ger` stays on a `sycl::range<2>`, unlike `gemv` and `symv`. `<SushiBLAS/engine/launch_1d.hpp>`
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

### `kernels/sycl/blas/level2/trmv.hpp`

In `trmv`, `x[i]` depends only on `x[k]` for `k` on one side of `i`. The effective triangle of
op(A) picks the side: `eff_upper` means the support is k >= i. The kernel walks `i` from the open
end towards the closed end, increasing for `eff_upper` and decreasing otherwise. Every `x[k]` a
step needs is then still unwritten when the step runs, because the indices already overwritten lie
outside the dependency range of the remaining steps. Each step reads what an earlier step left
alone, so the steps cannot run in parallel, and `trmv` is one sequential task.

### `kernels/sycl/blas/level3/syrk.hpp`

No SYRK tiling kernel exists upstream: portBLAS's `doc/MissingFeatures.md` lists SYRK as
unimplemented (see `NOTICE.md`). `syrk` composes SushiBLAS's ported GEMM tiling, the way
portBLAS's SYMM composes onto its GEMM kernel. C = alpha\*op(A)\*op(A)^T is
GEMM(A, A, transA=trans, transB=!trans) restricted to one triangle of C, so `syrk` reuses the
operand-addressing helpers of `gemm.hpp` and adds the triangular output mask. The mask and the
composition are SushiBLAS work, not a port.

`syrk` does not reuse GEMM's tile geometry. It owns `Detail::SyrkTileConfig`, so retuning GEMM's
register block cannot retune SYRK without anyone seeing it. `include/SushiBLAS/engine/README.md`,
under `device_policy.hpp`, makes the same argument for keeping tile sizes out of the device
profile. SYRK does not have GEMM's split between a local-memory and a no-local kernel. It always
takes the local-memory shape, as it did before, and the split is a named follow-up in `NOTICE.md`.

`Detail::SyrkTileConfig` holds the fixed shape SYRK has always used: a 4x4 register block in an
8x8 work-group. It is written out and not borrowed from `GemmLocalTile`, so GEMM's per-type
derivation of its register block stays inside GEMM. A shared config would make a change to one
kernel's tuning an unseen change to the other's.

### `kernels/sycl/blas/level3/trsm.hpp`

`trsm` also backs `Level2::trsv`, which is a TRSM with one right-hand-side column (n = 1); see
`src/engine/blas/level2/trsv.cpp`.

The substitution in TRSM is sequential along one matrix dimension. Each of the `n` columns (left
side) or `m` rows (right side) is an independent triangular solve against the same A, with no data
hazard between them. `trsm` parallelises over that dimension: one work-item runs a full serial
substitution for one column or row.

The kernel picks the direction of each substitution from `eff_lower`, the effective triangle,
which is derived as follows. The left side solves op(A)\*X = alpha\*B column by column. Define
`A_eff(i,k) = transA ? A(k,i) : A(i,k)`, which is op(A)(i,k). If A is upper and `transA` is set,
op(A) = A^T is lower; if A is lower and `transA` is set, op(A) is upper. So op(A) is lower exactly
when `transA` XOR lower(A), which is `eff_lower = transA ? upper : !upper`. With `eff_lower`
known, forward substitution (lower) or backward substitution (upper) over `A_eff` applies
directly, whatever the original transpose and layout.

The right side solves X\*op(A) = alpha\*B, which is equivalent to op(A)^T \* X^T = alpha\*B^T.
That is the left-side system with one more transpose, so `eff_lower` uses `transA_row = !transA`
in place of `transA`. The substitution walks A's `(p, q)` index against a row of B (fixed `i`,
varying column) where the left side walks a column.

### `kernels/sycl/random/orthonormalize.hpp`

`orthonormalize_mgs` backs `RandomOps::orthogonal`. Orthogonal weight initialisation is usually
written as "fill with N(0,1), take the Q of a QR decomposition". SushiBLAS has no QR (`LinalgOps`
is quarantined and unimplemented), so `orthonormalize_mgs` computes the same Q directly. Modified
Gram-Schmidt and Householder QR produce the same column space by different routes. MGS was chosen
for two reasons. It is about a dozen lines and stores no reflectors. It also orthogonalises each
vector against the updated vectors, not the originals, which keeps it accurate enough for
initialisation on well-conditioned Gaussian input.

`orthonormalize_mgs` is not a general QR. It returns Q, discards R, and has no pivoting or rank
detection, which is why it is not exported as a linear-algebra primitive.

`tiny` is the norm at or below which the sweep leaves a vector unscaled. Below it a vector is
numerically indistinguishable from zero, and dividing by its norm would produce infinities. The
Gaussian input this kernel is used with reaches that case with probability zero. The guard is
there so that a case nobody expected does not fill a weight matrix with NaN.

### `kernels/sycl/random/philox.hpp`

`philox.hpp` implements Philox4x32-10, a counter-based generator (Salmon, Moraes, Dror, Shaw,
"Parallel Random Numbers: As Easy as 1, 2, 3", SC'11). It is stateless: the same (key, counter)
pair always produces the same four words, so any element index can be drawn independently and in
parallel with no shared state. The round constants were checked against the reference Random123
implementation (DEShawResearch/random123, `include/Random123/philox.h`).

The file replaces oneMKL's `philox4x32x10` (SB-2). The algorithm is unchanged; the backend is
hand-written SYCL with no oneMKL dependency.

`uniform01_float` and `uniform01_double` map a raw 32-bit or 64-bit word to the open interval
(0,1) and never return exactly 0 or 1. Their callers take log(u): `log_normal` through Box-Muller,
and the inverse CDF of `exponential`.

`draw_uniform` makes one Philox call per element, with key = (seed_lo, seed_hi) and
counter = (offset_lo, offset_hi, index_lo, index_hi). The counter's high word is the per-task
`offset`. The old oneMKL `skip_ahead` call used offset\*size there. With the plain offset, two
tasks that generate random numbers never share a (key, counter) pair, whatever their tensor sizes.

`draw_normal` uses Box-Muller. One Philox call yields both uniforms for one Gaussian sample: word
pair 0 gives u1 and word pair 1 gives u2. Only the cosine branch (z0) is used and z1 is discarded:
the first version trades throughput for a shorter kernel.

`draw_index` returns a uniformly distributed integer in [0, bound), drawn from the same
(key, counter) scheme as the float generators. It uses integer arithmetic only. Fisher-Yates needs
an index, and routing the index through a float would make the result of a shuffle depend on the
device's fp64 support. The bias of the modulo is bounded by bound / 2^64, below one part in 10^13
for any tensor that fits in memory, so a rejection loop would change nothing measurable.

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

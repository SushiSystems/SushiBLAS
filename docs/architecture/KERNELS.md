# 7. Kernels: reference and SYCL

A kernel is a function template on the element type that receives pointers, extents, strides and
flags. The dense BLAS, sparse and spatial families have a kernel pair per operation:

- **`modules/kernels/<family>_kernels/include/SushiBLAS/<family>_kernels/reference/`** is plain
  C++ with no dependency beyond the support headers. It handles both `Core::Layout` values and
  states the result the operation is defined to have.
- **`modules/kernels/<family>_kernels/include/SushiBLAS/<family>_kernels/sycl/`** takes the
  `sycl::queue&` and a
  `const std::vector<sycl::event>& deps`, makes every command group depend on `deps`, and returns
  the event of its last submission.

`blas_kernels`, `random_kernels`, `sparse_kernels` and `spatial_kernels` are internal modules:
their headers are not installed and a consumer cannot include them. A kernel a public seam
instantiates in a consumer's translation unit cannot be internal, so those live in
`modules/kernels/blas_seam_kernels/` and are installed: GEMM's reference and SYCL kernels with the packed product they share with SYRK, the
epilogue headers, the access policies (`operand_access.hpp`, `symmetric_access.hpp`,
`band_access.hpp`, `packed_access.hpp`, `row_cursor.hpp`, `symmetric_row.hpp`,
`conjugation.hpp`), `packed_product_plan.hpp` and the two Givens headers.

The elementwise, non-linear, logic, reduction and random families have no kernel pair. Their SYCL
kernel is a launch primitive of the support layer around the operation's functor, and their
reference path is the serial loop of [section 3](EXECUTION_MODES.md).

## The support layer

`modules/foundation/support/` sits below the kernels and the engine and includes neither.
[Its README](../../modules/foundation/support/README.md) lists each header with what it declares
and holds the reasoning behind it. The headers this page names most are `device_policy.hpp`
(`Device::Profile` and the launch geometry), `launch_1d.hpp` and `serial_1d.hpp` (the per-element
launch and its host loop), and `reduce_1d.hpp`, `reduce_axis.hpp` and `serial_reduce.hpp` (the
fixed-order folds, their accumulators and their host folds).

**Cache blocking.** `Device::CacheBlocking` holds three byte budgets (the bytes the innermost
loop keeps resident, the packed block the inner loops sweep again, and the block walked once per
sweep), two thresholds below which a kernel does not pack, and the work-items a blocked launch
starts per compute unit. `cache_blocking_for(cache_bytes)` derives the budgets from the cache a
device reports, and from an assumed 256 KiB when it reports none. The values are starting points
derived from cache sizes; none has been tuned against a measurement.

**Determinism.** SushiBLAS links `SushiRuntime::SushiRuntime` PUBLIC and inherits
`sushiruntime_deterministic_fp` through it, and its kernels keep the
order of every floating-point sum a function of the device, the element type and the extents.
That is why the floating-point folds use `Reduce::reduce` and not `sycl::reduction`, whose
combination order the SYCL specification leaves free, and why every backward in the spatial
family and every sparse scatter is written without an atomic add. The design aim is a result
that repeats bit for bit on one device; the regression tests that hold it have run on one
device, an RTX 3080 Ti. Two devices, or the two modes, may plan different shapes and differ in the last bits
of a sum.

## Dense BLAS

**Flags.** `core/blas_types.hpp` declares the four flag enumerations, with the CBLAS values:

| Enumeration | Enumerators |
| --- | --- |
| `Core::Transpose` | `NoTrans`, `Trans`, `ConjTrans` |
| `Core::Uplo` | `Upper`, `Lower` |
| `Core::Diag` | `NonUnit`, `Unit` |
| `Core::Side` | `Left`, `Right` |

A flag outside its enumeration throws `std::invalid_argument` at record time (`require_flag`).

**Scalars.** `BLASScalar` (`operands/scalar.hpp`) is the scalar argument of a BLAS call. It
converts from `double` and from `std::complex<double>`, holds the value as given, and
`as<T>()` rounds it once to the element type inside the task. A scalar with a non-zero imaginary
part passed to a real tensor throws at record time (`require_real_scalar`). The scalars the
standard defines as real (`rot`'s `c` and `s`, the `alpha` of `her`, `hpr` and `herk`, the `beta`
of `herk` and `her2k`, and `ger`'s `alpha`) are `double`.

```cpp
SushiBLAS::BLASOps(*this->engine).herk(A, C, uplo, Core::Transpose::ConjTrans,
                                       2.0, 3.0);
// ...
SushiBLAS::BLASOps(*engine).trsv(A, b, uplo, a.trans, Core::Diag::NonUnit);
```

Source: `modules/operations/blas/tests/test_blas_complex.cpp`

`herk` takes its two scalars as `double`, and `trsv` takes the three flags of a triangular
routine.

**Routines.** The dtypes are HALF, FLOAT32, FLOAT64, COMPLEX32 and COMPLEX64 unless a row says
otherwise.

| Level | Routines |
| --- | --- |
| 1 | `axpy`, `scal`, `copy` (every dtype, the integers included), `swap`, `dot` (real only), `dotu`, `dotc`, `nrm2`, `asum`, `iamax` (zero-based `INT64` result), `rot`, `rotg`, `rotm` (real only), `rotmg` (FLOAT32 and FLOAT64) |
| 2, general | `gemv`, `ger` (real only), `geru`, `gerc` |
| 2, symmetric and Hermitian | `symv`, `hemv`, `syr`, `her`, `syr2`, `her2` |
| 2, triangular | `trmv`, `trsv` |
| 2, band | `gbmv`, `sbmv`, `hbmv`, `tbmv`, `tbsv` |
| 2, packed | `spmv`, `hpmv`, `tpmv`, `tpsv`, `spr`, `hpr`, `spr2`, `hpr2` |
| 3 | `gemm`, `symm`, `hemm`, `trsm`, `syrk`, `herk`, `syr2k`, `her2k` |

`trmm` is not implemented. The Level 3 routines are batched above rank 2: the output sets the
batch count, and an input of rank 2 is reused for every matrix. A Level 2 matrix has rank 2. A
band operand is the band storage of the standard with its `kl`, `ku` or `k` stated in the call,
and a packed operand is a rank-1 tensor of `n * (n + 1) / 2` elements. A Hermitian routine does
not read the imaginary part of the diagonal and leaves an updated diagonal real.

**Operands are described once.** `operands/matrix_operand.hpp`, `operands/vector_operand.hpp`,
`blas/band_operand.hpp`, `blas/packed_operand.hpp` and `blas/scalar_operand.hpp` each turn a tensor into the extents, leading
dimension and increment a kernel takes, and throw `std::invalid_argument` on a view the kernel
cannot read (a non-contiguous matrix, a zero or negative increment, mixed layouts, a batch count
that does not match). `operands/blas_metadata.hpp` writes the scalars and flags into the parameter slots of the task's
`TaskDescription`.

**Kernel shapes.** These are the shapes as written; none is a claim about speed.

- *Level 1.* The maps (`axpy`, `scal`, `copy`, `swap`, `rot`, `rotm`) launch through
  `Launch::indexed`. `asum`, `nrm2`, `dot`, `dotu`, `dotc` and `iamax` fold through
  `Reduce::reduce_indexed` with a leased scratch for the partials. `rotg` and `rotmg` are scalar
  routines.
- *GEMV, SYMV, HEMV, TRMV, GBMV and the band and packed symmetric products.*
  `row_reduction.hpp` launches one work-group per row for long rows and
  one work-item per row otherwise, each folding in a fixed order. TRMV takes a snapshot of `x`
  first, then computes one row product per element.
- *TRSM and TRSV.* `trsm.hpp` is a blocked substitution: each diagonal block is solved for every
  right-hand-side vector in parallel, then subtracted from the unknowns still to be solved in a
  second parallel launch. `triangular_system.hpp` decodes every side and flag into one
  substitution, and `triangular_plan.hpp` picks the block size from the device. TRSV calls the
  same kernel with one right-hand side. The solved elements are carried in the accumulation type
  between steps through `RunningSolution`, in a leased workspace when that type is wider than the
  element.
- *TBMV and TPMV.* `triangular_product.hpp` snapshots x, then computes one row product per
  work-item, as TRMV does.
- *TBSV and TPSV.* `triangular_solve.hpp` is the blocked substitution of TRSM for one vector:
  one work-item solves a diagonal block, then one work-item per row subtracts it from the rows
  it reaches. `plan_vector_solve` keeps a band too narrow to repay the launches in one block.
- *GEMM.* `Kernels::SYCL::gemm` chooses among three shapes from the `Device::Profile` and logs
  the choice at `SB_LOG_INFO`. `gemm_local_tiled` stages tiles of A and B in work-group local
  memory, for a device with dedicated local memory. `gemm_no_local` uses no local accessor and no
  barrier. The packed product (`packed_product.hpp`, `packed_panel.hpp`,
  `packed_micro_kernel.hpp`) packs panels of both operands into leased scratch and runs a
  register-block micro-kernel over them, one work-item per worker; `selects_packed_product`
  chooses it on a CPU device when the output spans at least four register blocks on each axis and
  the operands hold at least the device's cache in bytes. Without the scratch, or when the plan is
  not selected, the kernel falls back to one of the other two.
- *SYMM and HEMM* run the GEMM kernels through `symmetric_access.hpp`, which reads the mirrored
  triangle. *SYRK and HERK* use the packed product with a triangular store where the plan selects
  it and a local-memory tile otherwise. *SYR2K and HER2K* use a local-memory tile. The tiled SYRK
  and SYR2K throw `UnsupportedExecution` at record time on a device that cannot host the tile.

`beta` exactly zero skips reading the output in every routine that takes it, and the operand set
declares the output as written only in that case. SYRK, HERK, SYR2K and HER2K keep one triangle
of the output and declare it as read and written whatever `beta` is.

## Sparse BLAS

The sparse operands are plain structs of tensors, so they are views like any other:

- `SparseMatrix` (`sparse_storage/sparse_matrix.hpp`) is CSR as three rank-1 tensors, `row_offsets`
  (`rows + 1` elements), `column_indices` and `values`, with `rows`, `cols` and an
  `index_base`. The index tensors are `INT32` or `INT64` and share a dtype.
- `SparseVector` (`sparse_storage/sparse_vector.hpp`) is `indices` and `values` with a `size` and an
  `index_base`, no position stored twice.
- `Core::IndexBase` (`core/sparse_types.hpp`) is `Zero` or `One`.

`SparseBuilder` (`sparse_storage/sparse_builder.hpp`) builds them from host data on one device of a
`TensorFactory`. `matrix_from_triplets` takes COO arrays, adds the values of repeated
`(row, column)` pairs and orders each row by ascending column. `matrix_from_dense` keeps every
element that is not zero. `vector_from_entries` takes positions and values. The builder throws
`std::runtime_error`, as the factory does.

```cpp
const SparseBuilder builder(engine->tensor_factory());
return builder.matrix_from_triplets(
    host.rows, host.cols, stored_indices<Index>(host.row_indices, base),
    stored_indices<Index>(host.column_indices, base), host.values, base);
// ...
const SparseMatrix A = build(host, base);
const Tensor tx = vector_of(x);
Tensor ty = vector_of(y);
SushiBLAS::SparseOps(*engine).spmv(A, tx, ty, trans, BLASScalar(alpha),
                                   BLASScalar(beta));
```

Source: `modules/operations/sparse/tests/test_sparse_spmv.cpp`

The first statement pair is the body of the fixture's `build`, which the later lines call.
`stored_indices` gives the host indices in the index type and base of the case.

`SparseOps` records thirteen routines:

| Kind | Routines |
| --- | --- |
| Sparse vector with dense vector | `axpyi`, `doti` (real only), `dotui`, `dotci`, `gthr`, `gthrz`, `sctr`, `roti` |
| CSR matrix with dense vector | `spmv`, `spsymv`, `sphemv`, `sptrsv` |
| CSR matrix with dense matrix | `spmm` |

The record-time checks (`sparse/index_check.hpp`, `sparse/sparse_matrix_operand.hpp`,
`sparse/sparse_vector_operand.hpp`) cover the index base, the rank, length, stride and
storage of each array, the index dtype and whether it can hold the largest index. They do not
read the index values: a kernel takes the structure on trust, and
`modules/kernels/sparse_kernels/include/SushiBLAS/sparse_kernels/reference/structure.hpp` holds the host checks the reference kernels assert in a
build without `NDEBUG`.

**The scattering scheme.** A product with `NoTrans` gives each output row to one work-item,
which adds that row's terms in stored order. A transposed product has no such owner: output `j`
is column `j`, and every row scatters into the output. `modules/kernels/sparse_kernels/include/SushiBLAS/sparse_kernels/sycl/row_blocks.hpp` makes
that safe without atomic adds, in two stages. The rows are cut into consecutive blocks that hold
equal shares of the stored entries, found by a binary search of the row offsets, and one
work-item per block zero-fills a private slice of leased scratch and adds its rows' terms into
it. A second launch then gives each output to one work-item, which adds that output's partial
from every block in ascending block order and stores the result with `alpha` and `beta`. No
element has two writers in either stage, and each output still receives its terms in ascending
row order. `scatter_blocks` picks the block count as the smallest of the work-items the device
keeps resident, two partial sums per stored entry, and 256. `spsymv` and `sphemv` use the same
scheme, since each stored entry also contributes to its mirror. A transposed `spmm` gives each
column of the result to one work-item, which scatters that column serially. `sptrsv` is one
work-item.

## Spatial

`SpatialOps` moves numbers around an NHWC grid: gathers (`im2col`, `col2im`,
`upsample_nearest`, `concat_channels`, `slice_channels`, `layout_convert`), windowed reductions
(`pool_max`, `pool_avg`), broadcasts (`channel_affine`, `broadcast_spatial`), the backward form
of each that has one, and the windowed product. The family serves `Domain::Real`, except
`concat_channels` and `slice_channels`, which move elements and serve every dtype. Parameter
ownership, training mode and differentiation belong to the layer above.

`Spatial::ConvGeometry` carries the kernel size, stride, dilation and the leading padding.
`Spatial::window_span_for` (`spatial/window.hpp`) resolves the operands' strides into a
`WindowSpan` of element counts on the host, so a kernel carries no layout branch.

**The windowed product.** `window_dot(in, weights, out, g)` sums each channel's window weighted
by that channel's taps: `in` is `[N, H, W, C]`, `weights` is `[kernel_h, kernel_w, C]`, and a tap
outside the input adds nothing. It is the per-channel (depthwise) product, computed without
materialising a column matrix. `window_dot_backward` is the gradient with respect to the input
and `window_dot_weights_backward` the gradient with respect to the taps.

```cpp
Spatial::ConvGeometry g;
g.kernel_h = kernel_h;
g.kernel_w = kernel_w;
g.stride_h = stride_h;
g.stride_w = stride_w;
g.dilation_h = dilation_h;
g.dilation_w = dilation_w;
g.pad_top = pad_top;
g.pad_left = pad_left;
// ...
Tensor out = sentinel_tensor<T>(engine, c.output_dims());
SushiBLAS::SpatialOps(engine).window_dot(in, weights, out, c.geometry);
```

Source: `modules/operations/spatial/tests/test_spatial_window_dot.cpp`

The eight assignments are every field of `ConvGeometry`. `c.geometry` is one built that way, and
`sentinel_tensor` is a helper of the test that allocates the output.

**Every backward is a gather.** `col2im` written as "each column entry adds into its pixel" needs
a floating-point atomic add wherever windows overlap, and the order of those additions varies
between runs. Each output element instead finds the inputs that contributed to it and sums them in
a fixed order. `pool_max_backward` adds a test on the recorded index to the same walk. A HALF sum
in these kernels is formed in FLOAT32 and rounded once, and HALF storage on a device without the
`fp16` aspect is refused at record time.

## Reductions

`ReductionOps` folds a contiguous tensor into a one-element result tensor, or along an axis into
a vector. The value reductions serve HALF, FLOAT32 and FLOAT64.

| Operation | Result | Empty input | NaN |
| --- | --- | --- | --- |
| `sum`, `asum`, `nrm2` | Same dtype | Writes 0 | IEEE arithmetic: a NaN term gives NaN |
| `prod` | Same dtype | Writes 1 | IEEE arithmetic: a NaN factor gives NaN |
| `count_nonfinite` | `INT64` | Writes 0 | Counts every NaN and every infinity |
| `sum_along` | Rank-1, same dtype | Writes 0 to every segment | IEEE arithmetic, per segment |
| `mean`, `var`, `std`, `logsumexp` | Same dtype | Throws `std::invalid_argument` | IEEE arithmetic: a NaN term gives NaN |
| `max`, `min` | Same dtype | Throws `std::invalid_argument` | Any NaN element makes the result NaN |
| `max_along`, `min_along` | Rank-1, same dtype | Throws `std::invalid_argument` | A NaN makes its segment's result NaN |
| `mean_spatial` | `[N, C]` from `[N, H, W, C]` | Throws `std::invalid_argument` | IEEE arithmetic, per segment |
| `argmax`, `argmin` | `INT64` index | Throws `std::invalid_argument` | A total order, stated below |

The zero-element rule is decided at record time, before either path runs, so both modes answer
the same way. `var` and `std` are the population forms, dividing by `n`.

`argmax` and `argmin` return the lowest index of the winning value under the total order of
`Reduce::RunningArgMax`: a NaN at index 0 wins, a NaN at any other index loses to every element
that is not a NaN, and ties go to the lower index. That is the answer of a serial scan that starts
from element 0 and replaces its best only on a strictly greater value.

**Shapes.** A whole-tensor fold is `Reduce::reduce`, in one of two shapes its `Plan` picks. The
two-stage shape is a grid-strided pass per work-item, a hand-written halving tree per
work-group, then one task that adds a fixed number of partials in ascending order; the partials
live in a scratch leased at record time. The single-pass shape is one kernel that folds in index
order and stages no partials. `sum`, `mean` and `asum`
use `CompensatedSum` (Neumaier), `nrm2` uses `ScaledSumOfSquares` so that squaring does not
overflow or underflow, and a HALF input accumulates in FLOAT32. `var`, `std` and `logsumexp` are
two fixed-order stages through `CentredReductionDispatch`: a centre (the mean, or the maximum),
then a compensated sum around it. An axis reduction is `Reduce::reduce_axis` over an `AxisSpan`,
which names no axis: `axis_span_for` resolves the axis, the layout and the strides on the host.
`mean_spatial` is the one shape with a pair of segment axes. An axis fold stages no partials and
leases no scratch.

Each executor takes a dispatcher with two members, the SYCL fold and `reference`, the serial host
fold. `execute_folding_reduction<Accum>` builds one from a transform and a finalize functor, and
is the seam a consumer uses for a fold of its own.

`LogicOps::all` and `any` are reductions too, and the one place a `sycl::reduction` remains: a
minimum and a maximum over ones and zeros, whose value no combination order can change. On an
empty tensor `all` writes 1 and `any` writes 0.

## Random

`modules/kernels/random_kernels/include/SushiBLAS/random_kernels/sycl/philox.hpp` is a Philox4x32-10 counter-based generator. Every distribution
is a pure function of `(seed, offset, block index)`, so a fill is reproducible however the work
was scheduled. `RandomOps` has `constant`, `uniform`, `normal`, `log_normal`, `exponential`,
`poisson`, `discrete_uniform`, `truncated_normal`, `bernoulli`, the Xavier and He initialisers,
`orthogonal` and `shuffle`. A distribution that draws values refuses HALF and the integer dtypes
at record time; `constant` and `shuffle` only move values. `poisson` draws in double precision
for every dtype and throws `UnsupportedExecution` on a device without `fp64`, under every mode.

The words Philox produces are the same on the host and on the device. A distribution that goes
on through `log`, `sqrt`, `cos`, `exp` or `erfc` uses the math library of whichever side runs it,
so the two modes are not expected to agree to the bit there.

## `convert`

`ElementwiseOps::convert(src, dst)` is the one operation whose operands may differ in dtype. It
serves the nine pairs of HALF, FLOAT32 and FLOAT64. A source or a destination of another dtype
throws at record time and names the operand and its dtype, and so does a destination that
overlaps the source. Converting a dtype to itself copies.

# Regression tests

Each test here pins one defect that was found and fixed, so that a later change cannot bring it
back unnoticed. The tests are fast and deterministic. The suites carry the `Regression_` prefix,
which `tests/CMakeLists.txt` turns into the `regression` CTest label.

| File | What it pins |
| --- | --- |
| `test_io.cpp` | IO reports a file it cannot read whole through its status and reaches device storage by copy |
| `test_logger.cpp` | The logger writes every line before an assertion aborts or a shutdown returns |
| `test_regression.cpp` | One sentinel per fixed defect in the BLAS, RNG, tensor and reduction paths |
| `test_spatial_determinism.cpp` | The spatial backward kernels give the same bits on every run |
| `test_task_dependencies.cpp` | One recorded task of each recording seam runs and computes its result |
| `test_operand_lifetime.cpp` | A recorded task keeps its operands and its scratch allocated until it has run |
| `test_sparse_recording.cpp` | A recorded sparse task computes its result, keeps its operands and is ordered behind the task that wrote them |
| `test_blas_symmetric_bits.cpp` | The unconjugated reference kernels equal a plain host loop bit for bit |

None of the four files added on 2026-10-07 had been built or run when this page was written.

## test_task_dependencies.cpp

`ARecordedTaskComputesItsResult` records one operation of each recording seam on a `GraphProbe`,
reads from the plan that it is one task with no dependency, runs it and compares the output with
the host formula.

A task body receives the events of the tasks it depends on and must order its own device work
behind them. No test checks that at run time: the runtime's plan view holds no body, so a test
cannot hand one a dependency that has not finished. `tools/layering/check_layering.py` checks the
rule in the source, under `rule_task_dependencies`.

## test_operand_lifetime.cpp

A task holds raw pointers into its operands, and the recorder retains the storage behind them
(`OperandSet`, `Internal::StorageKeepAlive`). The fixture counts the runtime's live USM
allocations around the record and execute calls. `ATaskKeepsItsInputAllocatedUntilItHasRun`
destroys an input tensor after the record call and checks its allocation lives until the task
ran. `AnEpilogueOperandIsKept` does the same for a tensor declared through
`Dispatch::gemm_epilogue`. `AScratchReturnsToThePoolWithItsTask` runs a reduction twice and
checks that its scratch block is idle in the pool after the first run and lent again in the
second.

## test_sparse_recording.cpp

Three properties of a sparse task, whose operand is three tensors and not one:
`ARecordedTaskComputesItsResult`, `AMatrixDestroyedBeforeExecuteIsKept`, and
`AProductIsOrderedBehindTheTaskThatWroteItsValues`, which asserts on the recorded graph that a
product depends on the task that wrote the matrix's `values` and on no task it reads nothing
from. `TheSameCallsReplayIntoASecondGraph`
records the same calls through a second recorder.

## test_blas_symmetric_bits.cpp

The conjugation policy turned each symmetric kernel into a template that also serves its
Hermitian sibling. Under `Unconjugated` the kernel must do what it did before: the same
operations in the same order. Each test (`dot`, `ger`, `symv`, `syr`, `syr2`, `syrk`, `trsm`)
runs the reference kernel beside a plain host loop written out in the test and compares the two
bit for bit.

## test_regression.cpp

### SyrkAcceptsHalfDtype

SYRK's oneMKL-backed implementation threw for `sycl::half`, because oneMKL has no half-precision
SYRK. The SB-2 port to the tiling kernel lifted that restriction. The typed suite in
`modules/operations/blas/tests/test_syrk.cpp` covers float, double and complex only, so nothing else
would catch a return of the restriction.

### GemmColumnMajorOperandsUseColumnStrides

GEMM and SYRK read a `ColumnMajor` A and B with row-major indexing: `a[row*lda + col]` where the
standard says `a[row + col*lda]`. That had two consequences.

- `layout` was a no-op for both operands, so a column-major GEMM computed the wrong product.
- The driver passes the column-major leading dimension (`lda = m` for an m x k A), so the wrong
  index runs past the end of the buffer whenever m > k. The largest index touched is
  `(m-1)*m + k-1` against an allocation of `m*k`.

The defect survived because the reference kernel carried the same one, so every differential
test of SYCL against the reference agreed. This test cannot agree by accident: it uses
hand-computed numbers, runs both execution modes, and picks a shape with m > k so that an
out-of-bounds read would be back in range of the sanitizers.

The numbers:

```
A (3x2) = [[1,2],[3,4],[5,6]]   column-major [1,3,5,2,4,6],   lda = 3
B (2x2) = [[1,0],[1,1]]         column-major [1,1,0,1],       ldb = 2
A*B     = [[3,2],[7,4],[11,6]]  column-major [3,7,11,2,4,6],  ldc = 3
```

### SetSeedRewindsTheRNGStream

Setting the seed, now `RandomOps::set_seed`, set the seed and left the RNG offset where earlier
draws had pushed it. The
offset only grows, and every RNG kernel mixes (seed, offset) into Philox, so re-seeding an engine
that had already drawn did not reproduce its stream. Two generators seeded alike on one engine
produced different data.

The test compares bytes, not values within a tolerance. A reproduced stream is the same bits or
it is not reproduced. A third draw with a different seed checks that the reset did not collapse
every seed onto one stream.

### Fp64Level1ScalarsAreNotNarrowedToFloat

`axpy`, `scal` and `gemv` declared their scalars `float` while `gemm` declared `double`. An fp64
computation received a float-rounded scalar: about 1e-7 relative error and no diagnostic.

The test uses alpha = 1 + 2^-30. A double holds that value exactly and a float rounds it to
exactly 1.0, so the right answer and the narrowed answer are different numbers and not two close
ones. `scal` is checked in the same test because it had the same defect in the other Level 1
entry point.

### BatchedGemmAdvancesEveryBatchSlice

Tensor's column-major constructor gave the batch axis of a rank-3 tensor stride 1. The GEMM
driver read `str_a = A.strides[rank-3]`, which came back as 1 where it should be m*k, and every
batch element read the same slice.

The rule the test pins: the batch axis of a rank-3 tensor is outermost and slowest-varying in
either layout. `layout` orders the memory inside a slice only.

The test uses three batch elements and varies all three operands per batch, so an operand that
aliased slice 0 changes the answer:

```
A_i = (i+1) * I(2)     B_i = {{1,2},{3,4}} + i     C_i = A_i * B_i
```

### AccumulatingOpsDeclareTheirAccumulatorRead

`ger`, and `gemv` with beta != 0, declared their accumulator write-only while reading it. The
order still came out right because the tracker inserts a write-after-write edge anyway, so the
defect was latent and no behavioural test could see it. The test reads the recorded graph
instead: it records through a `GraphProbe` (`tests/common/graph_probe.hpp`), whose
`recorded_tasks()` returns what the operation declared.

`gemv` with beta == 0 overwrites y completely and must stay a pure write. A read it does not
perform would cost the scheduler a false edge.

### ReductionsOrderAgainstViewsOfOneAllocation

`Dispatch::execute_reduction` described its operands to the dependency tracker as bare `data()`
pointers where every other op family passes `Tensor::region()`. This family was missed in the
conversion. For a tensor at offset zero the two are the same, so nothing caught it.

For a view inside a shared arena they differ. A `BufferRegion` built from a bare pointer keys
on that pointer as its allocation base. A region from `Tensor::region()` keys on the allocation
base and carries the view's byte interval. Two views of one arena got two different bases,
`BufferRegion::overlaps` returned false, and the tracker emitted no edge. The reduction was
free to run before the op that produced its input.

This happened in practice. SushiAI's first arena-packed training run reported a loss of -57 where
ln 6 was expected: the `max` reduction raced the `bias_add` feeding it, and the BLAS `scal` beat
the `sum` reduction to the loss.

The test asserts on the recorded graph and not on the answer. Without the edge the two tasks may
still run in the right order by luck, and a test that passes by luck guards nothing. The
`dependency_count` of the second task decides it. The two views are disjoint and the consumer's operand is
not at offset zero, which is the shape a memory planner produces and the one where a bare-pointer
region and a real region disagree. The same pair of ops is then run once more and its sum
compared, for a reader who wants to see the behaviour as well.

### scalar_of

The nrm2 sentinels check values near 1e38 and 1e-40. The absolute tolerance `verify_tensor`
applies means nothing there: 1e-4 of absolute slack accepts any answer against 2e38 and rejects
every answer at 5e-40. The sentinels compare relative error, so they read the raw number.

### Nrm2DoesNotOverflowNearTheFloatMaximum

nrm2 computed the naive two-norm: square, sum, square root. For any fp32 element past about
1.8e19 the square overflows to infinity long before the norm does. The compensated-fold work made
the summation accurate and left the squaring unsafe.

The input is an ordinary one. Gradient-norm clipping calls nrm2, and a gradient that has already
exploded is the vector whose squares overflow. The mechanism meant to rescue the run returned
`inf` and ended it.

The test input is {1.2e38, 1.6e38}. Its norm is 2e38, inside fp32's range, while 1.2e38 squared
is 1.44e76. Both entry points are checked: `ReductionOps::nrm2`, the deterministic folding
reduction, and `Level1::nrm2`, a Level 1 kernel pair. They are separate implementations
and both had the defect.

### Nrm2DoesNotUnderflowOnDenormalInput

This is the mirror of the overflow, and the reason the scale has to move down as well as up. An
fp32 element below about 1e-22 squares to zero, so the naive form returned exactly 0 for a vector
of denormals whose true norm is an ordinary denormal.

The input is {3e-40, 4e-40}. Both are denormal (fp32's smallest normal is about 1.18e-38), both
have denormal resolution to spare, and the true norm of 5e-40 is representable. The tolerance is
1% on purpose: a denormal has few significand bits left and the two entry points round
differently. Anything within 1% is the right answer and not the zero the naive form produced.

### Nrm2UsesTheModulusOfAComplexElement

Level 1 nrm2 took its per-element magnitude from `abs1`, which is BLAS's |Re| + |Im|. That is the
right magnitude for xASUM and IxAMAX and the wrong one for a 2-norm, which needs Re^2 + Im^2. The
cross term made every complex norm too large, by sqrt(2) for a vector of (1 + i). The test uses
3 + 4i: the modulus is 5 and `abs1` would give 7.

## test_spatial_determinism.cpp

This file is the bit-reproducibility gate of the spatial domain.

Every backward in the spatial domain is written as a gather, so that each output element sums its
own contributions in a fixed order (spec CV-6). The failure it guards against is silent. The
floating-point add of `sycl::atomic_ref` is correct to any tolerance a functional test would use
and not reproducible in the last bits. A tolerance-based test passes while the library's
bit-reproducibility contract, inherited from the PUBLIC link against
`sushiruntime_deterministic_fp`, is gone.

Nothing in the unit tier catches that, so these tests live in the regression tier and compare raw
bit patterns. `EXPECT_FLOAT_EQ` would accept the very difference they look for.

Every case uses overlapping windows. A non-overlapping geometry has one contribution per output
element and would pass with an atomic add in place.

### fill_order_sensitive

A ramp of similar magnitudes can be reordered without changing one bit of the total, which would
let every test in the file pass without proving anything. The helper alternates a large magnitude
with a small one, which makes the sum non-associative in binary32: `1e8 + 1.0` rounds back to
`1e8`, so whether the `1.0` is added before or after the `-1e8` decides whether it survives.

### Geometry of each case

- `Col2ImIsBitReproducible`: 3x3, stride 1, padded by one. Every interior pixel gathers nine
  terms of alternating magnitude.
- `MaxPoolBackwardIsBitReproducible`: 2x2, stride 1, the pool of YOLOv3-tiny. A pixel selected by
  several overlapping windows accumulates all of their gradients, and that is the only place this
  operation adds anything. The input has one dominant pixel per channel plane so that every
  window in reach selects it. A generic input would give most pixels a single contributor.
- `AvgPoolBackwardIsBitReproducible`: 3x3, stride 1. Every interior pixel gathers nine shares.
- `UpsampleBackwardIsBitReproducible`: scale 3 in both axes. Every input pixel gathers nine terms.
- `MeanSpatialIsBitReproducible`: few segments over a long extent (2 x 256 x 256 x 2). The
  reduction takes the cooperative work-group-per-segment shape and folds through the hand-written
  halving tree, not through one item's serial loop. That is the shape whose combination order a
  `sycl::reduction` would have left free.

# Tests of blas

The tests of the `blas` module, built into `sushiblas_blas_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

Each `benchmark_*.cpp` file is a program of its own, built when `SB_BUILD_BENCHMARKS` is on;
the benchmark section of `tests/README.md` describes them.

| File | Tests |
| --- | --- |
| `benchmark_band_packed.cpp` | Times the band and packed Level 2 routines beside their dense siblings, in both layouts |
| `benchmark_gemm.cpp` | Times GEMM in both layouts, FLOAT32 and FLOAT64, and prints GFLOP/s per size |
| `benchmark_gemm_blocking.cpp` | Times one square GEMM kernel call under a cache blocking given on the command line |
| `benchmark_level1.cpp` | Times nine level 1 routines, axpy to iamax, and prints GB/s per vector length |
| `benchmark_triangular.cpp` | Times the triangular and symmetric BLAS routines in both layouts, in GFLOP/s per size |
| `test_band_operand.cpp` | Tests the record-time checks on a band operand and on a triangle band operand |
| `test_blas/test_asum.cpp` | Tests BLASOps::asum against fixed expected values |
| `test_blas/test_axpy.cpp` | Tests BLASOps::axpy against fixed expected values |
| `test_blas/test_blas_band.cpp` | Tests the band routines against hand-computed results, their dense siblings and each other's kernels |
| `test_blas/test_blas_complex.cpp` | Tests the conjugating BLAS routines, CONJ_TRANS and their forwards on a real tensor |
| `test_blas/test_blas_half_accumulation.cpp` | Tests that HALF GEMM, SYRK, GEMV, DOT, ASUM and NRM2 accumulate in FLOAT32 on both kernels, and that HALF TRSM and TRSV carry their solution in FLOAT32 |
| `test_blas/test_blas_packed.cpp` | Tests the packed routines against hand-computed results, their dense siblings and each other's kernels |
| `test_blas/test_blas_structured_validation.cpp` | Tests what the band, packed and rotation routines refuse at record time, and the forwards of the Hermitian routines on a real tensor |
| `test_blas/test_blas_symmetric_bits.cpp` | Tests that the unconjugated reference kernels equal a plain host loop bit for bit |
| `test_blas/test_blas_validation.cpp` | Tests the record-time operand checks of the BLAS operations and of tensor creation |
| `test_blas/test_copy.cpp` | Tests BLASOps::copy against fixed expected values |
| `test_blas/test_dot.cpp` | Tests BLASOps::dot against fixed expected values |
| `test_blas/test_gemm.cpp` | Tests gemm against the reference kernel, hand-computed results and its epilogue seam |
| `test_blas/test_gemv.cpp` | Tests BLASOps::gemv against fixed expected values |
| `test_blas/test_ger.cpp` | Tests BLASOps::ger against fixed expected values |
| `test_blas/test_iamax.cpp` | Tests that BLASOps::iamax returns the zero-based index of the largest magnitude as INT64 |
| `test_blas/test_level1_parallel_reductions.cpp` | Tests the Level 1 reductions on the deterministic machinery and HALF accumulation |
| `test_blas/test_level1_reference.cpp` | Tests the BLAS Level 1 reference kernels against hand-computed results |
| `test_blas/test_level2_reference.cpp` | Tests the BLAS Level 2 reference kernels against hand-computed results |
| `test_blas/test_level2_sycl_parity.cpp` | Tests that the GEMV and SYMV kernels stay within their rounding bound of a double-precision truth, and that the SYCL kernels repeat to the bit |
| `test_blas/test_nrm2.cpp` | Tests BLASOps::nrm2 against fixed expected values |
| `test_blas/test_packed_symm.cpp` | Tests SYMM and HEMM on the packed product against the reference kernel |
| `test_blas/test_packed_syrk.cpp` | Tests SYRK and HERK on the packed product against the reference kernel |
| `test_blas/test_rot.cpp` | Tests BLASOps::rot against fixed expected values |
| `test_blas/test_rotg.cpp` | Tests rotg against hand-computed rotations, across both kernels and inside a graph |
| `test_blas/test_rotm.cpp` | Tests rotm on each flag, on strided and empty vectors, across kernels and after rotmg |
| `test_blas/test_rotmg.cpp` | Tests rotmg on its four outcomes, on both rescalings and across both kernels |
| `test_blas/test_scal.cpp` | Tests BLASOps::scal against fixed expected values |
| `test_blas/test_swap.cpp` | Tests BLASOps::swap against fixed expected values |
| `test_blas/test_symm.cpp` | Tests symm and hemm against hand-computed products, gemm on the full matrix and refusals |
| `test_blas/test_symv.cpp` | Tests BLASOps::symv against fixed expected values |
| `test_blas/test_syr.cpp` | Tests BLASOps::syr against fixed expected values |
| `test_blas/test_syr2.cpp` | Tests BLASOps::syr2 against fixed expected values |
| `test_blas/test_syr2k.cpp` | Tests syr2k and her2k against hand-computed updates, a host integer loop, a truth and refusals |
| `test_blas/test_syrk.cpp` | Tests BLASOps::syrk against the reference kernel over dtypes, layouts and triangles |
| `test_blas/test_syrk_trsm_parity.cpp` | Tests SYRK and TRSM SYCL kernels against the reference ones at multi-tile sizes |
| `test_blas/test_triangular_parallel.cpp` | Tests the blocked TRSV, TBSV, TPSV and TRSM kernels and the parallel TRMV kernel against exact integer systems, a double-precision truth and each other |
| `test_blas/test_trmv.cpp` | Tests BLASOps::trmv against fixed expected values |
| `test_blas/test_trsm.cpp` | Tests BLASOps::trsm in row-major and column-major layouts |
| `test_blas/test_trsm_reference.cpp` | Tests the TRSM reference kernel against hand-computed solutions |
| `test_blas/test_trsv.cpp` | Tests BLASOps::trsv on a 3x3 lower-triangular system |
| `test_blas_task.cpp` | Tests that a BLAS task records for a real or complex dtype and refuses an integer one |
| `test_dispatch/test_channel_bias_epilogue.cpp` | Tests ChannelBiasEpilogue's stride pair through Dispatch::execute_gemm |
| `test_dispatch/test_packed_product.cpp` | Tests the packed product kernel against the reference GEMM under injected blockings |
| `test_dispatch/test_reference_mode_guard.cpp` | Tests that ForceReference refuses device-only USM and runs on host-reachable USM |
| `test_level1.cpp` | Tests that the Level 1 facet of BLASOps records copy, scal and swap with exact results |
| `test_level2.cpp` | Tests that the Level 2 facet of BLASOps records gemv and ger with exact results |
| `test_level3.cpp` | Tests that the Level 3 facet of BLASOps records gemm with exact results |
| `test_local_tile.cpp` | Tests the local tile demand per dtype and the record-time check that a device hosts it |
| `test_packed_operand.cpp` | Tests the packed triangle order and the record-time checks on a packed operand |
| `test_packed_product_scratch.cpp` | Tests that the packed product leases the scratch its plan names, and none without a plan |
| `test_scalar_operand.cpp` | Tests the record-time checks on a one-element scalar operand |
| `test_sparse.cpp` | Tests that SparseBLAS is built from a recorder and only from one |

## test_blas/test_gemm.cpp

`test_accuracy` seeds both C buffers identically, or a non-zero beta would compare two
different problems. When beta is zero they are poisoned, not zeroed: BLAS says that case must
not read C at all, and a NaN that leaks through fails the comparison where a zero would hide in
a plausible number. A NaN is the strongest available witness, because 0*NaN is NaN, so any read
of C at beta == 0 propagates and `EXPECT_NEAR` rejects it. Half and the complex types all carry
a quiet NaN.

`RaggedOnEveryAxis`: every extent is prime and none divides any work-group block, so the row,
column and K tails are ragged at the same time.

`KShorterThanOneTile` spans several work-groups in both output dimensions while K is shorter
than one local-memory K-tile, so the staged tile is mostly zero padding.

`BetaZeroNeverReadsC` is ragged on purpose, so the tail path is covered too.

`Medium_GEMMKernelTest` does two things the typed differential suite cannot. The first is
hand-computed expectations; see "Expectations are worked out by hand". The layout and transpose
cases are checked against numbers worked out by hand. The second is variant coverage. GEMM
picks between its local-tiled and no-local shapes from the device profile, so on any one
machine the public API only ever reaches one of them. These tests call both directly.

`LocalTiledVariantMatchesReferenceOnWholeAndEdgeTiles` reaches every staging path of
`gemm_local_tiled`. Each extent spans at least one whole 64-wide block and an edge, and K = 40 or
48 gives two or three whole K-tiles with an edge after the first. The five cases cover A walked
rows first (ColumnMajor NoTrans, RowMajor Trans) and columns first, and B walked depth first
(ColumnMajor NoTrans) and columns first. The first case is a batch of two, so the second product
starts at a batch offset; the second case has lda = 130, which no 128-bit load divides, so its
whole tiles take the element path. FLOAT64 and HALF run the same cases where the device holds
their aspect, for their two and eight lanes per load.

`ColumnMajorMatchesHandComputedResult` stores the row-major problem column-major: A (2x3) is
`[1,4,2,5,3,6]` with lda = 2, and B (3x2) is `[1,0,1,0,1,1]` with ldb = 3. This is the case
that was wrong. The old kernels read a column-major A as if it were row-major with a
column-major leading dimension. That produced the wrong element and, because that leading
dimension is m and not k, ran off the end of the buffer whenever m > k. It survived because the
reference kernel had the identical defect.

`ColumnMajorReferenceMatchesHandComputedResult` pins the oracle itself to the same numbers.
Without it the SYCL kernel could only ever be as right as whatever the reference does.

`TransposedAMatchesHandComputedResult`: op(A) is the same `[[1,2,3],[4,5,6]]`, so A is stored
3x2 as its transpose `[[1,4],[2,5],[3,6]]`. This pins the RowMajor and transposed arm, which
must address identically to ColumnMajor and untransposed.

`EpilogueFusesBiasAndReLUIntoTheStore` is worked out as follows, with A (2x2) =
`[[1,2],[3,4]]`, B (2x3) = `[[1,0,-1],[0,1,-1]]`, alpha = 2, row_bias = `[0, 1]` and col_bias =
`[0.5, -1, 5]`:

```
A*B       = [[1, 2, -3], [3, 4, -7]]
2*(A*B)   = [[2, 4, -6], [6, 8, -14]]
+ biases  = [[2.5, 3, -1], [7.5, 8, -8]]
relu      = [[2.5, 3,  0], [7.5, 8,  0]]
```

`Medium_GEMMSeamTest` runs the epilogue through the public seam and not through the kernel. The
kernel test calls `Kernels::SYCL::gemm` directly, which no consumer can do through the Engine
API. The seam tests go through `Dispatch::execute_gemm`: recorded as a task on the graph, run
by `RecordingFixture::run()`, dispatched on the runtime dtype. That is the seam that is supported,
and a hook that only works when called from inside this repository is what these tests rule
out. C is square (3x3) on purpose, and both biases are distinct. A square C makes a transposed
`row`/`col` a wrong answer and not an out-of-range read: with a 2x3 C, swapping the coordinates
indexes a 2-entry row bias with a column index, and the failure is undefined behaviour instead
of a comparison the test can report. The expected values differ under the swap (row 0 would end
`[2, 0, 9]` and not `[2, 0, 6]`), so the coordinates are pinned individually and not merely
jointly.

`expected()` is hand-computed. `A*B = [[1,2,3], [4,5,6], [5,7,9]]`. With row_bias =
`[0, -6, 5]` and col_bias = `[1, -8, 3]`, `value + row_bias[row] + col_bias[col]` is

```
[[  2,  -6,   6],
 [ -1,  -9,   3],
 [ 11,   4,  17]]
```

and the ReLU clamps the three negatives to zero.

`ExecuteGemmAppliesTheSameEpilogueOnTheReferencePath`: the reference kernel is this operation's
specification, so it has to apply the epilogue too. Otherwise ForceReference would quietly
compute a different function from ForceSYCL, and every comparison of the two modes on a fused
GEMM would disagree.

`ExecuteGemmWithTheIdentityEpilogueIsPlainGemm` asserts what `Level3::gemm` now is: the unfused
entry point and the seam with the identity must agree element for element, or the delegation in
`modules/operations/blas/source/level3/gemm.cpp` changed behaviour.

### The epilogue's own operands

`execute_gemm` declares A, B and C. Whatever the epilogue itself touches is invisible to it.
The functor is an opaque callable, and that opacity is the feature, so an undeclared bias is
not a wrong dependency edge, it is no edge, and the graph is free to run this GEMM at the same
time as whatever writes the bias. Nothing throws and nothing is NaN; the run computes with a
value that had not arrived yet. `Dispatch::gemm_epilogue(...).reads(t)/.writes(t)` is how the
caller hands that back. Every assertion in this group is on `dependency_count`; see "Ordering is
asserted on in-degree".

`BiasColumnEpilogue` is deliberately simpler than `BiasReLUEpilogue`. These tests are about
what the task declares, not about what it computes, and a bias-only epilogue is the one the
ordering hazard was found on.

`BiasSpillEpilogue` is the shape cuBLASLt calls `GELU_AUX_BIAS`: the pre-activation the
backward pass has to differentiate at, written out on the way past so it still exists after the
fused store. No activation is applied. What is under test is whether the seam can express an
auxiliary write at all, which decides whether such an epilogue can ever be safe, not any
particular activation's arithmetic.

`ADeclaredEpilogueReadIsOrderedBehindItsProducer`: the bias is written inside the same graph,
which is the case that matters, because a parameter an optimizer rewrites within one compiled
plan looks exactly like this. A and B have no producer here, so the GEMM's in-degree counts the
bias edge and nothing else.

`AnUndeclaredEpilogueOperandGetsNoEdgeAtAll` is the control, and the reason the declared-read
test is worth writing: the same graph with the same functor, passed bare. This is what every
fused GEMM in this library did before `gemm_epilogue` existed. It is asserted and not merely
described because a fix whose test would also pass without the fix is not a test. It is also
the compatibility pin: a bare epilogue must go on producing exactly this access set, so nothing
that never declared anything changes behaviour.

`DeclaredOperandsOrderAcrossTwoViewsOfOneAllocation` is the bare-pointer hazard on this seam;
see "Views at a non-zero offset". Here `overlaps()` says false for a view and its own arena.
That is why `GemmEpilogue::reads` takes a Tensor and calls `Tensor::region()` itself, and does
not accept a region a caller built.

`ADeclaredEpilogueWriteOrdersItsReaderBehindTheGemm` is the direction that decides whether an
activation can ever be fused into a training forward pass: the epilogue spills its
pre-activation, and whatever differentiates against that spill has to be ordered behind the
GEMM that produced it. Nothing else in the task's access set mentions the aux buffer, so this
edge exists only if the write was declared.

`AnUndeclaredEpilogueWriteGetsNoEdgeAtAll` is its control, and the shape of the bug an aux
buffer would otherwise have: the epilogue really does write `aux`, the reader really does read
it, and without the declaration the two are unordered.

`DeclaringOperandsChangesNoArithmetic`: the declaration is bookkeeping on the host. The bundle
is unwrapped before the closure is built, so the functor that reaches the kernel is the same
object either way and the two spellings must agree to the bit. If they do not, the unwrapping
copied something it should not have, or failed to.

`DeclaringAnOperandWithNoStorageIsRefused`: a storage-less tensor yields a base-less region,
which the tracker orders against nothing. Accepting it would let a declaration compile, read
like a fix in review and do nothing, which is the failure mode declaring operands exists to
remove.

### Kernel variants

`LocalTiledVariantMatchesReferenceOnRaggedShape` runs the variant a GPU would take. The machine
these tests were written on has no GPU; the only reachable device is an OpenCL CPU. What the
test proves is that the kernel compiles, launches, synchronises and computes the right answer
on a device whose "local memory" is really just cache. It is not evidence about performance on
real scratchpad hardware, and it is not a substitute for running it on one.

`LocalTiledVariantHandlesColumnMajorAndTransposes` drives the same variant through the stride
pair and not the identity case, so the layout and transpose hoisting is exercised on the path
the profile will not select on that machine.

`StrideTableMatchesTheStandardIdentity`: transposing an operand and flipping its layout are the
same operation. The check is cheap to state, and it would have caught the inverted
ColumnMajor arm the moment it was written.

## test_blas/test_iamax.cpp

IAMAX answers "which element has the largest |Re| + |Im|", so its result is an index and its
result tensor is INT64. An earlier version of this file allocated two FLOAT32 elements and
hoped, with a comment reading "Wait, different sizes!", because there was no integer dtype to
allocate instead. It then asserted `EXPECT_TRUE(true)`, so nothing checked the operation's
answer. Now the result has a dtype that fits it, and the index is read back and compared.

`TiesGoToTheLowestIndex`: BLAS specifies the first of two tied magnitudes. The reference and
SYCL kernels both scan in ascending order and only replace on a strict improvement, which makes
that reproducible and not incidental.

`RejectsAResultTensorTooNarrowToHoldTheIndex` pins the defect the INT64 dtype was added for.
`iamax` writes eight bytes; a FLOAT32 result element is four. With tensors packed into an arena
the neighbouring four bytes belong to something else, and `Tensor::region()` declared a
four-byte write, so the dependency tracker did not order anything against the corruption
either. Nothing can detect that from outside, which is why the fix is a refusal and not a
repair: a result tensor that cannot hold the answer is rejected at record time, by name.

## test_blas/test_syr.cpp

With x = `[1, 2, 3]`, `x * x^T` is

```
[1 2 3]
[2 4 6]
[3 6 9]
```

and since the call passes `Core::Uplo::Lower` only the lower triangle of A is updated, so A in
row-major order is

```
[1 0 0]
[2 4 0]
[3 6 9]
```

## test_blas/test_syr2.cpp

The update is `A + alpha * x * y^T + alpha * y * x^T` on the lower triangle. Y is `[1, 0, 0]`
to make the result easy to check. `x * y^T` and `y * x^T` are

```
[1 0 0]      [1 2 3]
[2 0 0]      [0 0 0]
[3 0 0]      [0 0 0]
```

and their sum is `[2 2 3; 2 0 0; 3 0 0]`. The lower-triangle update only writes the entries
with i >= j, so the matrix the test expects is

```
[2 0 0]
[2 0 0]
[3 0 0]
```

## test_blas/test_trsm.cpp

`SimpleTRSM` solves `A * X = B` with A = `[2 0; 1 2]` and B = `[4 6; 4 7]`, row-major:

```
2*x11 = 4            => x11 = 2
2*x12 = 6            => x12 = 3
1*x11 + 2*x21 = 4    => x21 = 1
1*x12 + 2*x22 = 7    => x22 = 2
```

so X, which overwrites B, is `[2 3; 1 2]`.

`SimpleTRSMColumnMajor` solves with the upper-triangular A = `[2 1; 0 2]` and the same B:

```
2*x21 = 4            => x21 = 2
2*x22 = 7            => x22 = 3.5
2*x11 + 1*x21 = 4    => x11 = 1
2*x12 + 1*x22 = 6    => x12 = 1.25
```

so X is `[1.00 1.25; 2.00 3.50]`, which is `[1.0, 2.0, 1.25, 3.5]` in column-major memory.

## The `Medium_Level2SyclParityTest` fixture

`tests/operations/blas/test_blas/test_level2_sycl_parity.cpp` runs GEMV and SYMV in a chosen execution mode
and measures each against a `TruthSum`, as `modules/foundation/blas_test_oracle/README.md` describes under `blas_truth.hpp`.

## test_dispatch/test_channel_bias_epilogue.cpp

`Medium_ChannelBiasEpilogueTest` checks the stride pair on `Kernels::ChannelBiasEpilogue` both
ways round. A `(row_stride, col_stride)` that is only ever exercised at `(0, 1)` is a parameter
nobody proved: the column arm and a hard-coded `bias[col]` are indistinguishable from outside.
Each direction is checked against a hand-computed expectation, and C is square (3x3) so that a
transposed `row`/`col` is a wrong answer and not an out-of-range read. Everything runs through
`Dispatch::execute_gemm`, not through `Kernels::SYCL::gemm`, because the seam is what a consumer
can reach and is where the epilogue is instantiated in the consumer's own translation unit.

`bits_of` exists because the identity check is about the generated arithmetic being unchanged;
see "Comparing floats by bit pattern".

`AColumnChannelBiasMatchesAPlainRowVectorBroadcast`: at `(0, 1)` the channel is the column,
which is what the NHWC convolution GEMM `[N*H*W, C_in*R*S] x [C_in*R*S, C_out]` produces and
what the existing `GEMM_BIAS` row-vector broadcast already does.

`ARowChannelBiasBroadcastsDownTheOtherAxis`: at `(1, 0)` the channel is the row, the NCHW
orientation. The bias vector is the same as in the column test, so the two expectations differ
only in the axis it was broadcast along, and a swapped stride pair fails one of them.

`TheSameEpilogueAppliesOnTheReferencePath`: the reference kernel is this operation's
specification and invokes the epilogue on the host, so an addressing arm that only works in the
SYCL instantiation would make every ForceSYCL against ForceReference comparison of a fused
GEMM disagree. The row orientation
is the one pinned here, since nothing else in the library exercises that arm.

`AZeroBiasIsTheIdentityEpilogue` checks that the fusion hook costs nothing when it does
nothing: against the identity epilogue, bit for bit. A tolerance would pass on a kernel that had
silently changed its accumulation order, which is the regression this rules out. The operands
are chosen so every element of `A*B` is strictly positive. `x + 0.0f` is the identity on every
float except -0.0f, which it turns into +0.0f, and that one case is the only way the test could
fail for a reason that is not a defect.

`TheEpiloguesAccessSetReachesTheTask`: the bias is written by a task recorded before the GEMM,
so there is a real producer to be ordered behind. A and B have no producer, so the GEMM's
in-degree counts the bias edge alone. See "Ordering is asserted on in-degree".

`TheGemmSeesTheProducedBiasValues` is the other half: the edge exists, and the values that
cross it are the produced ones. On its own it proves nothing about ordering, because it can
pass by luck without the declared read. It is the companion to the in-degree assertion, not a
replacement for it.

## test_blas/test_blas_symmetric_bits.cpp

The conjugation policy turned each symmetric kernel into a template that also serves its
Hermitian sibling. Under `Unconjugated` the kernel must do what it did before: the same
operations in the same order. Each test (`dot`, `ger`, `symv`, `syr`, `syr2`, `syrk`, `trsm`)
runs the reference kernel beside a plain host loop written out in the test and compares the two
bit for bit.

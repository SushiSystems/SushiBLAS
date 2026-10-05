# Unit tests

One file per operation or seam, each a GoogleTest translation unit built on the fixtures in
`tests/common/`. The source comments state what a test checks. This document holds why a test
is written the way it is: the defect it pins, the number it was measured against, the quirk of
the machine it ran on. Shared reasoning comes first, then one section per file in path order.

## Shared reasoning

### Expectations are worked out by hand

Where a file says its expectations are hand-computed, they do not come from another path in
this library. A differential oracle agrees with the code under test in every case where both
are wrong the same way. That is how an inverted `COLUMN_MAJOR` addressing arm in GEMM survived
for months: the SYCL kernel and the reference kernel had the identical defect, so the
differential tests agreed.

### Comparing two kernels outside VERIFY mode

The `TheSyclKernelAgreesWithTheReferenceKernel` tests fill two buffers under
`ExecutionMode::FORCE_SYCL` and `ExecutionMode::FORCE_REFERENCE` and compare them. They do not
run under `ExecutionMode::VERIFY`. VERIFY submits the device kernel and then runs the reference
kernel over the same output buffer, so the reference answer is what survives and a disagreement
leaves no trace (see `Internal::dispatch_with_mode`). Whatever the enum's comment says, VERIFY
is not an assertion.

### Comparing floats by bit pattern

The `*_float_bits` helpers and `bits_of` copy a float into a `uint32_t` with `std::memcpy`.
`std::bit_cast` is C++20 and this tree is C++17; `std::memcpy` between two trivially copyable
objects of equal size is the same operation with the same guarantees, and every compiler this
project targets folds it to a register move. Reading a float through a `uint32_t` lvalue would
be undefined behaviour the compiler is free to optimise on. The comparison is on bits because
`==` on floats says two NaNs differ and says +0 and -0 agree, and "the same answer" means
neither. `EXPECT_FLOAT_EQ` and a zero tolerance both compare values, so they have the same
blind spot.

### Ordering is asserted on in-degree

The ordering tests assert on a node's `in_degree`, never on an output value. With no dependency
edge the two tasks may still come out in the right order by luck, so a correct number proves
nothing about ordering. `Regression_Sentinel.ReductionsOrderAgainstViewsOfOneAllocation`
explains this at length.

### Views at a non-zero offset

The ordering and storage tests put their operands partway into an arena. At offset zero a bare
`data()` pointer and a real `region()` name the same resource, so an operation that declared the
wrong one would still get its edge and the test would pass without proving anything. Partway
into an arena they diverge: `data()` is `base + offset`, and a region built from it widens to
the whole allocation of that address, which the tracker files under a resource the producer
never touched. Two views of one arena then look like two unrelated resources and the tracker
emits no edge at all. That cost SushiAI's first arena-packed training run.

## test_axis_reductions.cpp

The file covers the segmented reduction seam: `max_along` and `sum_along`, the geometry they
are planned from, and the two kernels that geometry chooses between. Every expectation is
worked out by hand. A differential oracle against `gemv(X, ones)` would have agreed with a row
sum that read the wrong stride in exactly the layouts where both read it wrongly.

Determinism for this seam is tested here and not in `test_reduction_determinism.cpp`, because
that file's helpers are built around a scalar result and this seam's answer is a vector. The
property and the means are the same: `std::memcmp`, not a tolerance.

The file holds three fixtures, which the removed separator lines used to mark: the planner
(`Unit_AxisReductionPlan`, a pure function that needs no device), the public operations over
every dtype a reduction serves (`Unit_AxisReductionsTyped`), and layout, shape, rejection and
the two kernels (`Unit_AxisReductions`).

`kRowMajor` holds small whole numbers on purpose. They are exact in HALF as well as in the
wider types, so one expectation serves every dtype and a mismatch is a real disagreement, not a
rounding difference.

```
     1  -2   3   0      row max  3    row sum   2
     5   4  -1   2      row max  5    row sum  10
    -3  -7  -2  -9      row max -2    row sum -21

  col max  5   4   3   2
  col sum  3  -5   0  -7
```

`EnoughSegmentsTakesOneItemPerSegment`: one item per segment is the cheap shape whenever there
is already enough work to go round. It needs no local memory and no barriers, and its fold
order is trivially fixed.

`FewLongSegmentsTakeAWorkGroupEach`: too few segments to fill the machine, and segments long
enough to be worth a work-group, is the only case where the tree earns its barriers.

`FewShortSegmentsStillTakeOneItemPerSegment`: a group whose segment is shorter than the group
is mostly idle lanes paying log2(local) pairs of barriers, so the tree is declined below one
full pass.

`EmptySpansTakeTheCheapShape`: a degenerate span must not select a shape that would launch
`segments * local` work-items over nothing, or divide by a zero extent.

`ColumnMajorOperandsFoldTheSameAxes`: the axis is a question about the matrix, not about memory
order, so the same logical matrix must reduce to the same vectors under either layout. The
kernel never learns which layout it is on. `Tensor::strides` already carries it, and
`axis_span_for` turns the pair into a stride pair.

`AViewAtANonZeroOffsetReducesItsOwnElements`: every address the kernel forms is
`data() + s * segment_stride + k * element_stride`, and `data()` already carries the offset, so
this checks that the offset was not dropped on the way through the seam.

`AxisReductionsOrderAgainstViewsOfOneAllocation` is the bare-pointer hazard described under
"Views at a non-zero offset", on this seam.
`Regression_Sentinel.ReductionsOrderAgainstViewsOfOneAllocation` pins it for the whole-tensor
seam.

`NonRealDtypesAreRefusedByName`: a fold needs an ordering and an identity, which `std::complex`
does not have, and an integer tensor is not an arithmetic type in this library. It is the same
`Domain::REAL` rule the whole-tensor reductions carry.

`TheCooperativeTreeAgreesWithTheItemPerSegmentFold`: `plan_axis` picks one shape per input, so
on any single machine the public operations only ever reach one of the two kernels for a given
size. The test drives both directly, over one input, at a size the planner would never send to
the cooperative kernel. That is the only way the work-group tree gets exercised on a machine
with no GPU. The extent is deliberately not a multiple of any work-group width, so the tree's
trailing lanes fold fewer elements than the leading ones, and the lanes past the extent fold
none and contribute the identity. The two results are not bit-equal to each other, because they
add in different orders; that is the point of having two shapes. Both must be right, and both
must be stable run to run, which `BothShapesAreBitwiseStable` asserts.

`BothShapesFindTheSamePerSegmentMaximum`: the per-segment answers differ from the whole-tensor
answer by a lot. `max` is exactly associative, so the two shapes must agree to the bit here and
not merely to a tolerance. Segment `s` peaks at `100 * s + 1`, at a position that moves with
`s`, so no shape can be right by reading a fixed lane.

`RowSumsAreCompensated`: each row is one large head followed by 4096 ones. In float, 1e8 has an
ulp of 8, so `1e8 + 1` rounds straight back to 1e8 and an uncompensated serial fold returns
exactly 1e8, short by the entire tail. 100004096 is a multiple of 8 and therefore exactly
representable, so the test asserts a value, not a tolerance.

`BothShapesAreBitwiseStable`: the guarantee SushiBLAS inherits from
`sushiruntime_deterministic_fp` is byte equality for the same binary on the same device, so the
test compares bytes. A tolerance-based check cannot tell that apart from "close enough", which
is the whole distinction under test. It runs at both shapes by driving the kernel directly, for
the same reason the cooperative-tree test does. A degenerate result would make byte equality
trivial and the case vacuous, so the input has to reduce to a number first.

`ARowWiseShiftSurvivesAGapThatDestroysAGlobalOne` is the numerical property the axis reduction
was added for: a log-sum-exp shifted by each row's own maximum survives a gap between a row and
the batch maximum that destroys the globally shifted form. Both spellings are computed from the
same logits, so the comparison does not appeal to a remembered number. Every row is three equal
values, whose log-softmax is exactly -ln(3) for any offset, which makes the expectation
hand-computable and independent of the offset under test. The second half of the test shifts by
the global maximum, which is all the library offered before this seam existed.

## test_channel_bias_epilogue.cpp

`Unit_ChannelBiasEpilogueTest` checks the stride pair on `Kernels::ChannelBiasEpilogue` both
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
SYCL instantiation would make VERIFY mode declare every fused GEMM wrong. The row orientation
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

## test_convert.cpp

The values are chosen so the assertions cannot pass without the conversion happening. A round
trip of small integers through fp16 is a true statement about almost any implementation,
including a `memcpy` that got the widths right by luck, so it appears only as a labelled sanity
case. Everything else asserts on a value that changes: an fp32 number fp16 cannot represent
must come back as the correctly rounded fp16 neighbour (not merely "close"), one past fp16's
range must come back as an infinity, and one below it must come back as zero.

The expectations are exact, and they are worked out by hand from IEEE-754 binary16. A
differential oracle against `static_cast` on the host would agree with a device kernel that
made the same mistake. The ordering cases assert on `in_degree`; see "Ordering is asserted on
in-degree".

The file is laid out in five groups: the cases the conversion has to change a value to pass,
the sanity case, what the operation refuses and whether the refusal says enough to act on, what
it declares to the dependency tracker, and the offset case.

`Fp32ToFp16RoundsToTheNearestRepresentableValue`: the assertion is on the exact correctly
rounded neighbour, and a second assertion says the result is not the input. An implementation
that widened, reinterpreted or otherwise carried the fp32 value through unchanged fails here.

`Fp32ToFp16BreaksTiesToEven`: the rule is round-to-nearest-even, not round-away and not
truncate. `1 + 2^-11` sits exactly halfway between the binary16 neighbours 1.0 (mantissa even)
and `1 + 2^-10` (mantissa odd), so only ties-to-even gives 1.0. The second value is just past
halfway and must go the other way, which stops "always round down" from passing the first.

`Fp32PastTheHalfRangeBecomesInfinity` is the behaviour a loss-scaling backoff is built on: a
gradient scaled past fp16's range does not wrap, saturate or clamp. It becomes an infinity,
which `count_nonfinite` can then find.

`Fp32BelowTheHalfRangeUnderflowsToZeroButSubnormalsSurvive`: 1e-8 is below half of the
smallest subnormal and must flush to zero. 1e-7 is not, and must land on a subnormal and not be
flushed with it.

`NonFiniteValuesSurviveTheConversion`: this pairing makes `reductions().count_nonfinite`
meaningful on a converted tensor. If a cast quietly turned a NaN into a number, screening the
fp16 copy would say the fp32 original was clean.

`ARoundTripThroughHalfKeepsOnlyWhatHalfCouldHold` is the complement of the sanity case: it says
what the round trip costs when the value is one fp16 cannot hold.

`EveryRealDTypePairConvertsExactlyRepresentableValues` is the sanity case. Exactly representable
values survive every conversion, which is necessary and nowhere near sufficient: it would pass
for an implementation that had the element widths right and the rounding wrong. Its whole job
is to cover all nine arms of the dtype cross product in one place.

`ConvertingADTypeToItselfIsAPlainCopy`: this is the decision documented on
`ElementwiseOps::convert`. A mixed-precision policy holds its compute dtype as runtime data,
and when it equals the master dtype the operation has to degenerate into a copy and not throw.
The test writes 2^40 and not a decimal like 1e30, because the expectation is an exact equality
and a decimal that fp32 cannot hold would be rounded by `write_exact` before `convert` ever saw
it, which would make this a test of the test.

`ANonContiguousOperandIsRefused`: a slice along an inner axis keeps its parent's strides, so
its elements are not where a flat walk would look for them.

`OverlappingOperandsAreRefused`: overlapping operands are a race between work-items, not an
in-place update, because each item reads one element and writes another of a different width.

`AConvertIsOrderedBehindTheOpThatWroteItsSource`: both operands are views at a non-zero offset,
for the reason given under "Views at a non-zero offset". Verified by mutation: rewriting
`src.region()` as the bare pointer makes the assertion report 0.

`ConvertsOverDisjointViewsOfOneArenaDoNotSerialise` is the control: two converts over disjoint
views of one arena must not serialise. The two ordering tests catch an under-declared operand
(no edge where one is needed); this one catches an over-declared operand (an edge where none
is). Verified by mutation as well: declaring `src.storage->allocation_base()` in place of
`src.region()` makes the assertion report 1, while both ordering tests stay green.

## test_count_nonfinite.cpp

`count_nonfinite` is the exact non-finite screen a loss-scaling step runs before it decides
whether to apply an update. The file is careful about three things.

Infinity and NaN are tested separately. They are two different failures, and the obvious
hand-rolled predicate `x != x` catches only one of them, because an infinity compares equal to
itself. A screen that found NaNs and waved infinities through would pass a suite that only ever
fed it NaNs, and fp16 overflow produces infinities first. In the other direction, a screen
built out of a range comparison and not `isfinite` would catch the infinities and let every NaN
through, and only `ANaNIsCounted` would notice.

There is a control. A clean tensor must count zero. Without `ACleanTensorCountsZero`, an
implementation that returned a positive number unconditionally would pass every other case.

The last index has its own case. A fold whose loop bound is off by one drops exactly the final
element and is otherwise perfect, so the bad element is planted at index 0, at the middle and
at n-1 in turn. This runs at a size above `Reduce::K_CPU_TWO_STAGE_MIN_ELEMENTS`, where the
two-stage kernel with its grid-strided walk and halving tree is the one running.

`EveryBadElementIsCounted`: the answer is a count and not a flag. Several bad elements of both
kinds, spread across the tensor, must add up. This distinguishes the operation from a
saturating "any" and is the extra information the INT64 result buys.

`BothLaunchShapesGiveTheSameCount` runs the same content at a size on each side of the CPU's
crossover from the single pass to the two-stage kernel. Both shapes must give the same count,
which is what lets the threshold be a pure performance choice. The local is named `below` and
not `small` because `<rpcndr.h>`, which reaches this translation unit through the Windows
headers, defines `small` as a macro for `char`.

`AsumReportsOverflowOnAHalfTensorThisOpCallsClean` decides whether this operation needed to
exist. `asum` would nearly have served: a sum of magnitudes is non-finite whenever any element
is, and a caller could have written `isfinite(asum(g))`. The test runs both operations on one
tensor of eight finite fp16 values, and they disagree. The tensor is clean, and `asum` says it
is not, because it accumulates in the input's own dtype and 8 x 20000 leaves binary16's range
long before any element does. It is a measurement, not an assertion of intent. In a
loss-scaling loop that disagreement halves the scale on a step that never overflowed, and does
it again on the next step, for ever; the mechanism meant to rescue a mixed-precision run would
be what stalls it. The test asserts nothing about which value `asum` returns, only that it is
not finite.

`NonRealInputDTypesAreRefusedByName`: an integer element cannot be infinite or NaN, so "0"
would be true and useless. It would tell a caller who handed over the wrong tensor that the
right one was screened.

`IsOrderedBehindTheOpThatWroteItsInput` puts the input partway into an arena; see "Views at a
non-zero offset". `AnOverflowProducedInsideTheGraphIsFound` is the same thing behaviourally,
end to end.

## test_dispatch_complex.cpp

Before the dispatch seam was made public, the elementwise and logic families handled HALF,
FLOAT32 and FLOAT64 and ended their dtype switch with `default: break;`. A COMPLEX32 tensor
therefore reached a command group that enqueued no kernel at all: `add()` "succeeded", the
graph reported no error, and the output tensor was left holding whatever it held before. Every
case in this file is one of those silently dropped paths, now resolved one way or the other:
computed where the operation has a complex meaning, thrown for where it does not.

`BinaryArithmeticOnComplex32`: the imaginary cross terms of the two products are exactly what
the old no-op path could never have produced, and what a real-only kernel would drop.

The rejected operations threw nothing before; they enqueued nothing and returned. Each now
fails at record time, at the call site, before a task reaches the graph.

`OrderingComparisonsRejectComplex32`: C is not an ordered field. The dispatch used to answer
these comparisons lexicographically, which is a well-formed answer to a question nobody asked.

`RemainingRealOnlyOpsRejectComplex32`: trigonometry and `pow` do have complex extensions, but
SushiBLAS does not implement them yet. The contract is that "not implemented" is observable:
these report the gap and do not leave the tensor untouched while claiming success.

## test_dispatch_seam.cpp

This translation unit reaches `SushiBLAS::Dispatch` through installed public headers and hands
it functors this file wrote, which is the point of promoting those five headers out of `src/`.
It is the in-tree half of the proof. The out-of-tree half is `tests/package/main.cpp`, which
does the same against a `find_package()`'d install.

`float_bits` serves the determinism checks; see "Comparing floats by bit pattern".

`RealOnlyDomainRejectsComplexAtRecordTime`: the default domain is REAL, so a functor that was
only ever meant for real scalars is rejected before it can be handed a complex operand. It is
not compiled for one, and it is not silently skipped as it once was.

`FusesAFiveTensorAdamWUpdateIntoOneKernel` uses 1023 elements on purpose. That is not a
multiple of any plausible work-group size, so the grid-stride loop has to wrap, and not a
multiple of any vector width, so the scalar tail is exercised on any device where the wide path
is taken. The fused call is one task, one kernel and one pass over memory, with five tensors
advanced together. `g` is passed as an operand even though the functor only reads it, because
the seam cannot know that, so the update must leave it untouched. The comparison spells the
same arithmetic as the library's own operations: fifteen recorded tasks and fifteen kernel
launches, every one of them re-reading state the fused kernel kept in registers. That is what
`execute_zip_inplace` exists to collapse. A read-only operand comes back bit for bit: the
kernel stages every slot and writes it back, so "the functor did not touch it" has to mean "the
value that was there is the value that returns".

`FusedInPlaceUpdateIsRepeatableToTheByte`: the launch geometry is a pure function of the
device, the element type, the element count and the operands' alignment, none of which change
between two runs in one process. Two identical fused updates must agree exactly, not nearly.

`RngMetadataSlotsAreInRangeAndRoundTrip` pins a bug. The random dispatch wrote the seed to slot
10 and the offset to slot 11 of an eight-slot array. `TaskMetadata::set_param` ignores an
out-of-range index without a word, so every random task recorded a zero seed and a zero offset.
With the slots in range these read back exactly, including a seed whose value would not survive
the round trip through a double.

`ATypedTaskReportsTheFamilyItsCallerDeclared`: the family string is what a caller outside BLAS
needs from this seam. An operation recorded by the spatial domain must not be reported as a
BLAS operation when its dtype is refused. It is pinned here and not in the spatial tests
because the guarantee belongs to the seam and not to any one of its callers.

## test_gemm.cpp

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

`Unit_GEMMKernelTest` does two things the typed differential suite cannot. The first is
hand-computed expectations; see "Expectations are worked out by hand". The layout and transpose
cases are checked against numbers worked out by hand. The second is variant coverage. GEMM
picks between its local-tiled and no-local shapes from the device profile, so on any one
machine the public API only ever reaches one of them. These tests call both directly.

`ColumnMajorMatchesHandComputedResult` stores the row-major problem column-major: A (2x3) is
`[1,4,2,5,3,6]` with lda = 2, and B (3x2) is `[1,0,1,0,1,1]` with ldb = 3. This is the case
that was wrong. The old kernels read a column-major A as if it were row-major with a
column-major leading dimension. That produced the wrong element and, because that leading
dimension is m and not k, ran off the end of the buffer whenever m > k. It survived because the
reference kernel had the identical defect.

`ColumnMajorReferenceMatchesHandComputedResult` pins the oracle itself to the same numbers.
Without it the SYCL kernel could only ever be as right as whatever the reference does.

`TransposedAMatchesHandComputedResult`: op(A) is the same `[[1,2,3],[4,5,6]]`, so A is stored
3x2 as its transpose `[[1,4],[2,5],[3,6]]`. This pins the ROW_MAJOR and transposed arm, which
must address identically to COLUMN_MAJOR and untransposed.

`EpilogueFusesBiasAndReluIntoTheStore` is worked out as follows, with A (2x2) =
`[[1,2],[3,4]]`, B (2x3) = `[[1,0,-1],[0,1,-1]]`, alpha = 2, row_bias = `[0, 1]` and col_bias =
`[0.5, -1, 5]`:

```
A*B       = [[1, 2, -3], [3, 4, -7]]
2*(A*B)   = [[2, 4, -6], [6, 8, -14]]
+ biases  = [[2.5, 3, -1], [7.5, 8, -8]]
relu      = [[2.5, 3,  0], [7.5, 8,  0]]
```

`Unit_GEMMSeamTest` runs the epilogue through the public seam and not through the kernel. The
kernel test calls `Kernels::SYCL::gemm` directly, which no consumer can do through the Engine
API. The seam tests go through `Dispatch::execute_gemm`: recorded as a task on the graph, run
by `Engine::execute()`, dispatched on the runtime dtype. That is the seam that is supported,
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
specification, so it has to apply the epilogue too. Otherwise VERIFY mode would declare every
fused GEMM wrong, and FORCE_REFERENCE would quietly compute a different function from
FORCE_SYCL.

`ExecuteGemmWithTheIdentityEpilogueIsPlainGemm` asserts what `Level3::gemm` now is: the unfused
entry point and the seam with the identity must agree element for element, or the delegation in
`src/engine/blas/level3/gemm.cpp` changed behaviour.

### The epilogue's own operands

`execute_gemm` declares A, B and C. Whatever the epilogue itself touches is invisible to it.
The functor is an opaque callable, and that opacity is the feature, so an undeclared bias is
not a wrong dependency edge, it is no edge, and the graph is free to run this GEMM at the same
time as whatever writes the bias. Nothing throws and nothing is NaN; the run computes with a
value that had not arrived yet. `Dispatch::gemm_epilogue(...).reads(t)/.writes(t)` is how the
caller hands that back. Every assertion in this group is on `in_degree`; see "Ordering is
asserted on in-degree".

`BiasColumnEpilogue` is deliberately simpler than `BiasReluEpilogue`. These tests are about
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
COLUMN_MAJOR arm the moment it was written.

## test_iamax.cpp

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

## test_integer_dtypes.cpp

A dtype that is half-supported is worse than one that is not supported at all, because
"accepted" reads as "handled". So this file is deliberately lopsided: a handful of tests that
the supported operations give the right answer, and a sweep asserting that every operation
outside that list throws, by name, at record time. The sweep is the part that has to stay green
as operations are added.

`ElementWidthsAndNamesAreDeclared`: the rest of the file depends on these widths. They are
cheap to assert, and a memory planner sizes an arena from them.

`ConstantFillsAnIntegerTensorExactly`: the value is stored exactly, well past where a float
would have started rounding.

`CopyMovesIntegerElements`: `copy` is the one BLAS operation that takes `Domain::ALL`, because
it does no arithmetic to what it moves.

`ComparisonsWorkOnIntegerTensors`: the results are 1 and 0, which is what an integer element
is for.

`ArgmaxSearchesAnIntegerTensor`: an ordering exists over integers, so the operation is defined,
and the answer is still an INT64 index and not an element of the input's dtype.

`ArgmaxIsExactPastTheFloatMantissa` uses an index past 2^24, which is where the old
`static_cast<T>(idx)` into a FLOAT32 result started returning a neighbour instead of the
answer. 16,777,217 is the first integer a float cannot represent.

`UnsupportedOpsRejectIntegerOperandsByName` is the sweep. Every operation family outside the
supported list must refuse an integer operand at record time, before anything is queued, and
the message must name the operation and the dtype and not an enum ordinal.

`OrderingComparisonsStillRejectComplex`: widening the comparisons to integers must not have
undone the refusal of complex operands.

## test_mean_spatial.cpp

The file covers `mean_spatial`, and the regression that the second segment axis it added did
not disturb the reductions that have only ever had one. Every expectation is worked out in
double from the same generating rule the input was filled from, never from another path in this
library. A differential oracle against `sum_along` over a reshaped view would agree with
`mean_spatial` in exactly the cases where both read the strides wrongly.

`FoldsTheTwoMiddleAxesOfAnNhwcTensor` is the shape the second segment axis exists for: one
result per (n, c), whose input offset is `n * H*W*C + c` and which no single segment stride
reaches.

`ASingleSegmentPairIsStillCorrect`: N = 1 and C = 1 collapses both segment counts to one, the
case a generalisation most easily breaks. A stride that is never multiplied by anything but
zero is a stride nothing checks.

`ALongExtentReachesTheCooperativeShape`: one fold of 262144 elements is too few segments to
fill the machine and a segment far longer than a work-group, so `plan_axis` takes the
cooperative shape. Both shapes compute their segment's base address, and both had that
computation changed, so covering only the item-per-segment shape would leave half the change
untested.

`ASpanWithNoSecondAxisReducesExactlyAsItAlwaysDid`: every existing caller aggregate-initialises
`AxisSpan` with five values and so gets `segments2 = 1`, `segment_stride2 = 0` by default. If
the fold index is decomposed wrongly at that default, every row-wise reduction in the library
is wrong, and this is the cheapest place in the suite to see it.

## test_orthogonal.cpp

`RandomOps::orthogonal` was declared in an installed header, implemented as a forward to
`normal()` and commented out of the build. A consumer calling it failed at link, and anyone who
un-commented it got a Gaussian matrix under an orthogonal name. It is now a modified
Gram-Schmidt orthonormalization, and the tests check the only property that distinguishes it
from the fallback it replaced.

`expect_orthonormal` is the whole contract: an orthogonal matrix is exactly one whose vectors
have unit norm and vanishing pairwise inner products.

`ResultIsNotJustAGaussianFill` is the regression the rewrite was made for. The earlier
implementation returned `normal(0, 1)`, whose columns are neither unit-norm nor orthogonal. A
Gaussian 6x6 has column norms near sqrt(6), about 2.45, so a unit diagonal cannot be an
accident.

## test_reduction_determinism.cpp

The folding reductions were rebuilt for two properties: a bit-identical answer on every run,
and a summation accurate enough to trust. Both are asserted at both reduction shapes.
`Reduce::planned_partials` sends a CPU below 16K elements down a single sequential pass and
everything else down the two-stage work-group tree. A suite that only ever reduced 100 elements
would leave the entire parallel path untested: the one a GPU always takes, and the one where
combination order is in question. Every case therefore runs at a size on each side of that
threshold.

The segmented reductions, `max_along` and `sum_along`, assert the same two properties in
`test_axis_reductions.cpp` at both of their own launch shapes. They live there because every
helper in this file is built around reading one float back.

The stability helper compares with `std::memcmp`, not `EXPECT_NEAR`. The guarantee SushiBLAS
inherits from `sushiruntime_deterministic_fp` is byte equality for the same binary on the same
device, and a tolerance-based check cannot tell that from "close enough". Each run gets a fresh
Engine so nothing is reused between runs. A degenerate result would make byte equality trivial
and the case vacuous: zero, an infinity or a NaN compares stable no matter how the reduction
combined. So the input has to produce a number first.

`multiplicative_value` exists because `prod` needs a different input. The additive generator is
mostly sub-unit, so its running product flushes to zero long before 256K terms, and zero is
bit-stable for trivial reasons, which would have made the product's determinism case assert
nothing. The multiplicative values sit just either side of 1.0 with a zero-mean logarithm over
their period, so the product stays finite and non-zero at every size used and the ordering is
observable in it.

`BothShapesAreReachedHere` guards the file's own premise. If a retune ever moved the CPU
threshold past `kLarge`, every case would silently test the sequential path twice and the
parallel one never, and still pass.

`CompensatedSumRecoversASmallTailUnderALargeHead` shows the accuracy Neumaier compensation buys
on an input built to lose it: one large magnitude followed by many small ones. In `float`, 1e8
has an ulp of 8, so `1e8 + 1` rounds straight back to 1e8 and a naive running sum discards
every one of the 4096 ones that follow. It returns exactly 1e8, off by the entire tail. The
compensation term holds what each addition dropped and gives back 100004096, which is exactly
representable because it is a multiple of 8, so the test asserts an exact value. The gap
between right and wrong is 4096, not an ulp, so the test cannot pass by accident. It was
verified to fail without the compensation by removing it: the result came back as 1e8.

`CompensatedSumBeatsNaiveSummationOnRepeatedTenths` checks the same property on a long run of
0.1f, a value with no exact binary representation, so every addition rounds. The test pins no
constant. It computes the exact answer (in double, over the float values stored) and what a
naive float pass would produce, and asserts the device beat the naive one. On the single-pass
shape the device kernel adds the same values in the same order as `naive`, so without the
compensation the two are bit-identical and `EXPECT_LT` fails outright.

`TwoStageTreeAgreesWithTheSequentialPass`: `Reduce::reduce` picks its shape from the device, so
on the machine these were written on the public `reductions().sum` path only reaches the
two-stage kernel above 16K elements and the sequential one below. The test calls both at the
same small size, by handing the two-stage path a scratch buffer it would not otherwise get, so
the work-group tree is checked against an answer that does not depend on it. The tree is the
part a CPU under-exercises: on a GPU it is the only path there is, and this project has no GPU
to run it on. The two results are not bit-equal, because they sum in different orders, but both
must be right and both must be stable run to run.

## test_reductions.cpp

`ArgMax` and `ArgMin` write an INT64 result, not one of the input's own dtype. `argmax` answers
with an index, and an index stored in a float stops being exact past 2^24 (past 2048 in HALF).
The index is compared exactly for the same reason: it is either right or wrong.

## test_shuffle.cpp

`RandomOps::shuffle` was declared in an installed header, implemented as a no-op stub and
commented out of the build, so any consumer that called it failed at link. It is now a
Fisher-Yates permutation, and the tests are the properties that make it one.

`ResultIsAPermutationOfTheInput`: being a permutation is the defining property, and the old
stub satisfied it trivially by doing nothing. Hence the second half of the test, which asserts
that something moved. Identity is a legal permutation, but the odds of drawing it from 64! are
not distinguishable from zero, so seeing it means the kernel never ran.

`SameSeedReproducesTheSamePermutation`: a fresh engine restarts the RNG offset counter, so the
second run uses the same (seed, offset) pair as the first. That is the reproducibility contract
the header advertises.

`ConsecutiveShufflesOnOneEngineDiffer` guards against the failure mode of an RNG keyed on the
seed alone, with no offset.

## test_spatial_broadcast.cpp

`nchw_view` builds its view through `Tensor`'s stride-explicit constructor, which takes the
whole fixed-width stride array, so nothing in the helper reaches past the library's own seams.
`create_tensor` only ever lays a shape out contiguously, so the allocation comes from it and
the layout comes from the helper. The logical shape stays `[N, H, W, C]` because that is what
`layout_convert` compares: a conversion permutes strides, not shapes.

`BroadcastSpatialAtOneOverHwIsTheMeanBackward` checks the property `SPATIAL_MEAN_BACKWARD`
needs: each (n, c) plane sums back to the 1.0 that global average pooling's forward divided
out. The sum is not exact and is not expected to be. 1/20 has no finite binary expansion, so
twenty copies of `float(1/20)` add up to 1.0 only to within the rounding each addition carries.
A few ulps of 1.0 is roughly 1e-7; the tolerance of 1e-6 leaves room without hiding a real
error, which would be off by a factor of H, of W or of N.

## test_spatial_channels.cpp

`ConcatPlacesEachSourceAtItsOffset` uses two pixels, so a source that ignored its column stride
would still pass the first pixel and fail the second.

`TheyAreEachOthersBackward` is the property the autograd rule stands on: the gradient of a
concat is a slice at the same offset and of the same width. It is written as one round trip
through the offset the second source occupies, so an off-by-one in either direction shows up
and does not cancel.

`AStridedSourceIsAddressedThroughItsStrides` builds its source through `Tensor`'s
stride-explicit constructor, the same public seam `test_tensor_view.cpp` exercises: a
`[1,2,2,2]` view over a `[1,2,2,4]` parent whose channel stride is 2, so it sees the parent's
even channels only. A kernel that assumed a unit channel stride would copy channels 0 and 1
instead of 0 and 2.

## test_spatial_col2im.cpp

`EachPixelReceivesOncePerWindowThatReadIt`: col2im of an all-ones column matrix is the overlap
count of each pixel, the number of windows that read it. The test computes that answer
independently, and it is the strongest cheap check that the gather visits exactly the right
set. A rule that drops contributions or double-counts fails it, where a round trip against
im2col would not necessarily. With a 3x3 stride-1 window padded by 1, a pixel is read by every
window whose origin is within one step of it and inside the image: 3x3 in the interior, fewer
at the edges. The count is made by brute force and not by a closed form, so the expectation
cannot share a mistake with the kernel.

`StrideAndDilationSelectTheRightContributors`: a stride-2 window means every pixel is read by
exactly one window. Getting the divisibility test wrong, by dropping the `% stride != 0` guard,
turns this into a count of 4 and is otherwise invisible.

`PixelsNoWindowReadsAreWrittenZeroRatherThanLeftAlone`: a 2x2 stride-2 window over a 5x5 image
covers rows 0..3 and columns 0..3, and the last row and column are read by nothing. The gather
form writes every pixel exactly once, so those must come back zero, which is what lets a caller
pass an uninitialised destination. The destination is seeded with a sentinel so "left alone" is
visible.

`TheSyclKernelAgreesWithTheReferenceKernel` uses a ramp for the column entries and not all
ones, so a pixel's total depends on which entries it gathered and not only on how many. The
all-ones case cannot tell a wrong set of the right size from the right one.

## test_spatial_geometry.cpp

The file is device-free by construction: `geometry.hpp` includes `<cstdint>` and nothing else,
so this file links no SYCL and needs no Engine. That is the property the header exists to have.
SushiAI's host-lane shape inference shares this arithmetic and must not inherit the device
stack to get it.

`AsymmetricPaddingIsWhatHoldsYoloTinysLastPool`: YOLOv3-tiny's 2x2 stride-1 max-pool is padded
on the bottom and right only, so 13x13 stays 13x13. This is the case CV-5 exists to make
ordinary, and it is the reason `pad_bottom` is a host-side number and not a ninth attribute
slot. The second assertion pads the other side, because symmetric and asymmetric padding of the
same total are not interchangeable.

`AWindowWiderThanThePaddedInputYieldsNoOutputRatherThanANegativeOne`: the formula's numerator
goes negative here. Integer division of a negative numerator is implementation-flavoured and
would produce a garbage extent that later multiplies into an allocation size, so the case is
answered explicitly.

`TheDefaultGeometryIsTheIdentityWindow`: a default-constructed geometry is 1x1, stride 1, no
dilation and no padding, so it maps every extent to itself. Anything that reads a half-built
geometry therefore gets a shape it can recognise as wrong, not a plausible one.

## test_spatial_im2col.cpp

`AUnitWindowIsTheIdentityGather`: with a 1x1 window, stride 1 and no padding, the column matrix
is the input reshaped, element for element and in the same order. If this fails, the row and
element index arithmetic disagree with each other and nothing further is worth reading.

`EachRowIsOneWindowWithChannelInnermost`: a 2x2 stride-1 window over a 1x3x3x2 input gives 4
windows of 8 entries each. The rows are written out by hand and not derived, so the test is a
specification and not a second implementation.

`OutOfBoundsTapsReadZeroRatherThanNeighbouringMemory`: padding is a parameter, not a separate
zero-padded copy, which saves a full-tensor copy forward and a crop backward. This test is the
property that makes that safe.

`AsymmetricPaddingNeedsTheExplicitExtentOverload`: a 2x2 stride-1 window over a 3x3 image
padded on the bottom and right only gives 3x3 windows. The two-operand overload cannot express
this, because it has no `pad_right` to read, and the test checks that it refuses and does not
silently produce a differently shaped answer.

`TheSyclKernelAgreesWithTheReferenceKernel` compares bit-exact, not near. im2col only ever
copies an element or writes a zero, so the two kernels must produce the same bits or one of
them is addressing memory differently from the other. See "Comparing two kernels outside VERIFY
mode" for why the comparison is made in the test.

## test_spatial_pool_avg.cpp

`TheAverageIsOverTheWholeWindowIncludingPaddedTaps` is the pin for decision D-A4. Without a
padded case the convention is invisible: an unpadded pool divides by the area and by the
in-bounds count at once, so every other test in the file passes under either rule. The top-left
window's only in-bounds tap is the element 4. Divided by the window area that is 1; divided by
the in-bounds count it would be 4. That difference is the whole of `count_include_pad`.

`OverlappingWindowsSumTheirShares`: with a 2x2 stride-1 window over a 3x3 image, the centre
pixel is covered by all four windows and the corners by one each. A gather that stopped at its
first covering window would give the centre 0.25.

`TheBackwardOfAConstantForwardIsUniform`: a stride-1 pool padded so that every pixel is covered
by exactly R*S windows. Each contributes 1/(R*S), so every pixel must come back exactly 1.0,
the closed-form check that the divisor and the window set agree with each other. A divisor
computed per window from the in-bounds count would make the border pixels larger.

## test_spatial_pool_max.cpp

`four_by_four` returns the image every hand-checked case pools. It is chosen so each 2x2
stride-2 window's maximum sits in a different corner of the window. A kernel that always
picked, say, the last tap in scan order would pass a window whose maximum happened to be there.

```
1  3 | 2  0
4  2 | 1  5
     +
0  7 | 6  2
8  1 | 3  9
```

`PaddingNeverWinsAWindow` was written and run before any kernel existed. A zero-initialised
running maximum is the single likeliest defect in this operation, and it is invisible in every
test whose data is non-negative. Every element is negative, and the window at (0,0) sees
exactly one of them; its other three taps fall outside the image. Those taps are the absence of
an element, not zeros. A max initialised to `T(0)` would report 0, a plausible number that no
shape or dtype check can catch. The elements are ordered so each window's maximum is a
different element. The largest sits at the bottom right, so a window that saw only the top-left
corner cannot accidentally agree with one that saw everything.

`TheIndexNamesTheElementTheMaximumCameFrom`: the index is the logical flat position
`((n*H + h)*W + w)*C + c`, not an offset into this particular view. A strided offset would stop
being comparable the moment the gradient arrived through an alias.

`ATieGoesToTheFirstElementInScanOrder`: deterministic tie-breaking is what makes the recorded
index single-valued, and therefore what stops the backward's equality test from matching two
taps of one window and double-counting the gradient.

`TheBackwardRoutesEachGradientToItsArgmaxAndNowhereElse`: every non-argmax pixel must be
exactly zero, not merely small. The gather writes each pixel once, so an unrouted pixel is a
written zero and never a leftover.

`OverlappingWindowsAccumulateAtASharedArgmax` is YOLOv3-tiny's 2x2 stride-1 pool. One dominant
pixel is selected by all four windows, so its gradient is the sum of all four. A scatter would
have needed an atomic add for this case, and a gather that stopped at its first match would get
it wrong.

`TheSyclKernelAgreesWithTheReferenceKernel` compares the indices as well as the values. Two
kernels can agree on every maximum and still disagree on where it came from, which is what a
tie broken the other way is, and the gradient follows the index, not the value.

## test_spatial_upsample.cpp

`TheBackwardSumsTheBlockThatReadEachInput` sums the expectation by walking the block index by
index, with no closed form the kernel could be wrong in the same way.

## test_storage_offset.cpp

Every operation family runs against tensors that are views at a non-zero `storage_offset`, not
at the base of their allocation. This is the shape a memory planner produces: one arena
allocation with many tensors packed into it, each handed to an operation as a view at an
offset. Most families once captured `tensor.storage->data_ptr` and dropped the offset, so a
view silently aliased the start of its arena, reading the wrong inputs and writing over
whatever else lived there.

Each test asserts two things, and the second carries the weight. That the result is correct
often passes by accident: with a single tensor per allocation, offset 0 and "ignore the offset"
are the same address. That the poison bytes before and after the view survive untouched cannot
pass by accident, because an operation that ignores the offset writes over them.

The tests run in family order: BLAS levels 1 to 3, elementwise, nonlinear, logic, reductions,
random, and last the two tests about what an offset counts.

`OffsetIsCountedInElementsNotBytes`: the unit of `storage_offset` is a decision, not an
accident, and it is the one that keeps every view naturally aligned. An element count scaled by
the element width can never land an fp64 view on a 4-byte boundary. The test pins it so that
"offsets are bytes" cannot be introduced later and pass the family tests, which use one dtype
at a time and would not notice.

`DisjointViewsOfOneAllocationDoNotAlias` is what the memory planner in the layer above needs:
two tensors packed into one arena must not be reported as touching the same memory, or every
operation in the graph serialises behind every other one.

## test_syr.cpp

With x = `[1, 2, 3]`, `x * x^T` is

```
[1 2 3]
[2 4 6]
[3 6 9]
```

and since `upper=false` only the lower triangle of A is updated, so A in row-major order is

```
[1 0 0]
[2 4 0]
[3 6 9]
```

## test_syr2.cpp

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

## test_tensor_view.cpp

The file asks whether a view's metadata describes the memory it occupies. The question it asks
of `slice` is not "are the values right": a slice does not move anything, so there is nothing
to get right in that sense. It is "do the strides the view reports lead to its elements".
`slice` once answered no for every `dim > 0`, because it rebuilt the view through the
shape-derived Tensor constructor, which lays contiguous strides out afresh from the new shape.
For `dim == 0` that coincides with the truth; for an inner axis it describes a compacted tensor
that does not exist. Nothing in the tree called it that way, so nothing caught it.

Each test therefore addresses the view's elements through its own reported strides and checks
what it finds against the value the parent put there. That is the only formulation that fails
when the strides are wrong, and it is what a downstream kernel would do.

`SliceAlongEachAxisAddressesItsElementsRowMajor`: dim 0 always worked; dim 1 is the case that
reported contiguous strides for non-contiguous memory. In column-major the strides run the
other way and a dim-0 slice is the non-contiguous one. The rank-3 test adds an axis that is
neither the outermost nor the innermost, which a two-dimensional test cannot distinguish.

`InnerAxisSliceReportsItselfNonContiguous`: a slice keeps its parent's strides, so it reports
non-contiguous whenever narrowing an inner axis leaves gaps. `reshape`, which requires
contiguity, refuses it and does not produce a second wrong view.

`FullWidthSliceIsTheIdentity`: a whole-axis slice changes nothing, so it must leave a tensor
that compares equal to its parent in offset, strides and shape.

`RegionCoversEveryByteAStridedViewReaches`: `region()` is what SushiRuntime's dependency
tracker orders on, so it has to cover every byte the view can reach, not merely as many bytes
as the view has elements. For a strided view those are different numbers, and the smaller one
would leave the tail of the view unguarded.

`ContiguousTensorSpansExactlyItsElements`: the contiguous case must not have moved.
`element_span` and `num_elements` agree there, which is why the distinction went unnoticed for
so long.

`TransposeSpansItsWholeMatrix`: a transpose swaps strides without moving data, so it spans its
whole matrix, and it always did, which makes it a useful control on `element_span`.

`StrideExplicitConstructorRejectsIncoherentMetadata`: the stride-explicit constructor is the
seam `slice` now goes through, so its guards are the last thing between a mis-described view
and a kernel reading past the end of an allocation. In the last case, 4 rows at a stride of 64
reach element 193 of a 16-element allocation. The element count alone (16) would have fitted,
which is why the capacity check has to be asked in terms of the span.

## test_trsm.cpp

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

# Tests of reductions

The tests of the `reductions` module, built into `sushiblas_reductions_test_objects` and linked
into `sushiblas_operations_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

Each `benchmark_*.cpp` file is a program of its own, built when `SB_BUILD_BENCHMARKS` is on;
the benchmark section of `tests/README.md` describes them.

| File | Tests |
| --- | --- |
| `benchmark_reductions.cpp` | Times full-tensor sum, max and var and prints GB/s per element count |
| `test_arg_reduction.cpp` | Tests the arg reduction dispatcher on the reference path and on the host device |
| `test_centred_reduction.cpp` | Tests the centred reduction dispatcher as a variance, on the reference path and on SYCL |
| `test_dispatch.cpp` | Tests the partial sizes, the lease rule and an index reduction recorded on a host engine and on a runtime engine |
| `test_reductions/test_axis_reductions.cpp` | Tests the axis reductions, the plan they launch from and both of their kernels |
| `test_reductions/test_count_nonfinite.cpp` | Tests ReductionOps::count_nonfinite, its counts and its ordering in the graph |
| `test_reductions/test_max_min_nan.cpp` | Tests max, min, max_along and min_along on NaN, infinite and signed-zero input |
| `test_reductions/test_mean_spatial.cpp` | Tests ReductionOps::mean_spatial and the single-axis reductions that share its span |
| `test_reductions/test_reduce_segments.cpp` | Tests Reduce::reduce_segments in both launch shapes and the axis fold that runs on it |
| `test_reductions/test_reduction_determinism.cpp` | Tests that the folding reductions repeat to the bit and sum accurately |
| `test_reductions/test_reductions.cpp` | Tests each reduction's results, and that its host and SYCL paths agree |
| `test_reductions/test_refusals.cpp` | Tests what the reductions refuse, what an empty operand writes and that a reference reduction leases no scratch, `Small_` on a host engine |
| `test_reductions/test_scratch_buffer.cpp` | Tests ScratchBuffer, the lease on one pooled device scratch block |
| `test_reductions/test_sum_edge_cases.cpp` | Tests ReductionOps::sum, mean, max and min on edge-case inputs |

## test_reductions/test_axis_reductions.cpp

The file covers the segmented reduction seam: `max_along` and `sum_along`, the geometry they
are planned from, and the two kernels that geometry chooses between. Every expectation is
worked out by hand. A differential oracle against `gemv(X, ones)` would have agreed with a row
sum that read the wrong stride in exactly the layouts where both read it wrongly.

Determinism for this seam is tested here and not in `test_reductions/test_reduction_determinism.cpp`, because
that file's helpers are built around a scalar result and this seam's answer is a vector. The
property and the means are the same: `std::memcmp`, not a tolerance.

The file holds these suites: the planner (`Small_AxisReductionPlan` and `Small_AxisReductionWidth`,
rows over a `Device::Profile` value that need no device), the public operations over every dtype a
reduction serves (`Medium_AxisReductionsTyped`), layout, shape and the two kernels
(`Medium_AxisReductions`), the byte stability
of each kernel shape (`Medium_AxisReductionStability`).

`K_ROW_MAJOR` holds small whole numbers on purpose. They are exact in HALF as well as in the
wider types, so one expectation serves every dtype and a mismatch is a real disagreement, not a
rounding difference.

```
     1  -2   3   0      row max  3    row sum   2
     5   4  -1   2      row max  5    row sum  10
    -3  -7  -2  -9      row max -2    row sum -21

  col max  5   4   3   2
  col sum  3  -5   0  -7
```

`TheShapeFollowsTheSegmentGeometry` is one row per geometry. An enough-segments row takes one
item per segment, the cheap shape whenever there is already enough work to go round: it needs no
local memory and no barriers, and its fold order is trivially fixed. A few-long-segments row has
too few segments to fill the machine and segments long enough to be worth a work-group, the only
case where the tree earns its barriers. A few-short-segments row at one short of the group width
declines the tree, since a group whose segment is shorter than the group is mostly idle lanes
paying log2(local) pairs of barriers; the row at the group width takes it. The empty rows, a zero
extent and a zero segment count, must not select a shape that would launch `segments * local`
work-items over nothing, or divide by a zero extent.
`ACooperativePlanTakesTheGroupWidthOfItsDevice` holds that a cooperative plan uses the whole
group width of a CPU and of a GPU profile.

`ColumnMajorOperandsFoldTheRowsOfTheSameMatrix` and
`ColumnMajorOperandsFoldTheColumnsOfTheSameMatrix`: the axis is a question about the matrix, not
about memory order, so the same logical matrix must reduce to the same vectors under either
layout. The kernel never learns which layout it is on. `Tensor::strides` already carries it, and
`axis_span_for` turns the pair into a stride pair.

`AViewAtANonZeroOffsetReducesItsOwnElements`: every address the kernel forms is
`data() + s * segment_stride + k * element_stride`, and `data()` already carries the offset, so
this checks that the offset was not dropped on the way through the seam.

`AxisReductionsOverViewsOfOneAllocationSeeTheirWriter` checks the value and
`AnAxisReductionIsOrderedBehindTheOpThatWroteItsInput` checks the recorded dependency; together
they are the bare-pointer hazard described under "Views at a non-zero offset", on this seam.
`Medium_Sentinel.ReductionsOrderAgainstViewsOfOneAllocation` pins it for the whole-tensor
seam.

`MaxAlongRefusesIt` and `SumAlongRefusesIt`, one row per dtype (`test_refusals.cpp`): a fold needs an ordering and an
identity, which `std::complex` does not have, and an integer tensor is not an arithmetic type in
this library. It is the same `Domain::Real` rule the whole-tensor reductions carry.
`MaxAlongRefusesItAtRecordTime`, `MinAlongRefusesItAtRecordTime` and
`SumAlongRefusesItAtRecordTime` run one row per malformed pair of operands.

`TheCooperativeTreeAgreesWithTheItemPerSegmentFold`: `plan_axis` picks one shape per input, so
on any single machine the public operations only ever reach one of the two kernels for a given
size. The test drives both directly, over one input, at a size the planner would never send to
the cooperative kernel, and holds each sum to `n * u * sum|x|` of the sum in double. That is the only way the work-group tree gets exercised on a machine
with no GPU. The extent is deliberately not a multiple of any work-group width, so the tree's
trailing lanes fold fewer elements than the leading ones, and the lanes past the extent fold
none and contribute the identity. The two results are not bit-equal to each other, because they
add in different orders; that is the point of having two shapes. Both must be right, and both
must be stable run to run, which `AShapeIsBitwiseStable` asserts.

`TheItemPerSegmentShapeFindsTheSegmentMaximum` and `TheWorkGroupTreeFindsTheSegmentMaximum`: the
per-segment answers differ from the whole-tensor answer by a lot. `max` is exactly associative,
so the two shapes must give the expected values to the bit here and not merely to a tolerance. Segment `s` peaks at `100 * s + 1`, at a position that moves with
`s`, so no shape can be right by reading a fixed lane.

`RowSumsAreCompensated`: each row is one large head followed by 4096 ones. In float, 1e8 has an
ulp of 8, so `1e8 + 1` rounds straight back to 1e8 and an uncompensated serial fold returns
exactly 1e8, short by the entire tail. 100004096 is a multiple of 8 and therefore exactly
representable, so the test asserts that value with equality.

`AShapeIsBitwiseStable`: the guarantee SushiBLAS inherits from
`sushiruntime_deterministic_fp` is byte equality for the same binary on the same device, so the
test compares bytes. A tolerance-based check cannot tell that apart from "close enough", which
is the whole distinction under test. It runs at both shapes by driving the kernel directly, for
the same reason the cooperative-tree test does. A degenerate result would make byte equality
trivial and the case vacuous, so the input has to reduce to a number first.

`ARowWiseShiftSurvivesAGapThatDestroysAGlobalOne` is the numerical property the axis reduction
was added for: a log-sum-exp shifted by each row's own maximum survives a gap between a row and
the batch maximum that destroys the globally shifted form. Every row is three equal values, whose
sum of shifted exponentials is exactly 3 for any offset, which makes the expectation
hand-computable and independent of the offset under test. The library makes no promise about the
globally shifted form, so the test does not assert its underflow.

## test_reductions/test_count_nonfinite.cpp

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
at n-1, one test each. This runs at a size above `Reduce::K_CPU_TWO_STAGE_MIN_ELEMENTS`, where the
two-stage kernel with its grid-strided walk and halving tree is the one running.

`EveryBadElementIsCounted`: the answer is a count and not a flag. Several bad elements of both
kinds, spread across the tensor, must add up. This distinguishes the operation from a
saturating "any" and is the extra information the INT64 result buys.

`TheSinglePassShapeCountsBothBadEnds` and `TheTwoStageShapeCountsBothBadEnds` run the same
content at a size on each side of the CPU's crossover from the single pass to the two-stage
kernel. Both shapes must give the same count, which is what lets the threshold be a pure
performance choice. The local is named `below` and
not `small` because `<rpcndr.h>`, which reaches this translation unit through the Windows
headers, defines `small` as a macro for `char`.

`LargeFiniteValuesCountZero` holds that eight finite values of 20000, which binary16 represents
exactly, count zero. The values are chosen because their sum leaves binary16's range long before
any element does, so a screen built on a sum of magnitudes would call the tensor non-finite. In a
loss-scaling loop that wrong answer halves the scale on a step that never overflowed, and does it
again on the next step, for ever.

`IsRefusedByNameAndRecordsNothing` of `Small_ReductionDtype`, one row per dtype (`test_refusals.cpp`): an integer element cannot be infinite or NaN, so "0"
would be true and useless. It would tell a caller who handed over the wrong tensor that the
right one was screened.

`IsOrderedBehindTheOpThatWroteItsInput` puts the input partway into an arena; see "Views at a
non-zero offset". `AnOverflowProducedInsideTheGraphIsFound` is the same thing behaviourally,
end to end.

## test_reductions/test_mean_spatial.cpp

The file covers `mean_spatial`, and the regression that the second segment axis it added did
not disturb the reductions that have only ever had one. Every expectation is worked out in
double from the same generating rule the input was filled from, never from another path in this
library. A differential oracle against `sum_along` over a reshaped view would agree with
`mean_spatial` in exactly the cases where both read the strides wrongly.

`FoldsTheTwoMiddleAxesOfAnNhwcTensor` is the shape the second segment axis exists for: one
result per (n, c), whose input offset is `n * H*W*C + c` and which no single segment stride
reaches. Each mean is held to `u * sum|x|` of the mean in double.

`ASingleSegmentPairIsStillCorrect`: N = 1 and C = 1 collapses both segment counts to one, the
case a generalisation most easily breaks. A stride that is never multiplied by anything but
zero is a stride nothing checks.

`ALongExtentTakesTheCooperativeShape` and `TheMeanOfALongExtentIsCorrect`: one fold of 262144
elements is too few segments to fill the machine and a segment far longer than a work-group, so
`plan_axis` takes the cooperative shape, which the first test asserts on the test device. Both
shapes compute their segment's base address, and both had that computation changed, so covering
only the item-per-segment shape would leave half the change untested.

## test_reductions/test_reduction_determinism.cpp

The folding reductions were rebuilt for two properties: a bit-identical answer on every run,
and a summation accurate enough to trust. Both are asserted at both reduction shapes.
`Reduce::planned_partials` sends a CPU below 16K elements down a single sequential pass and
everything else down the two-stage work-group tree. A suite that only ever reduced 100 elements
would leave the entire parallel path untested: the one a GPU always takes, and the one where
combination order is in question. Every case therefore runs at a size on each side of that
threshold.

The segmented reductions, `max_along` and `sum_along`, assert the same two properties in
`test_reductions/test_axis_reductions.cpp` at both of their own launch shapes. They live there because every
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

`ACPUTakesTheSinglePassAtTheSmallSize` and `ACPUTakesTheTwoStageTreeAtTheLargeSize` guard the
file's own premise on a CPU profile, and `TheLargeSizeReachesTheTwoStageShapeHere` and
`TheMomentsSizeReachesTheTwoStageShapeHere` guard it on the test device. If a retune ever moved
the CPU threshold past `K_LARGE`, every case would silently test the sequential path twice and
the parallel one never, and still pass.

`CompensatedSumRecoversASmallTailUnderALargeHead` shows the accuracy Neumaier compensation buys
on an input built to lose it: one large magnitude followed by many small ones. In `float`, 1e8
has an ulp of 8, so `1e8 + 1` rounds straight back to 1e8 and a naive running sum discards
every one of the 4096 ones that follow. It returns exactly 1e8, off by the entire tail. The
compensation term holds what each addition dropped and gives back 100004096, which is exactly
representable because it is a multiple of 8, so the test asserts that value with equality. The
gap between right and wrong is 4096, not an ulp, so the test cannot pass by accident. It was verified to fail without the compensation by removing it: the result came
back as 1e8.

`CompensatedSumBeatsNaiveSummation` and `CompensatedSumLandsWithinTheRoundingBound` check the same
property on a long run of 0.1f, a value with no exact binary representation, so every addition
rounds. The tests pin no constant. They compute the exact answer (in double, over the float
values stored) and what a naive float pass would produce, and assert the device beat the naive
one and lands within the derived rounding bound of the exact sum. On the single-pass shape the
device kernel adds the same values in the same order as `naive`, so without the compensation the
two are bit-identical and `EXPECT_LT` fails outright.

`TheTwoStageTreeSumsWithinTheRoundingBound` and `TheSequentialPassSumsWithinTheRoundingBound`:
`Reduce::reduce` picks its shape from the device, so
on the machine these were written on the public `ReductionOps::sum` path only reaches the
two-stage kernel above 16K elements and the sequential one below. The test calls both at the
same small size, by handing the two-stage path a scratch buffer it would not otherwise get, so
the work-group tree is checked against an answer that does not depend on it. The tree is the
part a CPU under-exercises: on a GPU it is the only path there is, and this project has no GPU
to run it on. The two results are not bit-equal, because they sum in different orders, but both
must be within the rounding bound of the exact sum.

## test_reductions/test_reductions.cpp

`argmax` and `argmin` write an INT64 result, not one of the input's own dtype. `argmax` answers
with an index, and an index stored in a float stops being exact past 2^24 (past 2048 in HALF).
The index is compared exactly for the same reason: it is either right or wrong.

## test_reductions/test_refusals.cpp

The refusals of the reductions run `Small_` on a `HostEngineTest`: a host engine records under
ForceReference and reaches no device. The rows are the real-only reductions over an integer or complex
dtype (`Small_ReductionDtype`), the malformed operands (`Small_ReductionMalformed`), the axis
reductions over a malformed pair and over a dtype they cannot fold (`Small_AxisReductionRefusal` and
`Small_AxisReductionDtype`), and the lease rule (`Small_ReductionLease`). An empty sum is 0 and an empty
product is 1; an empty mean and max are refused. The device boundary of the same promises stays
`Medium_` in `test_reductions.cpp`: a device-only operand, and the empty identities on both paths.


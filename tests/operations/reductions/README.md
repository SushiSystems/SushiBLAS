# Tests of reductions

The tests of the `reductions` module, built into `sushiblas_reductions_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

Each `benchmark_*.cpp` file is a program of its own, built when `SB_BUILD_BENCHMARKS` is on;
the benchmark section of `tests/README.md` describes them.

| File | Tests |
| --- | --- |
| `benchmark_reductions.cpp` | Times full-tensor sum, max and var and prints GB/s per element count |
| `test_reductions/test_axis_reductions.cpp` | Tests the axis reductions, the plan they launch from and both of their kernels |
| `test_reductions/test_count_nonfinite.cpp` | Tests ReductionOps::count_nonfinite, its refusals and its ordering in the graph |
| `test_reductions/test_max_min_nan.cpp` | Tests max, min, max_along and min_along on NaN, infinite and signed-zero input |
| `test_reductions/test_mean_spatial.cpp` | Tests ReductionOps::mean_spatial and the single-axis reductions that share its span |
| `test_reductions/test_reduce_segments.cpp` | Tests Reduce::reduce_segments in both launch shapes and the axis fold that runs on it |
| `test_reductions/test_reduction_determinism.cpp` | Tests that the folding reductions repeat to the bit and sum accurately |
| `test_reductions/test_reductions.cpp` | Tests each reduction's results and refusals, and that its host and SYCL paths agree |
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

The file holds three fixtures: the planner
(`Unit_AxisReductionPlan`, a pure function that needs no device), the public operations over
every dtype a reduction serves (`Unit_AxisReductionsTyped`), and layout, shape, rejection and
the two kernels (`Unit_AxisReductions`).

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
`Domain::Real` rule the whole-tensor reductions carry.

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
representable, so the test asserts that value with `EXPECT_FLOAT_EQ`, which allows four ulps.

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
is not, because it stores its total in a HALF result and 8 x 20000 leaves binary16's range
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

## test_reductions/test_mean_spatial.cpp

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

`ASpanWithNoSecondAxisReducesExactlyAsItAlwaysDid`: every existing caller sets five members of
`AxisSpan` and so gets `segments2 = 1`, `segment_stride2 = 0` by default. If
the fold index is decomposed wrongly at that default, every row-wise reduction in the library
is wrong, and this is the cheapest place in the suite to see it.

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

`BothShapesAreReachedHere` guards the file's own premise. If a retune ever moved the CPU
threshold past `K_LARGE`, every case would silently test the sequential path twice and the
parallel one never, and still pass.

`CompensatedSumRecoversASmallTailUnderALargeHead` shows the accuracy Neumaier compensation buys
on an input built to lose it: one large magnitude followed by many small ones. In `float`, 1e8
has an ulp of 8, so `1e8 + 1` rounds straight back to 1e8 and a naive running sum discards
every one of the 4096 ones that follow. It returns exactly 1e8, off by the entire tail. The
compensation term holds what each addition dropped and gives back 100004096, which is exactly
representable because it is a multiple of 8, so the test asserts that value with
`EXPECT_FLOAT_EQ`. The gap between right and wrong is 4096, not an ulp, so the test cannot pass
by accident. It was verified to fail without the compensation by removing it: the result came
back as 1e8.

`CompensatedSumBeatsNaiveSummationOnRepeatedTenths` checks the same property on a long run of
0.1f, a value with no exact binary representation, so every addition rounds. The test pins no
constant. It computes the exact answer (in double, over the float values stored) and what a
naive float pass would produce, and asserts the device beat the naive one. On the single-pass
shape the device kernel adds the same values in the same order as `naive`, so without the
compensation the two are bit-identical and `EXPECT_LT` fails outright.

`TwoStageTreeAgreesWithTheSequentialPass`: `Reduce::reduce` picks its shape from the device, so
on the machine these were written on the public `ReductionOps::sum` path only reaches the
two-stage kernel above 16K elements and the sequential one below. The test calls both at the
same small size, by handing the two-stage path a scratch buffer it would not otherwise get, so
the work-group tree is checked against an answer that does not depend on it. The tree is the
part a CPU under-exercises: on a GPU it is the only path there is, and this project has no GPU
to run it on. The two results are not bit-equal, because they sum in different orders, but both
must be right and both must be stable run to run.

## test_reductions/test_reductions.cpp

`ArgMax` and `ArgMin` write an INT64 result, not one of the input's own dtype. `argmax` answers
with an index, and an index stored in a float stops being exact past 2^24 (past 2048 in HALF).
The index is compared exactly for the same reason: it is either right or wrong.

# Tests of graph_primitives

The unit tests of the `graph_primitives` module. The source comments state what a test checks; a
section below holds why a test is written the way it is.

| File | Tests |
| --- | --- |
| `test_combiners.cpp` | Tests what `Sum`, `Product`, `Minimum` and `Maximum` return, their identities and that each is trivially copyable |
| `test_fixed_order_fold.cpp` | Tests the two host folds against the fixed tile order, without a device |
| `test_fixed_order_levels.cpp` | Tests the level plan, the tile fold and the live level widths, without a device |
| `test_radix_plan.cpp` | Tests the digit count, the digit of a key and the per-digit plan, without a device |
| `test_reduce.cpp` | Tests `reduce()`: the fixed combination order, the late-bound count, the node widths and the refusals |
| `test_segmented_reduce.cpp` | Tests `segmented_reduce()`: the per-segment fold, the late-bound count, the node widths and the refusals |
| `test_prefix_scan.cpp` | Tests `prefix_scan()` against a host scan, across live counts, replays, node widths and refusals |
| `test_counting_sort.cpp` | Tests `counting_sort()`: stability, digit boundaries, late-bound replay, regions, node widths and refusals |
| `test_device_sized_primitives.cpp` | Tests that each primitive gives one result on the chain and on the scheduler as a device count changes |

## The runtime a test builds

A primitive records through a `SushiRuntime::Recorder`, which a `Graph` or a `Region`
hands out. Each test builds its own `SushiRuntime::Runtime` with `Runtime::create()` and takes
buffers and graphs from it, so a refusal one test provokes cannot reach another.
`test_device_sized_primitives.cpp` needs two runtimes with different settings in one test, which
the shared test runtime could not give.

Four files (`test_combiners.cpp`, `test_fixed_order_fold.cpp`, `test_fixed_order_levels.cpp`,
`test_radix_plan.cpp`) include
no SushiRuntime header and submit nothing to a device.

## Exact comparisons

The module promises bit-identical results, so a floating-point result is compared by its bit
pattern (`bit_equal`) against a host fold that combines in the same order, and never within a
tolerance. Where a test compares a sum with `EXPECT_EQ`, the sum is of small integers and is
exact in the element type.

Every expected tile count, level count and bin count is written in terms of `K_REDUCE_TILE` or
`K_RADIX_DIGIT_BINS`. A test that repeated the number would keep passing against itself if the
constant changed in one place only.

## test_reduce.cpp

`SumMatchesTheFixedOrderReferenceExactly`: the input is one value of 1e8 followed by ones. In
`float`, adding a one to 1e8 is lost, and adding a tile's worth of ones first is not, so the
fixed-order sum and the naive left-to-right sum differ. The test asserts that they differ; if
they did not, it could not tell a fixed order from a schedule-dependent one.

`ResultIsStableAcrossSchedules`: 32 runs of one graph, then one run on a second runtime with the
thermal rebalancer on, which migrates tasks between workers during a run.

`DeviceCountIsReadAfterItsProducer`: a kernel of the same graph writes the count. The reduce has
to fold the count written in its own run, not the one the previous run left.

`RejectsMisSizedArguments` checks that each call returns an `ErrorCode::InvalidGraph` error and
that the recorder holds no node afterwards.

## test_segmented_reduce.cpp

`FixedCountWidthIsTheSegmentCount` and `LateBoundWidthsAreOneAndTheCapacity`: the offsets buffer
is padded by nine entries and the output by eight, so a width taken from either buffer's size
shows up as a wrong number and cannot be mistaken for the segment count.

`LateBoundCountFoldsOnlyTheLiveSegments` fills the output with a value no segment folds to before
each run, so an entry past the live count that was written shows.

## test_prefix_scan.cpp

`AgreesWithTheHostScanAtTileEdges`: the offsets buffer has one element past the sentinel, filled
with a canary, so a write past `offsets[n]` shows.

`SharesOneNodeIndexAcrossStepsAndPasses` and `NodeWidthsMatchTheLevelPlan` use a capacity of two
levels, `2 * K_REDUCE_TILE + 88`. The call records five nodes in this order: the resolver, the
up pass of level 0, the down pass of the top level, the down pass of level 0 and the closing
write. Their widths are 1, 3, 1, 3 and 1.

## test_counting_sort.cpp

The stability cases first sort the same keys with `std::sort` and assert that its order differs
from the stable one. Keys whose ties `std::sort` happened to keep in place would let an unstable
sort pass.

`StableAcrossDigitsWhenTheLowDigitTies`: every key has a low byte of zero, so the first digit
pass ties on every element and the second has to keep the order the first left.

`DigitsAboveTheLiveKeyRangeDoNotRun`: the capacity needs two digits and the live key range one.
The key `0x0101` is outside the live range on purpose. One digit pass places it by its low byte,
between 0 and 2; a second pass would move it last. The order therefore says how many passes ran.

`NodeWidthsMatchThePlan`: 300 elements and 50 buckets record seven nodes: the resolver (1), the
histogram (2 tiles), the column scan (one per bin), the bin base (1), the placement (2), the
offsets (51) and the counts (50).

`LargeRandomCaseAgainstHostReference` sorts 10^6 keys over a range of 2^20 and
`LiveKeyRangeBelowCapacityLandsInIndices` allocates 2^24 + 1 offsets. They are the two slowest
tests of the module.

## test_device_sized_primitives.cpp

Each case builds the same graph on a default runtime and on one with `force_scheduler`, runs 48
steps and compares one digest per step. A producer kernel writes the live count from a schedule of
eleven values that includes zero and the capacity. The case also asserts that the default run
took the chain layer, that the forced one took the scheduler, and that the digest changed on more
than a quarter of the steps, so that two runs that both ignored the count would not pass.

The counting sort case reads its element count from the device and its key range from a host
provider that cycles through four ranges, so one, two and three digit passes all run.

## Not carried over from SushiRuntime

The cases that pinned a mechanism of the runtime itself stayed there, on a test-local kernel:
the resolution of a bare `LateCount`, a `sized_from_device` launch, and the refusal of a device
count slot that is not host-resident, which `LateCount::from_device` stops the process on
before a primitive is called.

# Unit tests

The unit tests that need more than one module, each a GoogleTest translation unit built on the
fixtures in `tests/common/`. The tests of one module are in that module's `tests/` folder,
with a README of their own. The source comments state what a test checks. This document holds why a test
is written the way it is: the defect it pins, the number it was measured against, the quirk of
the machine it ran on. Shared reasoning comes first, then one section per file in path order.

A test builds its operation views on the fixture's engine, as in `BLASOps(*engine)`, or on a
`GraphProbe` when it asserts on the recorded graph; `tests/common/README.md` describes both. A
refusal at record time is asserted as `std::invalid_argument`, and a mode or device refusal as
`SushiBLAS::UnsupportedExecution`.

The files this page has no section for are described by the table below and by their own source
comments. The table gives the `@brief` of each file, with an operation named by its view. The files added on 2026-10-07, and the changes
made to the older ones that day, had not been built or run when this page was written.

## Files

| File | Tests |
| --- | --- |
| `test_channel_bias_epilogue.cpp` | Tests ChannelBiasEpilogue's stride pair through Dispatch::execute_gemm |
| `test_dispatch_complex.cpp` | Tests which elementwise and logic operations compute on COMPLEX32 and which refuse it |
| `test_dispatch_seam.cpp` | Tests the public Dispatch seam with functors written outside the library |
| `test_empty_tensors.cpp` | Tests what elementwise, non-linear, logic and reduction ops do with an empty tensor |
| `test_functor_reference.cpp` | Tests that the functor seams give the same result on the host path as on the SYCL path |
| `test_integer_dtypes.cpp` | Tests which operations serve INT32 and INT64 tensors and which refuse them |
| `test_math_validation.cpp` | Tests the operand checks and numeric edge cases of the elementwise, nonlinear and reduction ops |
| `test_packed_product.cpp` | Tests the packed product kernel against the reference GEMM under injected blockings |
| `test_reference_mode_guard.cpp` | Tests that ForceReference refuses device-only USM and runs on host-reachable USM |
| `test_runtime_shapes.cpp` | Tests create_tensor and reshape with dimensions held in a std::vector |
| `test_scratch_buffer.cpp` | Tests ScratchBuffer, the lease on one pooled device scratch block |
| `test_sparse_builder.cpp` | Tests the sparse vectors and CSR matrices SparseBuilder makes from host data |
| `test_sparse_types.cpp` | Tests IndexBase, the two sparse operands, the kernel index views and accumulator_bytes |
| `test_spatial_random_io_validation.cpp` | Tests the record-time checks of spatial geometry, random fills and IO |
| `test_structured_operands.cpp` | Tests the record-time checks on a band, a packed and a scalar operand |

## Shared reasoning

### Expectations are worked out by hand

Where a file says its expectations are hand-computed, they do not come from another path in
this library. A differential oracle agrees with the code under test in every case where both
are wrong the same way. That is how an inverted `ColumnMajor` addressing arm in GEMM survived
for months: the SYCL kernel and the reference kernel had the identical defect, so the
differential tests agreed.

### Comparing two kernels under the two forced modes

The `TheSyclKernelAgreesWithTheReferenceKernel` tests fill two buffers under
`ExecutionMode::ForceSycl` and `ExecutionMode::ForceReference` and compare them. That is how
the library checks kernel agreement: `ExecutionMode` has no mode that runs both, and
`Engine::set_mode` is `noexcept`.

A BLAS kernel that sums is not compared this way. The reference sums in `double` and the SYCL
kernel in the element type, FLOAT32 for HALF, so the BLAS parity tests measure each against a
`TruthSum` within a rounding bound; see `blas_truth.hpp` in `tests/common/README.md`. A kernel
that only moves elements is still compared bit for bit.

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
SYCL instantiation would make every ForceSycl against ForceReference comparison of a fused
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
it functors this file wrote, which is the point of promoting those five headers out of `source/`.
It is the in-tree half of the proof. The out-of-tree half is `tests/package/main.cpp`, which
does the same against a `find_package()`'d install.

`float_bits` serves the determinism checks; see "Comparing floats by bit pattern".

`RealOnlyDomainRejectsComplexAtRecordTime`: the default domain is `Domain::Real`, so a functor
that was only ever meant for real scalars is rejected before it can be handed a complex operand.
It is not compiled for one, and it is not silently skipped as it once was.

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

`CopyMovesIntegerElements`: `copy` is the one BLAS operation that takes `Domain::All`, because
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


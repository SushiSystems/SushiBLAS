# Tests of umbrella

The tests of the `umbrella` module, built into `sushiblas_umbrella_test_objects` and linked
into `sushiblas_library_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`.
They belong to this module because each crosses several operation families, and this
module is the one that depends on all of them. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

Each `benchmark_*.cpp` file is a program of its own, built when `SB_BUILD_BENCHMARKS` is on;
the benchmark section of `tests/README.md` describes them.

| File | Tests |
| --- | --- |
| `benchmark_elementwise.cpp` | Times add, in-place exp and in-place relu and prints GB/s per element count |
| `test_SushiBLAS/test_dependency_stress.cpp` | Tests the dependency tracker under shard collisions |
| `test_SushiBLAS/test_dispatch_complex.cpp` | Tests which elementwise and logic operations compute on COMPLEX32 and which refuse it |
| `test_SushiBLAS/test_dispatch_checks.cpp` | Tests, on values and without a device, the record-time domain and parameter-capacity checks of the public Dispatch seam |
| `test_SushiBLAS/test_dispatch_seam.cpp` | Tests the public Dispatch seam with functors written outside the library |
| `test_SushiBLAS/test_empty_tensors.cpp` | Tests, on a host engine, what elementwise, non-linear, logic and reduction ops do with an empty tensor |
| `test_SushiBLAS/test_framework_sim.cpp` | Tests a ten-layer forward pass and 24 independent models through the engine |
| `test_SushiBLAS/test_integer_dtypes.cpp` | Tests which operations serve INT32 and INT64 tensors and which refuse them, and that every operation refuses UINT32 |
| `test_SushiBLAS/test_math_validation.cpp` | Tests the device-only operand refusal, the mode independence of operand errors and the softplus range |
| `test_SushiBLAS/test_pack_ordered_sum.cpp` | Tests that ordered_sum sums the same segment tensor of several packs into another pack; built with `SB_ENABLE_DISTRIBUTED` |
| `test_SushiBLAS/test_operand_refusals.cpp` | Tests, on a host engine, which operands the elementwise, nonlinear and reduction ops refuse at record time |
| `test_SushiBLAS/test_operand_lifetime.cpp` | Tests that a recorded task keeps its operands and its scratch allocated until it has run |
| `test_SushiBLAS/test_regression.cpp` | Tests one fast, deterministic sentinel per fixed defect |
| `test_SushiBLAS/test_spatial_random_io_validation.cpp` | Tests the device pooling extremes, the degenerate random fills the IO round trip and its refusals |
| `test_SushiBLAS/test_spatial_random_refusals.cpp` | Tests, on a host engine, which spatial geometries and random fills are refused at record time |
| `test_SushiBLAS/test_storage_offset.cpp` | Tests every operation family on views at a non-zero storage offset |
| `test_SushiBLAS/test_task_dependencies.cpp` | Tests that one recorded task of each recording seam runs and computes its result |

## test_SushiBLAS/test_storage_offset.cpp

Every operation family runs against tensors that are views at a non-zero `storage_offset`, not
at the base of their allocation. This is the shape a memory planner produces: one arena
allocation with many tensors packed into it, each handed to an operation as a view at an
offset. An operation that captures the allocation base of the storage drops the offset, so a
view aliases the start of its arena, reads the wrong inputs and writes over whatever else lives
there. These tests guard against that.

Each test asserts two things, and the second carries the weight. That the result is correct
often passes by accident: with a single tensor per allocation, offset 0 and "ignore the offset"
are the same address. That the poison bytes before and after the view survive untouched cannot
pass by accident, because an operation that ignores the offset writes over them.

The tests run in family order: BLAS levels 1 to 3, elementwise, nonlinear, logic, reductions,
random, the two tests about what an offset counts, the alignment rows and last one test
about `device`.

`OffsetIsCountedInElementsNotBytes`: the unit of `storage_offset` is a decision, not an
accident, and it is the one that keeps every view naturally aligned. An element count scaled by
the element width can never land an fp64 view on a 4-byte boundary. The test pins it so that
"offsets are bytes" cannot be introduced later and pass the family tests, which use one dtype
at a time and would not notice.

The region of a view and the overlap of two regions need no device. `Small_Tensor` in
`tests/data/tensor/test_tensor/test_tensor.cpp` holds `TheRegionsOfDisjointViewsDoNotOverlap` and
its two siblings: two tensors packed into one arena must not be reported as touching the same
memory, or every operation in the graph serialises behind every other one.

## test_SushiBLAS/test_dispatch_complex.cpp

Before the dispatch seam was made public, the elementwise and logic families handled HALF,
FLOAT32 and FLOAT64 and ended their dtype switch with `default: break;`. A COMPLEX32 tensor
therefore reached a command group that enqueued no kernel at all: `add()` "succeeded", the
graph reported no error, and the output tensor was left holding whatever it held before. Every
case in this file is one of those silently dropped paths, now resolved one way or the other:
computed where the operation has a complex meaning, thrown for where it does not.

`MulOnComplex32`: the imaginary cross terms of the two products are exactly what
the old no-op path could never have produced, and what a real-only kernel would drop.

The rejected operations threw nothing before; they enqueued nothing and returned. Each now
fails at record time, at the call site, before a task reaches the graph.

The `Less`, `LessEqual`, `Greater` and `GreaterEqual` rows of `RefusesComplex32AtRecordTime`: C is not an ordered field. A lexicographic answer to
these comparisons would be a well-formed answer to a question nobody asked, so the test holds the
refusal.

The `Sin`, `Cos`, `Tan` and `Pow` rows of `RefusesComplex32AtRecordTime`: trigonometry and `pow` do have complex extensions, but
SushiBLAS does not implement them yet. The contract is that "not implemented" is observable:
these report the gap and do not leave the tensor untouched while claiming success.

## test_SushiBLAS/test_dispatch_seam.cpp

This translation unit reaches `SushiBLAS::Dispatch` through installed public headers and hands
it functors this file wrote, which is the point of promoting those five headers out of `source/`.
It is the in-tree half of the proof. The out-of-tree half is `tests/package/main.cpp`, which
does the same against a `find_package()`'d install.

`float_bits` serves the determinism checks; see "Comparing floats by bit pattern".

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

`ATypedTaskReportsTheFamilyItsCallerDeclared`: the family string is what a caller outside BLAS
needs from this seam. An operation recorded by the spatial domain must not be reported as a
BLAS operation when its dtype is refused. It is pinned here and not in the spatial tests
because the guarantee belongs to the seam and not to any one of its callers.

## test_SushiBLAS/test_integer_dtypes.cpp

A dtype that is half-supported is worse than one that is not supported at all, because
"accepted" reads as "handled". So this file is deliberately lopsided: a handful of tests that
the supported operations give the right answer, and a sweep asserting that every operation
outside that list throws at record time. The sweep is the part that has to stay green
as operations are added.

`CopyMovesIntegerElements`: `copy` is the one BLAS operation that takes `Domain::All`, because
it does no arithmetic to what it moves.

`Medium_IntegerBinaryLogic.ServesIntegerTensors` runs the comparisons and the boolean
operations as rows: the results are 1 and 0, which is what an integer element is for.

`ArgmaxSearchesAnIntegerTensor` and `ArgminSearchesAnIntegerTensor`: an ordering exists over
integers, so the operations are defined, and the answer is still an INT64 index and not an element
of the input's dtype.

`ArgmaxIsExactPastTheFloatMantissa` uses an index past 2^24, which is where the old
`static_cast<T>(idx)` into a FLOAT32 result started returning a neighbour instead of the
answer. 16,777,217 is the first integer a float cannot represent.

`Medium_IntegerRefusal.ThrowsBeforeItRecordsATask` is the sweep, one row per operation. Every
operation family outside the supported list must refuse an integer operand at record time,
before anything is queued.

## test_SushiBLAS/test_task_dependencies.cpp

A task body receives the events of the tasks it depends on and must order its own device work
behind them. No test checks that at run time: the runtime's plan view holds no body, so a test
cannot hand one a dependency that has not finished. `tools/layering/check_engine_seams.py` checks the
rule in the source, under `rule_task_dependencies`.

## test_SushiBLAS/test_operand_lifetime.cpp

A task holds raw pointers into its operands, and the recorder retains the storage behind them
(`OperandSet`, `Internal::StorageKeepAlive`). The fixture counts the runtime's live USM
allocations around the record and execute calls. `ATaskKeepsItsInputAllocatedUntilItHasRun`
destroys an input tensor after the record call and checks its allocation lives until the task
ran. `AnEpilogueOperandIsKept` does the same for a tensor declared through
`Dispatch::gemm_epilogue`.

## test_SushiBLAS/test_regression.cpp

### SyrkAcceptsHalfDtype

SYRK's oneMKL-backed implementation threw for `sycl::half`, because oneMKL has no half-precision
SYRK. The SB-2 port to the tiling kernel lifted that restriction. This test is what catches a return of the restriction at the umbrella. It checks the lower triangle of A * A^T for a 4 x 3 integer A.

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

### Fp64AxpyIsNotNarrowedToFloat and Fp64ScalIsNotNarrowedToFloat

`axpy`, `scal` and `gemv` declared their scalars `float` while `gemm` declared `double`. An fp64
computation received a float-rounded scalar: about 1e-7 relative error and no diagnostic.

The test uses alpha = 1 + 2^-30. A double holds that value exactly and a float rounds it to
exactly 1.0, so the right answer and the narrowed answer are different numbers and not two close
ones. `scal` has its own test because it had the same defect in the other Level 1
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

### GerDeclaresItsAccumulatorRead, GemvWithBetaDeclaresItsAccumulatorRead and GemvWithoutBetaStaysAPureWrite

`ger`, and `gemv` with beta != 0, declared their accumulator write-only while reading it. The
order still came out right because the tracker inserts a write-after-write edge anyway, so the
defect was latent and no behavioural test could see it. The test reads the recorded graph
instead: it records through a `GraphProbe` (`modules/engine/graph_probe/include/SushiBLAS/graph_probe/graph_probe.hpp`), whose
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
ln 6 was expected: the `max` reduction raced the bias add feeding it, and the BLAS `scal` beat
the `sum` reduction to the loss.

The test asserts on the recorded graph and not on the answer. Without the edge the two tasks may
still run in the right order by luck, and a test that passes by luck guards nothing. The
`dependency_count` of the second task decides it. The two views are disjoint and the consumer's operand is
not at offset zero, which is the shape a memory planner produces and the one where a bare-pointer
region and a real region disagree. `TheSumOfTheProductsIsTheSumOfTheSquares` runs the same pair of ops once more
and compares its sum, for a reader who wants to see the behaviour as well.

### scalar_of

The nrm2 sentinel reads the one FLOAT32 word of its result to the host and compares it with
`EXPECT_NEAR`, because the answer is a single number and not a tensor to compare.

### Nrm2UsesTheModulusOfAComplexElement

Level 1 nrm2 took its per-element magnitude from `abs1`, which is BLAS's |Re| + |Im|. That is the
right magnitude for xASUM and IxAMAX and the wrong one for a 2-norm, which needs Re^2 + Im^2. The
cross term made every complex norm too large, by sqrt(2) for a vector of (1 + i). The test uses
3 + 4i: the modulus is 5 and `abs1` would give 7.

## test_SushiBLAS/test_dispatch_checks.cpp

`RealOnlyDomainRejectsADtypeOutsideIt`: the default domain is `Domain::Real`, so a functor
that was only ever meant for real scalars is rejected before it can be handed a complex operand.
It is not compiled for one, and it is not silently skipped as it once was.

## test_SushiBLAS/test_dependency_stress.cpp

### ShardCollisionStress

SushiRuntime's dependency tracker files an allocation under one of 1024 shards, picked by a
hash of its base address. With 3000 independent GEMMs over 9000 distinct tensors, many tensors
share a shard, and no task may gain a dependency from that. Each A is filled with a constant that depends on its
index and each B with 0.5, so the test knows what every C must hold.

## test_SushiBLAS/test_framework_sim.cpp

### DeepForwardPass

The graph is the forward pass of a deep network, ten layers of 512 x 1024 activations:

```
H1  = X * W1
H2  = H1 * W2
...
H_n = H_{n-1} * W_n
```

A ReLU follows every GEMM except the last. The weights are 0.01 in even columns and -0.01 in
odd ones, so the activations do not grow out of range across ten layers.

### ParallelBatchedInference

The graph serves 24 independent models at once, each a GEMM followed by a ReLU. It tests whether
the scheduler spreads independent tasks across all workers. The test was written against 12
workers, and 24 models is more models than threads.

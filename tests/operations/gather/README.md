# Tests of gather

The tests of the `gather` module, built into `sushiblas_gather_test_objects` and linked
into `sushiblas_operations_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `index_support.hpp` | Gives the tensor builders, the host oracles of the four operations and the refusal check both test files use |
| `test_gather/test_index_along.cpp` | Tests RowIndexOps::take_along and put_along against a host loop in both execution modes |
| `test_gather/test_index_rows.cpp` | Tests RowIndexOps::take_rows and add_rows against a host loop in both execution modes |
| `test_index_operands.cpp` | Tests that the record-time checks of the row index operations accept a fit operand and refuse a misfit one, on a host engine |
| `benchmark_index_rows.cpp` | Times RowIndexOps beside the sparse products it replaces and prints GB/s per table size |

## Shape of the tests

Each promise runs on HALF, FLOAT32 and FLOAT64, on INT32 values where the operation moves
elements without adding them, and on INT32 and INT64 indices: eight combinations for `take_rows`,
`take_along` and `put_along`, six for `add_rows`. A suite is a typed suite over `Combo<Element,
Index>`, and a case on a device without the element type's aspect is skipped.

Every value is a small integer, so every product and every sum is exact in HALF as well, the
oracle is a host loop in `double`, and the comparison is bitwise equality. No tolerance is written.
The loop is `host_take_rows`, `host_add_rows`, `host_take_along` or `host_put_along` in
`index_support.hpp`; each is written from the formula in the header comment of the operation and
shares no code with a kernel.

Each operation runs under `ForceSYCL` and under `ForceReference`, and both results are compared
with the same expected vector. Under `ForceReference` the host kernel runs and under `ForceSYCL`
the device one does.

## test_gather/test_index_rows.cpp

`AddRowsWithBetaZeroEqualsTheHostLoopAndSkipsTheTable` fills the table with NaN. A kernel that
read it would leave a NaN in every element, which no integer sum equals.

`AddRowsOfARepeatedIndexGivesTheSameBytesOnTenRuns` uses values no element type holds exactly and
six rows that select one table row, so a sum whose order changed from one run to the next would
change its last bits. It compares ten runs of each path against the first run of that path.

`AddRowsIntoAColumnSliceLeavesItsParentUnchanged` writes through a view whose row stride
is wider than its extent and whose first element is two elements into the parent. Every element
outside the view must keep its value, and the HALF workspace is indexed through the same strides.

`TakeRowsReadsATableViewAtANonZeroStorageOffset` reads the upper half of a table through a view
that starts at row V of its parent.

The positions of `K_OUTSIDE` hold -1 and V between valid ones. The expected rows are zeros for the
two and the table rows for their neighbours.

## test_gather/test_index_along.cpp

`TakeAlongAfterPutAlongReturnsThePickedValues` records both operations on one index in one graph,
so the second reads what the first wrote.

`PutAlongIntoAViewLeavesTheRowsAroundItUnchanged` writes the middle rows of a taller tensor
through a view at a non-zero storage offset; the rows above and below keep their sentinel.

## Refusals

Each record-time refusal is one test on FLOAT32 tensors and checks that the message names the
operation and the operand. An element type outside the operation's family is refused without
naming an operand, so those tests check the operation alone.

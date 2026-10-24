# Tests of index

The tests of the `index` module, built into `sushiblas_index_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_index_ops.cpp` | Tests IndexOps::for_each with a fixed count and with a count a previous node writes |

## test_index_ops.cpp

The scatter runs a chain of 33 vertices in two colours, the even edges and the odd edges. Inside a
colour no two edges share a vertex; across the colours they do, so only the order of the two
calls makes the result right. Weights and start values are multiples of 0.25, so every sum is
exact and a bit-for-bit match proves the order, not the rounding. The expectation is a host loop
written beside the test, and a second test compares the `ForceReference` run to it, so the
`ForceSYCL` comparison does not rest on two runs that agree by sharing a defect.

The late-count tests record two nodes: one writes the count, one marks indices. The count tensor
starts at capacity plus one, so a task that read it before the first node wrote it would mark
everything. A count of 0 and a count equal to the capacity run on both paths. A count above the
capacity and a negative count fail the run on `ForceSYCL`, where the runtime's dynamic node reports
them; the host body's refusal is tested in `test_index_task.cpp`. The ordering cases assert on
`dependency_count`, not on a value, on both paths. The determinism cases read
`PlanNode::determinism` from the runtime's plan, for a fixed and a late node, and a node that
states no class reads `Unstated`.

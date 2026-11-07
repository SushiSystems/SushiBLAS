# Tests of index

The tests of the `index` module, built into `sushiblas_index_test_objects` and linked
into `sushiblas_operations_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_index_ops/test_graph.cpp` | Tests IndexOps::for_each with a fixed count and with a count a previous node writes, against the runtime |
| `test_index_ops/test_recording.cpp` | Tests what IndexOps::for_each records and refuses, on a host engine with no device |

## test_index_ops/test_graph.cpp

The scatter runs a chain of 33 vertices in two colours, the even edges and the odd edges. Inside a
colour no two edges share a vertex; across the colours they do, so only the order of the two
calls makes the result right. Weights and start values are multiples of 0.25, so every sum is
exact and a bit-for-bit match proves the order, not the rounding. The expectation is a host loop
written beside the test, and a second test compares the `ForceReference` run to it, so the
`ForceSYCL` comparison does not rest on two runs that agree by sharing a defect.

The late-count tests are tables of rows, one test body per promise. A row names the execution
mode, the type of the count tensor (`Int32` or `UInt32`), who sets the count (a recorded node or
the host) and its value. The count tensor starts at capacity plus one when a node writes it, so a
task that read it before the node wrote it would mark everything. A count of 0 and a count equal
to the capacity run on both paths. A count above the capacity, a negative count and a count above
a capacity of zero fail the run on `ForceSYCL`, where the runtime's dynamic node reports them; the
host body's refusal is tested in `test_index_task.cpp`. The refused count tensors in shared memory (an `Int32` or `UInt32` count) are one table over
both modes; residency is the boundary. The ordering cases assert on `dependency_count`, not on a
value, since the graph plan owns the order. The determinism case reads `PlanNode::determinism`
from the runtime's plan, for a fixed and a late node; the default `Unstated` class is tested in
`tests/engine/engine/test_operation_options.cpp`, and so is the refusal of an empty name.

## test_index_ops/test_recording.cpp

The tests run `Small_` on a `HostEngineTest`: they read what the host sink kept. They cover the
name, operation id and determinism class of a recorded task, the one read and one write of an
update, the refusal of a float or `Int64` count tensor (also at a capacity of zero), the refusal of
a task that writes its own count, the overlapping declarations (also at a count of zero), and that
a count of zero records nothing. A refusal records nothing.

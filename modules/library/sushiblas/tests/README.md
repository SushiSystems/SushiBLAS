# Tests of sushiblas

The tests that run every operation family, built into `sushiblas_sushiblas_tests` by the
module's `CMakeLists.txt`. They belong to this module because it is the one that depends on
every family. The source comments state what a test checks; a section below holds why a test
is written the way it is. `tests/unit/README.md` holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_storage_offset.cpp` | Tests every operation family on views at a non-zero storage offset |

## test_storage_offset.cpp

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
random, the two tests about what an offset counts, and last three about alignment and one
about `device`.

`OffsetIsCountedInElementsNotBytes`: the unit of `storage_offset` is a decision, not an
accident, and it is the one that keeps every view naturally aligned. An element count scaled by
the element width can never land an fp64 view on a 4-byte boundary. The test pins it so that
"offsets are bytes" cannot be introduced later and pass the family tests, which use one dtype
at a time and would not notice.

`DisjointViewsOfOneAllocationDoNotAlias` is what the memory planner in the layer above needs:
two tensors packed into one arena must not be reported as touching the same memory, or every
operation in the graph serialises behind every other one.

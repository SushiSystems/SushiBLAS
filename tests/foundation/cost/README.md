# Tests of cost

The tests of the `cost` module, built into `sushiblas_cost_test_objects` and linked
into `sushiblas_foundation_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share.

| File | Tests |
| --- | --- |
| `compile_cost_shape.cpp` | Compile check, never run: `cost_shape.hpp` stands alone; the shapes are tested through `test_cost_table.cpp` |
| `test_cost.cpp` | Tests that `Cost` separates a stated cost from an absent one and that a complex type is counted in its real class |
| `test_cost_table.cpp` | Tests the rule of each family, the saturation of an overflowing product, the order and uniqueness of the rows and the reasons an operation states no cost |
| `test_composite_cost.cpp` | Tests the cost of a softmax |

The expected values are the conventions of [`modules/foundation/cost/README.md`](../../../modules/foundation/cost/README.md) worked by hand, so a change to a weight shows as a failing number.

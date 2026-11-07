# Tests of logic

The tests of the `logic` module, built into `sushiblas_logic_test_objects` and linked
into `sushiblas_operations_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `compile_logic_log_category.cpp` | Checks at compile time that the log category names its module and the project; compiled, never run |
| `test_dispatch.cpp` | Tests the logic predicates on the host, and a unary and a binary logic task, one instance per execution mode |
| `test_logic/test_all.cpp` | Tests LogicOps::all against fixed expected values on one four-element input |
| `test_logic/test_any.cpp` | Tests LogicOps::any against fixed expected values on one four-element input |
| `test_logic/test_equal.cpp` | Tests LogicOps::equal against fixed expected values on one four-element input |
| `test_logic/test_greater.cpp` | Tests LogicOps::greater against fixed expected values on one four-element input |
| `test_logic/test_greater_equal.cpp` | Tests LogicOps::greater_equal against fixed expected values on one four-element input |
| `test_logic/test_less.cpp` | Tests LogicOps::less against fixed expected values on one four-element input |
| `test_logic/test_less_equal.cpp` | Tests LogicOps::less_equal against fixed expected values on one four-element input |
| `test_logic/test_logical_and.cpp` | Tests LogicOps::logical_and against fixed expected values on one four-element input |
| `test_logic/test_logical_not.cpp` | Tests LogicOps::logical_not against fixed expected values on one four-element input |
| `test_logic/test_logical_or.cpp` | Tests LogicOps::logical_or against fixed expected values on one four-element input |
| `test_logic/test_logical_xor.cpp` | Tests LogicOps::logical_xor against fixed expected values on one four-element input |
| `test_logic/test_not_equal.cpp` | Tests LogicOps::not_equal against fixed expected values on one four-element input |
| `test_logic/test_where.cpp` | Tests LogicOps::where against fixed expected values on one four-element input |

Each test is named for the promise it holds, not for a dtype or a direction. The predicates are host
functions and run `Small_` with no fixture. The tests that record on the fixture's engine stay
`Medium_`: the engine reaches the process-wide runtime and a device queue, and the fixture offers no
host tensor.

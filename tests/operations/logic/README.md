# Tests of logic

The tests of the `logic` module, built into `sushiblas_logic_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_logic/test_all.cpp` | Tests LogicOps::all against fixed expected values |
| `test_logic/test_any.cpp` | Tests LogicOps::any against fixed expected values |
| `test_logic/test_equal.cpp` | Tests LogicOps::equal against fixed expected values |
| `test_logic/test_greater.cpp` | Tests LogicOps::greater against fixed expected values |
| `test_logic/test_greater_equal.cpp` | Tests LogicOps::greater_equal against fixed expected values |
| `test_logic/test_less.cpp` | Tests LogicOps::less against fixed expected values |
| `test_logic/test_less_equal.cpp` | Tests LogicOps::less_equal against fixed expected values |
| `test_logic/test_logical_and.cpp` | Tests LogicOps::logical_and against fixed expected values |
| `test_logic/test_logical_not.cpp` | Tests LogicOps::logical_not against fixed expected values |
| `test_logic/test_logical_or.cpp` | Tests LogicOps::logical_or against fixed expected values |
| `test_logic/test_logical_xor.cpp` | Tests LogicOps::logical_xor against fixed expected values |
| `test_logic/test_not_equal.cpp` | Tests LogicOps::not_equal against fixed expected values |
| `test_logic/test_where.cpp` | Tests LogicOps::where against fixed expected values |

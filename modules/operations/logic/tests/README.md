# Tests of logic

The unit tests of the `logic` module, built into `sushiblas_logic_tests` by the
module's `CMakeLists.txt`. The source comments state what a test checks; a section below
holds why a test is written the way it is. `tests/unit/README.md` holds the reasoning the
tests share, and `modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_all.cpp` | Tests LogicOps::all against fixed expected values |
| `test_any.cpp` | Tests LogicOps::any against fixed expected values |
| `test_equal.cpp` | Tests LogicOps::equal against fixed expected values |
| `test_greater.cpp` | Tests LogicOps::greater against fixed expected values |
| `test_greater_equal.cpp` | Tests LogicOps::greater_equal against fixed expected values |
| `test_less.cpp` | Tests LogicOps::less against fixed expected values |
| `test_less_equal.cpp` | Tests LogicOps::less_equal against fixed expected values |
| `test_logical_and.cpp` | Tests LogicOps::logical_and against fixed expected values |
| `test_logical_not.cpp` | Tests LogicOps::logical_not against fixed expected values |
| `test_logical_or.cpp` | Tests LogicOps::logical_or against fixed expected values |
| `test_logical_xor.cpp` | Tests LogicOps::logical_xor against fixed expected values |
| `test_not_equal.cpp` | Tests LogicOps::not_equal against fixed expected values |
| `test_where.cpp` | Tests LogicOps::where against fixed expected values |

# Tests of reduction

The unit tests of the `reduction` module, built into `sushiblas_reduction_tests` by the module's `CMakeLists.txt`. The source comments state what a test checks. `tests/unit/README.md` holds the reasoning the tests share, and `tests/common/README.md` describes the fixtures.

| File | Tests |
| --- | --- |
| `test_reduce_plans.cpp` | Tests the plan-taking forms of Reduce::reduce and Reduce::reduce_indexed |
| `test_reduce_segments.cpp` | Tests Reduce::reduce_segments in both launch shapes and the axis fold that runs on it |

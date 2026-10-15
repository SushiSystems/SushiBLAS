# Tests of reference

The unit tests of the `reference` module, built into `sushiblas_reference_tests` by the module's `CMakeLists.txt`. The source comments state what a test checks. `tests/unit/README.md` holds the reasoning the tests share, and `modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_reference_accumulator.cpp` | Tests the type and the running sum a reference kernel accumulates in |
| `test_serial_loops.cpp` | Tests the serial host loops and folds against the launch shapes and accumulators they mirror |

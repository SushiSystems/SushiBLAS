# Tests of blas_test_oracle

The tests of the `blas_test_oracle` module, built into `sushiblas_blas_test_oracle_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_blas_flags.cpp` | Tests that each flag helper maps true and false onto the two BLAS enumerators |
| `test_blas_structured.cpp` | Tests the layout, band, triangle-band and packed conversions against hand-built storage |
| `test_blas_truth.cpp` | Tests that TruthSum recovers the rounding a double sum loses, and the rounding bound |
| `test_dense_oracle.cpp` | Tests the dense oracle loops against hand-computed matrices and vectors |

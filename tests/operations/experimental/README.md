# Tests of experimental

The tests of the `experimental` module, built into `sushiblas_experimental_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `lapack/test_linalg.cpp` | Tests the shape of the quarantined LinalgOps declaration, which no source implements |
| `signal/test_transforms.cpp` | Tests the shape of the quarantined TransformsOps declaration, which no source implements |

# Tests of launch

The tests of the `launch` module, built into `sushiblas_launch_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_device_policy.cpp` | Tests the cache blocking, the group sizes, the 1D grid and the vector width of a profile |
| `test_launch_1d.cpp` | Tests the shape-taking launch primitives and the local tile fit check |

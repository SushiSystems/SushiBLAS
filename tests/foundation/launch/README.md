# Tests of launch

The tests of the `launch` module, built into `sushiblas_launch_test_objects` and linked
into `sushiblas_foundation_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_device_policy.cpp` | Tests the cache blocking, the group sizes, the 1D grid, the vector width, the vendor and the architecture of a profile |
| `test_gemm_tile_pin.cpp` | Tests the read of the GEMM tile pin from a variable's value |
| `test_kernel_variant.cpp` | Tests that a kernel variant note reaches the observer that is set, and only that one |
| `test_launch_1d.cpp` | Tests the shape-taking launch primitives and the local tile fit check |
| `test_launch_log_category.cpp` | Tests that `K_LAUNCH_LOG_CATEGORY` names the module |
| `test_submitted_events.cpp` | Tests that `SubmittedEvents` keeps every event of a body in order and converts to the last |

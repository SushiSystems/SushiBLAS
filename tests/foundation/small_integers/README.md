# Tests of small_integers

The tests of the `small_integers` module, built into `sushiblas_small_integers_test_objects` and linked
into `sushiblas_foundation_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_small_integers.cpp` | Tests that the operands are small integers, repeat for a seed and keep a unit diagonal |

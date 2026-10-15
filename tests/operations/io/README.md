# Tests of io

The tests of the `io` module, built into `sushiblas_io_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_io/test_host_transfer.cpp` | Tests IO's host transfer calls, load ordering, the .npy reader and the text of a tensor |
| `test_io/test_regression.cpp` | Pins that IO refuses a file it cannot read whole and reaches device storage by copy |
| `test_io_status.cpp` | Tests that every IOStatus has its own name and that Ok is the zero value |

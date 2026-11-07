# Tests of io

The tests of the `io` module, built into `sushiblas_io_test_objects` and linked
into `sushiblas_operations_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_file_remover.cpp` | Tests that the disk and in-memory `FileRemover` adapters delete a file they hold and refuse one they do not |
| `test_file_header.cpp` | Tests, on byte buffers and text, the .npy prefix, dictionary and match checks and the .sushi header checks |
| `test_io/test_host_text.cpp` | Tests the text of an empty tensor and the layout refusal of load_npy, `Small_` on a host engine |
| `test_io/test_host_transfer.cpp` | Tests IO's host transfer calls, loads after a run, the .npy round trip and the text of a tensor |
| `test_io/test_regression.cpp` | Pins that IO refuses a file it cannot read whole and reaches device storage by copy |
| `test_io_status.cpp` | Tests that every IOStatus has its own name and that Ok is the zero value |
| `test_partial_file.cpp` | Tests, `Small_` on an in-memory `FileRemover` and a scoped log capture, that `remove_partial_file` reports a removed file quietly and warns when it cannot |

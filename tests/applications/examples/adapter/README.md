# Tests of the adapter example

The tests of the adapter under `applications/examples/adapter/`, built into
`sushiblas_adapter_tests` by `sushiblas_add_application_test_program()` in
`cmake/SushiBLASTests.cmake`. The program compiles `sushiblas_adapter.cpp` with the tests and
sees the adapter's headers as the adapter does. The source comments state what a test checks;
[`tests/README.md`](../../../README.md) holds the reasoning the tests share.

| File | Tests |
| --- | --- |
| `test_host_program.cpp` | Tests the host program's status names and a multiply-add through the SushiBLAS backend |
| `test_status_mapping.cpp` | Tests that the adapter maps every runtime error and every exception to a host status |

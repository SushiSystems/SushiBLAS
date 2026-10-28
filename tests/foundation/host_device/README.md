# Tests of host_device

The tests of the `host_device` module, built into `sushiblas_host_device_test_objects` and linked
into `sushiblas_foundation_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_host_device_test.cpp` | Tests that HostDeviceTest hands a test the host CPU queue |
| `test_host_queue.cpp` | Tests that host_queue returns one in-order queue on a CPU device, or none |
| `test_shared_array.cpp` | Tests that SharedArray aligns and fills its elements and that a kernel reaches them |

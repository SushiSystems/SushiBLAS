# Tests of device_ceilings

The tests of the `device_ceilings` program, built into `sushiblas_device_ceilings_tests` by
`sushiblas_add_application_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../../README.md) holds the reasoning the tests share.

| File | Tests |
| --- | --- |
| `test_ceilings_options.cpp` | Tests the defaults, every option and every refusal of the command line |
| `test_ceilings_statistics.cpp` | Tests the median and the spread of the timed runs |
| `test_span_observer.cpp` | Tests that the span of a run is the first start to the last end |
| `test_device_meter.cpp` | Tests that the meter gives positive finite rates on the device the tests run on |

The program itself decides nothing and is not a test. These tests cover the parts that can be wrong without a GPU; the numbers that count come from running `device_ceilings` on the device.

# Tests of profiling

The tests of the `profiling` module, built into `sushiblas_profiling_test_objects` and linked
into `sushiblas_engine_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share.

| File | Tests |
| --- | --- |
| `test_ceilings_file.cpp` | Tests that the reader accepts what the writer writes and refuses a wrong format, version, member, dtype and number |
| `test_blas_environment.cpp` | Tests the facts the contributor states, the ceilings of a device with a file, a smoke-only file, a missing file and a refused file |
| `test_profiler_parts.cpp` | Tests that `register_profiler_parts` adds the contributor once, and, under a session on the shared runtime, that a GEMM span carries its cost and dtype and that the `roofline` table judges it against a ceiling |
| `compile_ceilings.cpp` | Checks at compile time that a default bandwidth ceiling and method state nothing a reader would accept as measured; compiled, never run |
| `compile_profiling_log_category.cpp` | Checks at compile time that the log category names its module and the project; compiled, never run |

The end-to-end suites write a ceilings file for the first device of the test runtime into a temporary folder and start one SushiProfiler session, so they run one at a time.

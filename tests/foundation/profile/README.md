# Tests of profile

The tests of the `profile` module, built into `sushiblas_profile_test_objects` and linked
into `sushiblas_foundation_tests` by `sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share.

| File | Tests |
| --- | --- |
| `test_profile.cpp` | Tests that every `SB_` macro is usable in one scope and that a value only a macro reads is not unused |
| `test_profile_gate.cpp` | Tests that a zone is compiled in or out by the level of its translation unit, and that a unit with the profiler off compiles every macro |
| `test_profile_keys.cpp` | Tests that each name the library writes into a record is listed once |

`profile_level_one.cpp` sets level 1 and `profile_off.cpp` turns the profiler off, each by undefining the generated setting before it includes `profile.hpp`, so the gate is tested across translation units of different levels. Neither is a test.

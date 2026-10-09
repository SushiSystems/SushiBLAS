# Tests

The tests that cross module boundaries, the helpers every test shares, the consumer of an
installed package and the benchmark programs. A test of one module lives in that module's
`tests/` folder, under `modules/<tier>/<module>/tests/`.

| Folder | Holds | Built by |
| --- | --- | --- |
| `unit/` | One brick or one check that needs more than one module; see [its README](unit/README.md) | `SB_BUILD_TESTS` |
| `integration/` | Several operations through the public views; see [its README](integration/README.md) | `SB_BUILD_TESTS` |
| `regression/` | One test per defect that was fixed; see [its README](regression/README.md) | `SB_BUILD_TESTS` |
| `common/` | `test_main.cpp` and the headers the tests share; see [its README](common/README.md) | with every test program |
| `package/` | A consumer of an installed SushiBLAS, configured on its own; see [its README](package/README.md) | `sb test --suite package` |
| `benchmark/` | Programs that print timings and never pass or fail; see [its README](benchmark/README.md) | `SB_BUILD_BENCHMARKS` |

## One program per module, three labels

`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake` builds every test program. A
module that has tests calls it through `sushiblas_add_module_tests()` and gets the program
`sushiblas_<module>_tests`; `CMakeLists.txt` here builds `sushiblas_tests` from the files under
`unit/`, `integration/` and `regression/`. Every program lands in `build/tests/`.

A test belongs to a group by the prefix of its GoogleTest suite name: `Unit_`, `Integration_` or
`Regression_`. `gtest_discover_tests` runs once per prefix on each program and gives the tests it
finds the CTest label `unit`, `integration` or `regression`. A suite whose name has none of the
three prefixes is found by no discovery call, so CTest never runs it.

A new test file is listed by hand, in its module's `CMakeLists.txt` or in the one here; nothing
globs the folders. A test that includes a header of a module in a higher tier than its own
belongs here, and `tools/layering/check_layering.py` reports it anywhere else.

## Running

Tests run through `sb`, which sets the loader path the test program needs.

| Command | Runs |
| --- | --- |
| `sb test` | The three labels together; the same as `--suite functional` |
| `sb test --suite unit` | The label `unit`; `integration` and `regression` work the same way |
| `sb test --suite all` | Every test CTest knows |
| `sb test --suite package` | Installs the library, then configures, builds and runs `package/` against the install |
| `sb test --filter <pattern>` | The tests whose name matches the pattern |
| `sb run benchmark_<name>` | One benchmark program, after `sb build --benchmarks` |

[`cli/README.md`](../cli/README.md) describes each flag, and
[the contributing guide](../docs/CONTRIBUTING.md) states what a new operation's tests carry.

## What the tests depend on

`sushiblas_tests` links the `sushiblas` library and GoogleTest, takes the project's build
policy, and links the include roots of the internal kernel modules, so a unit test can include a kernel
from the private `modules/kernels/` folder and call it directly. Those tests therefore see more than a consumer does; `package/` is
the one place that sees only the installed surface.

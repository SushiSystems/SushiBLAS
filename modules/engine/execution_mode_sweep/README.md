# execution_mode_sweep

The execution modes a routine's tests run it in. The module is internal and test-only: its headers are not
installed, and it is added only when `SB_BUILD_TESTS` is on.

It depends on `engine`. Its headers are included as `<SushiBLAS/execution_mode_sweep/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `execution_mode_sweep.hpp` | `K_BOTH_MODES`, `ForceReference` then `ForceSYCL` |

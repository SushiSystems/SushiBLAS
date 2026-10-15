# serial_reductions

Serial reductions over a host vector, which a reduction test takes its expected values from. The module is internal and test-only: its headers are not installed, and it is added
only when `SB_BUILD_TESTS` is on.

It depends on the standard library only. Its headers are included as `<SushiBLAS/serial_reductions/serial_reductions.hpp>`.

| File | Declares |
| --- | --- |
| `serial_reductions.hpp` | Serial reductions over a host vector, in namespace `ReferenceKernels` |

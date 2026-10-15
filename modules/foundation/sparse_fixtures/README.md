# sparse_fixtures

The host matrices the sparse tests build from and the host products they expect. The module is internal and test-only: its headers are not installed, and it is added
only when `SB_BUILD_TESTS` is on.

It depends on `core`, `arithmetic` and GoogleTest. Its headers are included as `<SushiBLAS/sparse_fixtures/sparse_samples.hpp>`.

| File | Declares |
| --- | --- |
| `sparse_samples.hpp` | The sample matrix, `HostTriplets`, the host products and the typed suite lists |

## sparse_samples.hpp

`sample_matrix()` is a 4 x 5 matrix of seven entries with an empty row and no two values equal,
listed out of row order so that assembly has to sort. `HostTriplets` holds a matrix as zero-based
triplets; the expected product of a test is computed from them on the host in double, by a loop
that shares no code with either kernel. `K_BOTH_BASES`, and `K_BOTH_MODES` of `execution_mode_sweep`, are what
every sparse suite runs over, and the typed suites cover every value dtype with both index
widths.

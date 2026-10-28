# Benchmarks tuning

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-19 | P2 | unmeasured | Tune the blocking of the packed GEMM on the CPU | `benchmark_gemm_blocking` has run on the CPU device and the blocking follows the procedure of `DESIGN_GEMM_CPU`, section 10.4, answering the eight unknowns of… |  |  | 2026-10-19 |
| RW-20 | P2 | unmeasured | Settle the solve block widths | `benchmark_triangular` has set the two widths, 256 and 128 today, and whether a small TRMV needs a serial arm (`DESIGN_BLAS_L3`, sections 8 and 13) |  |  | 2026-10-19 |
| RW-21 | P2 | blocked | Decide from the runs whether `ScratchShelf` gets size classes above 4 KiB | the runs show whether a recorded `trmv`, a packed GEMM and a scattering sparse product need classes above 4 KiB, and the shelf follows | RW-19, RW-20 |  | 2026-10-19 |
| RW-22 | P2 | unmeasured | Run the benchmarks on a device other than the CPU | the benchmark figures exist for a GPU; those of 2026-10-07 are from one Ryzen 5 7600X on the tree before the break |  |  | 2026-10-19 |
| RW-110 | P2 | unmeasured | Measure the GEMM tile shapes on the RTX 3080 Ti and fill its table entries | `K_GEMM_TILE_TABLE` holds the sm_86 entries `benchmark_gemm_tiles` printed, and `benchmark_gemm` on `cuda:0` with the chosen shapes is no slower than with… |  | `docs/agent/2026_10_19_PARAMETRIC_GEMM_TILES/PLAN.md` | 2026-10-26 |
| RW-111 | P2 | unmeasured | Measure the binary size each GEMM tile shape adds | the size of the library and of the test binary with one, two and three shapes per dtype is recorded, and the catalog limit of three shapes is kept or moved by… |  | `docs/agent/2026_10_19_PARAMETRIC_GEMM_TILES/SPEC.md` | 2026-10-26 |
| RW-112 | P2 | open | Choose the SYRK and SYR2K tiles per device | the `syrk` and `syr2k` kernels take their tile from a catalog and a chooser, as the tiled GEMM does |  |  | 2026-10-26 |
| RW-113 | P2 | open | Choose the no-local GEMM register block per device | `GemmNoLocalTile` comes from a measured choice per device instead of one budget |  |  | 2026-10-26 |
| RW-114 | P2 | open | Read the HALF `joint_matrix` shapes from `matrix_combinations` | the HALF matrix kernel runs a shape the device lists, not only 16 x 16 x 16 |  |  | 2026-10-26 |
| RW-115 | P2 | open | Read the work-group limit of each compiled GEMM tile kernel | `admits_gemm_tile` refuses a shape whose compiled kernel accepts fewer work-items than it launches, through `get_kernel_info` on a named kernel |  |  | 2026-10-26 |
| RW-116 | P2 | open | Let the GEMM tile tuner write a user cache file | entries `benchmark_gemm_tiles` measured on a user's device are read at start without a rebuild, through a format `file-formats` defines |  |  | 2026-10-26 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |

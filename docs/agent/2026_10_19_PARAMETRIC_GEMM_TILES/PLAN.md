# Parametric GEMM tiles: plan

Implements `SPEC.md` in this folder. Hierarchy first: the device identity, then the shape type,
then the kernel on it, then the catalog, the chooser and the table, then the tuner. Every task
has one acceptance criterion and one build. Every build and test carries `--jobs 4`.

## Before task 1

The owner started the plan on 2026-10-26 (RW-107). Before task 1, `sb test --jobs 4` runs once on
`opencl:cpu` and `cuda:0` and its failures go in `REPORT.md` as the baseline list; every task
passes only when it adds no failure to that list (`SPEC.md` 8.6).

## Tasks

| # | Task | Files | Acceptance | Build |
| --- | --- | --- | --- | --- |
| 1 | Add the architecture, the vendor and `local_mem_per_compute_unit` to `Device::Profile`, read once in `Detail::probe` through `info::device::architecture` behind `SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE`, `Unknown` without it; add the tile pin read of `SB_GEMM_TILE` | `device_policy.hpp`, new `tile_pin.hpp`, `test_device_policy.cpp`, new `test_tile_pin.cpp`, `launch/README.md` | A Medium test reads `nvidia_gpu_sm_86` and 100 KB per SM on `cuda:0`, and a known value or `Unknown` on `opencl:cpu`; a Small test fills a fake profile; a Small test reads a set and an unset pin | 1 |
| 2 | Introduce `GemmTileShape` and run `gemm_local_tiled` on a shape parameter; today's constants become the `Balanced` shape | `gemm.hpp`, new `gemm_tile_shape.hpp`, `blas_seam_kernels/README.md` | Every existing GEMM test passes unchanged, and `benchmark_gemm` on `cuda:0` stays within the spread of the run of 2026-10-19 | 1 |
| 3 | Add the catalog: two more shapes per real dtype, chosen from the survey `Wide` (twice the `Balanced` block rows with a 32 depth) and `Compact` (half the `Balanced` block in each axis) | new `gemm_tile_catalog.hpp`, `test_gemm.cpp` | A Medium test holds every shape to the bits of `Balanced` over whole and edge tiles, both layouts, every transpose, on `cuda:0` and `opencl:cpu`; the build log records the size of the library and the test binary before and after | 1 |
| 4 | Write `choose_gemm_tile`: the hard limits, the pin, the model of `SPEC.md` 5.3, the size classes of 5.2, each shape's work-group limit taken as an argument | new `gemm_tile_choice.hpp`, new `test_gemm_tile_choice.cpp` | Small tests on fake profiles of an sm_86, a gfx90a, a pvc, an unknown GPU and a CPU give the expected shape per size class, a shape past a hard limit is never returned, and a bad pin logs one WARN | 1 |
| 5 | Route `gemm_accessed` through the chooser with the `get_kernel_info` limits cached per device and dtype, add the empty `GemmTileTable` | `gemm.hpp`, new `gemm_tile_table.hpp`, `test_gemm.cpp` | A Medium test pins each shape through `SB_GEMM_TILE` and gets its bits; an unknown name logs one WARN and runs the chosen shape; every GEMM, SYMM and packed test passes | 1 |
| 6 | Write `benchmark_gemm_tiles`: every catalog shape, every size class, median of five runs, one table line per winner | new `benchmark_gemm_tiles.cpp`, the benchmark list, `tests/README.md` | It runs through `sb run` on `cuda:0` in under ten minutes and prints entries the table header compiles | 1 |
| 7 | Measure the RTX 3080 Ti and fill its entries; record the 7600X's absence of a local-tiled path | `gemm_tile_table.hpp`, `REPORT.md` | `benchmark_gemm` on `cuda:0` with the chosen shapes is no slower than with `Balanced` pinned at any size, five runs each; the gap at order 2048 to the f3e6382 baseline is reported, closed or not | 1 |
| 8 | Close: the architecture page, the module READMEs, `GPU_VALIDATION.md` 3.4, `KNOWN_ISSUES.md`, the changelog, the backlog entries for what section 5.5 leaves out | docs only | The ten checkers of `AGENTS.md` pass | 0 |

## Order and waits

Each task waits on the one above it, except task 4, which needs only tasks 1 and 2 and can be
written beside task 3. The plan runs inline unless the owner chooses waves; tasks 3 and 4 are
the only pair with disjoint files.

## Measurements

Every `perf` claim carries five runs before and after, median and spread, as `performance`
asks. The before of task 7 is the run of 2026-10-19 and the f3e6382 baseline. Task 3 records
the binary sizes so the catalog limit of `SPEC.md` 7.3 is a measured number.

## What this plan does not do

The items of `SPEC.md` 5.5, each a backlog entry written in task 8.

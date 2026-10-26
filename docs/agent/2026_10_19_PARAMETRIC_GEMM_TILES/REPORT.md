# Parametric GEMM tiles: report

Written on 2026-10-26. The owner ruled that the plan is written without a build and that the
build and the tests run at the end, so nothing below has been compiled into a binary or run.
Each changed C++ file passed `clang++ -fsyntax-only` with the flags of its neighbour in
`build/compile_commands.json`, device pass included for the SYCL files.

## Tasks

| # | State | What was done |
| --- | --- | --- |
| Baseline | not run | The list of RW-106 failures is not recorded; it needs `sb test` |
| 1 | written | `Device::Vendor`, `Device::Architecture`, `local_mem_per_compute_unit_of`, three `Profile` fields, `gemm_tile_pin.hpp`; Small and Medium tests |
| 2 | written | `GemmTileShape`; `gemm_local_tiled` takes `Shape`, defaulting to `Balanced`; `GemmLocalTile` removed |
| 3 | written | `Wide` and `Compact`; a Medium test holds both to the bits of `Balanced`. The binary sizes are not measured (RW-111) |
| 4 | written | `choose_gemm_tile` and its Small tests on fake sm_86, gfx90a, PVC, unknown and CPU profiles |
| 5 | written | `gemm_accessed` routes through the chooser; one WARN per process for a refused pin; empty `K_GEMM_TILE_TABLE` |
| 6 | written | `benchmark_gemm_tiles` on the benchmark list; not run |
| 7 | not done | Needs the 3080 Ti run (RW-110) |
| 8 | done | READMEs, `KERNELS.md`, `GPU_VALIDATION`, changelog, backlog RW-110 to RW-116 |

## Changes from the plan

- The work-group limit of a compiled kernel is not queried (`SPEC.md` 8.7, RW-115).
- No test pins each shape through `SB_GEMM_TILE`, because the variable is read once per process
  (`SPEC.md` 8.10).
- `benchmark_gemm_tiles` takes the harness's median of ten timed runs per shape and order, not
  five.

## Risks the run must check

- Until RW-110 fills the table, the 3080 Ti runs the model's choice: `Wide` for every FLOAT32
  product of the large class. `Wide` stages 25.3 KB per work-group against `Balanced`'s 8.6 KB,
  and the audit saw a larger tile lose at order 2048 to occupancy. The first `benchmark_gemm`
  run can show a slowdown there; `SB_GEMM_TILE=Balanced` restores the shape of before.
- Complex and HALF products now also reach `Wide` and `Compact`. The bit test covers FLOAT32,
  FLOAT64 and complex FLOAT32, not HALF or complex FLOAT64.

## SOLID breaches and dependencies

- `device_policy.hpp` holds the profile, its probe, the cache blocking, the grid planning and
  now the architecture table: more than one reason to change. Splitting the table out changes
  the public header shape, the owner's decision. Seen, kept.
- `Detail::probe` fills three more fields, widening the one function that fills all of them.
  Made, kept.
- Adding a shape edits two switches over the roles, `Detail::gemm_local_tiled_as` and the
  tuner's `time_role`. The set is closed and small (`sushiskills`, default 1, "not when").
- `K_LOCAL_ACC_BYTES` in `sycl/level3/gemm.hpp` is now read only by `syr2k.hpp`, and the catalog
  writes the same 256 B as a literal. The two can drift apart. Made, kept.
- No dependency was added. `blas_seam_kernels` already depended on `launch` and `core`.

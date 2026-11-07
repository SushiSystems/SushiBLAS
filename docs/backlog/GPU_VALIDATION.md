# Gpu validation

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-23 | P2 | unmeasured | Confirm the device profile on the card | `Device::Profile` reports `has_dedicated_local_memory = true` on the RTX 3080 Ti |  | `docs/design/GPU_VALIDATION/README.md` | 2026-10-19 |
| RW-24 | P2 | unmeasured | Measure the tiled GEMM's bank conflicts and register spill | the bank conflicts and the spill of the 8x8 block are measured |  | `docs/design/GPU_VALIDATION/README.md` | 2026-10-19 |
| RW-25 | P2 | unmeasured | Run the 128-bit vector path and the grid-stride launch | both shapes of `launch_1d.hpp` have run on a GPU |  | `docs/design/GPU_VALIDATION/README.md` | 2026-10-19 |
| RW-26 | P2 | unmeasured | Measure where an axis reduction switches to one work-group per segment | the switch point is measured on a GPU |  | `docs/design/GPU_VALIDATION/README.md` | 2026-10-19 |
| RW-27 | P2 | unmeasured | Run SushiAI's determinism regressions on the GPU | the regressions have passed on the card |  | `docs/design/GPU_VALIDATION/README.md` | 2026-10-19 |
| RW-28 | P2 | unmeasured | Run the group-per-row GEMV and SYMV paths | both paths have run on a device that chooses them; on the CPU only a 4 x 16384 GEMV reaches them and SYMV never does |  | `docs/design/GPU_VALIDATION/README.md` | 2026-10-19 |
| RW-29 | P2 | unmeasured | Confirm the device memory alignment on a run | `Storage` is seen to receive 128-byte aligned memory on CUDA, and on Level Zero when a machine with it is at hand |  | `docs/design/GPU_VALIDATION/README.md` | 2026-10-19 |
| RW-30 | P2 | unmeasured | Measure the weight gradient of `window_dot` | the cost of its two 64-bit divisions per term is measured |  | `docs/design/GPU_VALIDATION/README.md` | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |

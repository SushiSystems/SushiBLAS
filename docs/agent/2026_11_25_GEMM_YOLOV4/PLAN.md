# GEMM speed for YOLOv4 training: plan

**Status:** Open — tasks 1 to 5 written on 2026-11-25, unbuilt and unmeasured

The owner ruled on 2026-11-25 that the GEMM speed a SushiAI YOLOv4 training step needs is the
work, the design left to the worker, SOLID and maintainability first. A second ruling the same day
added that the stack also runs on AMD GPUs (HIP on Windows), so no change may tune for NVIDIA alone.

## Measured

One step takes 486 ms, of which `device.run` takes 448 ms; Darknet takes 330 to 350 ms an
iteration. Nsight puts `gemm_local_tiled` at 46.8 % of device time and `gemm_depth_split` at
25.5 %. The profiler record of 20 steps holds 86 GEMM shapes and 264.9 ms of GEMM span time per
step. Every product is RowMajor FLOAT32 with beta 0. SushiAI lowers a convolution to three of
them, with P the pixels of a batch of two:

- forward `F`: `M = P`, `N = Cout`, `K = taps`, A by B transposed;
- input gradient `dX`: `M = P`, `N = taps`, `K = Cout`, A by B;
- weight gradient `dW`: `M = Cout`, `N = taps`, `K = P`, A transposed by B.

The top rows, measured on the RTX 3080 Ti at about 25 to 30 TFLOP/s FP32 peak:

| Role | M x N x K | Calls | Median us | TFLOP/s | ms per step | Variant today |
| --- | --- | --- | --- | --- | --- | --- |
| dW | 256 x 1152 x 11552 | 4 | 3371 | 2.02 | 13.48 | depth_split, 1152 groups |
| dW | 64 x 288 x 184832 | 2 | 6560 | 1.04 | 13.12 | depth_split, 80 groups |
| dW | 256 x 2304 x 2888 | 8 | 1623 | 2.10 | 12.98 | Balanced, 144 tiles |
| dW | 512 x 2304 x 2888 | 6 | 2162 | 3.15 | 12.97 | Wide, 144 tiles |
| dW | 128 x 1152 x 11552 | 8 | 1616 | 2.11 | 12.93 | depth_split, 576 groups |
| dW | 1024 x 4608 x 722 | 6 | 2020 | 3.37 | 12.12 | Wide, 576 tiles |
| F | 722 x 1024 x 4608 | 6 | 1965 | 3.47 | 11.79 | Wide, 96 tiles |
| dW | 64 x 64 x 184832 | 3 | 2730 | 0.55 | 8.19 | depth_split, 128 groups |
| dX | 4608 x 722 x 1024 | 6 | 1337 | 5.10 | 8.02 | Wide |
| F | 2888 x 512 x 2304 | 6 | 1322 | 5.15 | 7.93 | Wide |
| dX | 2888 x 2304 x 512 | 6 | 1304 | 5.23 | 7.82 | Wide |
| F | 2888 x 256 x 2304 | 8 | 945 | 3.6 | 7.56 | Wide, 92 tiles |
| dX | 11552 x 1152 x 128 | 8 | 866 | 3.93 | 6.93 | Wide |
| dW | 32 x 64 x 184832 | 1 | 6609 | 0.11 | 6.61 | depth_split, 128 groups |
| dW | 27 x 32 x 739328 | 1 | 3874 | 0.33 | 3.87 | depth_split, 56 groups |

The depth-split rows take about 79 ms a step and the local-tiled rows about 185 ms, at 2 to 6
TFLOP/s.

## Causes

- **Local-tiled inner loop.** Each K-step a work-item reads 8 scalars of A and 8 of B from local
  memory for 64 multiply-adds. Its rows are a strided lattice (`local_row + i * WG_ROWS`) and the
  A tile is stored row by row of C, so no two of its values are adjacent and every read is one
  32-bit load. The load issue rate, not the multiply-adds, sets the ceiling: about 4.5 TFLOP/s
  even at order 2048.
- **Under-filled tiled launches.** `256 x 2304 x 2888`, `512 x 2304 x 2888`, `722 x 1024 x 4608`
  and `2888 x 256 x 2304` launch 92 to 144 tiles on 80 compute units that each hold several groups.
- **Depth split.** It reads global memory straight into a 4 x 4 register block, two multiply-adds
  per load, and launches one 256-item group per block of C. A small C gives 56 to 128 groups,
  about one per compute unit, whatever K is.

## Decisions

1. **Fragments, not a lattice.** A work-item owns runs of `K_FRAGMENT` neighbouring rows and
   columns, a run every `WG * K_FRAGMENT`. `K_FRAGMENT` is the elements of one 128-bit load of T
   (`GemmVectorLanes<T>`: float 4, double 2, HALF 8, complex 1), capped at the register block. The
   A tile is stored K-major (one row per K-step, leading dimension `K_BLOCK_M + K_FRAGMENT`) and
   the B tile keeps its K-step rows (`K_BLOCK_N + max(K_FRAGMENT, 2)`), both as `sycl::vec<T,
   K_FRAGMENT>` so a run is one aligned load. Float `Balanced` goes from 16 loads to 4 per 64
   multiply-adds. Every element still sums K in order, so the bits of every shape stay those of
   `Balanced`. `StagedLanes<T>` now reads its value from `GemmVectorLanes<T>`, so the lane count is
   written once.
2. **Depth slices across work-groups.** When the `Balanced` tiles of C times the batch number fewer
   than `compute_units * min(4, groups one unit's local memory holds)`, K is cut into slices of
   whole K tiles, at least 256 K-steps each. The tiled kernel sums slice s into its own row-major C
   in scratch (`DepthSlices` on `gemm_local_tiled`); a fold kernel adds the slices in slice order,
   then applies alpha, beta and the epilogue through `GemmStore`, which moves from
   `gemm_packed.hpp` into `gemm_store.hpp` so the packed product and the fold share one store. The
   result is the same on every run, though not the bits of the unsliced sum. HALF is never sliced:
   its partials would round to HALF.
3. **One scratch lease per GEMM.** `prepare_gemm` leases `max(packed bytes, slice bytes)` through
   the new `gemm_scratch.hpp`; `PreparedGemm::pack` becomes `scratch`. The scratch forms of `gemm`
   and `gemm_accessed` run the packed product, else the slices, else the form without scratch. A
   GEMM that leases slice scratch records addresses, as a packed one already does: it takes no
   bindings and no multi-device split.
4. **Vendor-neutral selection.** Every number the selection reads comes from `Device::Profile`:
   compute units, local memory per unit and per group, work-group size. Nothing names an
   architecture. `K_MAX_SLICE_GROUPS_PER_UNIT = 4` and `K_MIN_SLICE_DEPTH = 256` are the only new
   constants, and both are first guesses, verified on no device yet.
5. **The variant is visible.** `KernelVariant::GemmDepthSliced` notes the instant
   `blas.gemm.depth_sliced`, so a profile shows which products took slices.

## On a 64-wide wavefront

| Change | AMD (64-wide wavefront, 64 KiB local memory per CU) | Verified on |
| --- | --- | --- |
| Fragments | Holds: the work-group shape is unchanged, LDS reads `ds_read_b128` natively, and the padding is in elements, not in banks of one vendor. It could regress if the LDS bank pattern of 128-bit reads across a 64-lane wavefront conflicts where the 32-lane warp did not. | Nothing yet |
| Depth slices | Holds: the group target follows `compute_units` and `local_mem_per_compute_unit`; on a 110-CU gfx90a it asks for 440 groups. A unit that holds fewer groups aims for fewer. | A Small test of the plan on a fake gfx90a profile only |
| Scratch lease | Holds: no device property enters it. | Nothing yet |
| `K_MAX_SLICE_GROUPS_PER_UNIT`, `K_MIN_SLICE_DEPTH` | Unknown: chosen for neither vendor, to be measured on both | Nothing yet |

Every number this plan expects to move is a prediction for the RTX 3080 Ti; none is measured on
AMD.

## Tasks

| # | Task | Files | Acceptance | State |
| --- | --- | --- | --- | --- |
| 1 | Fragment geometry in the tile shape and the index helpers | `gemm_tile_shape.hpp`, `gemm_index.hpp`, `test_gemm_tile_shape.cpp`, `test_gemm_tile_catalog.cpp`, `test_gemm_index.cpp` | The Small tests of the shapes, the fragment coordinate, the slice group and the slice range pass | written |
| 2 | Fragment ownership and K-major A tile in `gemm_local_tiled`, `DepthSlices` parameter | `sycl/level3/gemm.hpp` | `Medium_GemmLocalTiledAgreement` and `Medium_GemmTileShapeBits` pass unchanged on `cuda:0` and `opencl:cpu` | written |
| 3 | Depth-slice plan, sliced kernel and fold, scratch routing, `GemmStore` header, variant | new `gemm_depth_slice_plan.hpp`, `gemm_depth_sliced.hpp`, `gemm_store.hpp`; `gemm_packed.hpp`, `gemm_accessed_packed.hpp`, `kernel_variant.hpp`, `engine_profile.cpp`, new `test_gemm_depth_slice_plan.cpp`, `test_gemm_depth_sliced.cpp`, `test_gemm_store.cpp`, operations `test_gemm.cpp` | `Small_GemmDepthSlicePlan`, `Small_GemmStore`, `Medium_GemmDepthSliced` (host CPU, exact) and `Medium_GemmDepthSlicedAgreement` (device) pass | written |
| 4 | One scratch lease per GEMM | new `gemm_scratch.hpp`, new `test_gemm_scratch.cpp`; `gemm_recording.hpp`, `dispatch.hpp`, `test_gemm.cpp` | `Small_GemmScratch` passes, and every row of `Medium_GemmAgreement`, the two new deep weight-gradient rows among them, passes on `cuda:0` | written |
| 5 | YOLOv4 rows in the GEMM benchmark | `benchmark_gemm.cpp` | `sb run --type release benchmark_gemm` prints 15 `gemm yolov4` rows under RowMajor | written |
| 6 | Measure: five runs before and after on the RTX 3080 Ti, then the step | `REPORT.md` | Each `gemm yolov4` row and the step time are reported with spread; a row that does not move reverts its change | open |
| 7 | Measure the same rows on an AMD GPU | `REPORT.md` | The rows are reported, or the plan says no AMD device was reachable | open |

## Bench

`sb run --type release benchmark_gemm` prints the rows `gemm yolov4 <role> <M>x<N>x<K>` in the
RowMajor section. The GFLOP/s column is the number to watch. The depth-split rows (`dW`, K of
11552 and more) should rise severalfold from 0.1 to 2 TFLOP/s; the tiled rows should rise from
3 to 5 TFLOP/s. `sb test -j 7` runs the new Small and Medium tests.

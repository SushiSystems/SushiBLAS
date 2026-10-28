# Kernels reductions

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-31 | P2 | open | Give SYRK an epilogue and a variant without local memory | SYRK runs on a device that cannot host its tile; `syr2k` and `her2k` stay on the local-memory shape on a CPU |  |  | 2026-10-19 |
| RW-32 | P2 | open | Share GEMM's local-memory padding and store interleave with SYRK | both are helpers outside `gemm.hpp` that SYRK uses |  |  | 2026-10-19 |
| RW-33 | P2 | open | Add `trmm` | `trmm` exists, built from `TrmvTerm` and `TriangleRow` |  |  | 2026-10-19 |
| RW-34 | P2 | open | Add a `rot` that takes c and s as tensors | `rotg` chains into `rot` inside one graph |  |  | 2026-10-19 |
| RW-35 | P2 | open | Coalesce the reads of TRSM's diagonal-block solve | the diagonal-block solve of `trsm.hpp` reads coalesced; lines 126 to 163 map one work-item per vector today |  |  | 2026-10-19 |
| RW-36 | P2 | open | Close the gap between the SYCL `tpsv` and the dense `trsv` | a 128-row block of `tpsv` costs what `trsv` does; 0.30 ms against 0.19 ms up to order 1024 and 0.42 ms against 0.20 ms from 2048 (FLOAT32, RTX 3080 Ti) |  |  | 2026-10-19 |
| RW-37 | P2 | unmeasured | Measure where a narrow-band `tbsv` is cheaper on the host | the crossing point against a GPU is measured |  |  | 2026-10-19 |
| RW-38 | P2 | unmeasured | Measure the two bounds of `plan_depth_split` on a second GPU | both bounds, set from one RTX 3080 Ti, are measured on another card |  |  | 2026-10-19 |
| RW-39 | P2 | open | Give the `gemv` with few strided rows a benchmark row | the benchmark measures it |  |  | 2026-10-19 |
| RW-40 | P2 | open | Merge the mirrored symmetric reads into one brick | `symv`, `SymmetricAccess` and `symmetric_product` share one text, and the dense `symv`, `trmv` and one-vector solve run on the structured algorithm bricks |  |  | 2026-10-19 |
| RW-41 | P2 | open | Move the reference rank updates onto `ReferenceAccumulator<T>` | they accumulate through it and the parity tests have a complex truth |  |  | 2026-10-19 |
| RW-42 | P2 | open | Add an implicit `BLASScalar` from `std::complex<float>`, and `rotmg` for `HALF` | both exist with tests |  |  | 2026-10-19 |
| RW-43 | P2 | open | Move the complex operators onto `safe_mul` and `safe_div` | the elementwise and nonlinear kernels use them for complex values |  |  | 2026-10-19 |
| RW-44 | P2 | blocked | Bring the batch feed into the compiled plan | a batch step costs no extra dispatch round | sushiruntime RW-59 |  | 2026-10-19 |
| RW-45 | P2 | open | Batch level 1 and level 2 operations over tensors above rank 1 | each level 1 and level 2 operation takes a batched tensor |  |  | 2026-10-19 |
| RW-46 | P2 | open | Record a plan once | `Engine::execute` keeps the compiled graph instead of discarding it on every call |  |  | 2026-10-19 |
| RW-47 | P2 | open | Fuse kernels as a pass over the recorded graph | the pass exists and is measured |  |  | 2026-10-19 |
| RW-48 | P2 | open | Write a parallel `shuffle`, then freeze the permutation a seed gives | `shuffle` runs in parallel and a test pins the permutation of a seed |  |  | 2026-10-19 |
| RW-49 | P2 | open | Design what SushiAI's gather-based loss asks for | broadcasting binaries, a concatenation along a non-channel axis and a row gather with a gather-form backward are designed, with a general group count for… |  |  | 2026-10-19 |
| RW-50 | P2 | open | Round host `HALF` in the subnormal range | the host conversion rounds below 2^-13 as a device does |  |  | 2026-10-19 |
| RW-93 | P2 | open | Fill a random tensor from a stream position read when the task runs | `PositionedRandomOps` fills from a position held in a tensor operand, and a plan replayed twice with the position rewritten between runs draws two different… |  |  | 2026-10-19 |
| RW-94 | P2 | open | Add an index family: rows by index and one element per row | `IndexOps` gathers rows by an INT32 or INT64 index with a defined result out of range and a gather-form backward, with tests |  |  | 2026-10-19 |
| RW-95 | P2 | open | Add a broadcast seam, one value per row or per column under any functor | `Dispatch::execute_broadcast` and `BroadcastOps` exist with tests, and no existing plan records different tasks |  |  | 2026-10-19 |
| RW-96 | P2 | open | Add bilinear upsampling and its gradient | `upsample_bilinear` and its backward exist in both kernels, with tests |  |  | 2026-10-19 |
| RW-97 | P2 | open | Add the index of the maximum along one axis | `argmax_along` folds through `Reduce::RunningArgMax`, the lowest index winning a tie, with tests |  |  | 2026-10-19 |
| RW-98 | P2 | open | Add GELU by the error function | `gelu_erf` and `gelu_erf_backward` exist beside `gelu`, whose formula stays, with tests |  |  | 2026-10-19 |
| RW-99 | P2 | open | Add a mask of the k largest of each row | `ReductionOps::top_k_mask_along` exists with tests, the lowest index winning a tie |  |  | 2026-10-19 |
| RW-100 | P2 | blocked | Add the index of the rows a mask selects | `IndexOps::select_rows` exists with tests | RW-94 |  | 2026-10-19 |
| RW-103 | P2 | open | Add a bfloat16 `DataType` | `Core::DataType` holds bfloat16 and the operations that take HALF take it, with tests |  |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |

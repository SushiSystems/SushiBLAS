# Changelog

The entries dated before 2026-09-07 were reshaped to one line each on 2026-10-05, and their
original text is frozen in the changelog document in the version history.

## Unreleased

- 2026-11-21 — cli: Changed `sb run --sort` to list the built examples, each with its description; `sb run --sort --all` lists every program as before (RW-164).

## v0.2.0 — 2026-11-19

- 2026-11-19 — deps: Pinned SushiRuntime v0.6.0 in place of v0.5.2, which names a buffer by its `GlobalHandle` and releases a borrow when its buffer dies (`sushistack.deps.toml`).
- 2026-11-17 — deps: Upgraded SushiLog to v0.4.1 (`sushistack.deps.toml`, `23e874aa`).
- 2026-11-17 — cluster: Shared a tensor pack by its `global_handle()`, gave a remote task `TaskBuffer` views with their handles and kept a late count's borrowed words with the owner token (`d93d2d1c`).
- 2026-11-16 — tools: Took the shared tools from SushiCore v0.14.1, which leaves out its two own checkers and writes this repository's licence header, through `hub bump` (`tools/shared_tools.lock`).
- 2026-11-16 — broadcast: Bound the matrix, the output and the vectors of `execute_broadcast` by handle, so a planned tensor is accepted unless the functor declares reads (GLOBAL_HANDLE R10).
- 2026-11-16 — functor_map: Moved `MapOperand` and the two calls that bind an operand into `map_operand.hpp`, so a second seam binds as the elementwise executors do (GLOBAL_HANDLE R10).
- 2026-11-16 — engine: Replaced `OperandPointers` with the runtime's `BoundOperands`, read through `operand_address()`, which refuses a wrong slot, offset or element type (GLOBAL_HANDLE R10).
- 2026-11-16 — blas: Changed the GEMM split slice and `bound_gemm_operands()` to take `BoundOperands`, and the copy to read operands through `operand_address()` (GLOBAL_HANDLE R10).
- 2026-11-16 — spatial: Added `SpatialOps::upsample_bilinear` and `upsample_bilinear_backward`, which interpolate by an integer scale with the taps shared by the reference and SYCL kernels (`0a48d98e`).
- 2026-11-16 — reductions: Added `ReductionOps::argmax_along`, which returns the index of the largest element of each row or column (`0a48d98e`).
- 2026-11-16 — random: Added `PositionedRandomOps::bernoulli`, which reads its seed and offset from a rank-1 INT64 position tensor when the plan runs (`0a48d98e`).
- 2026-11-16 — gather: Added `RowIndexOps`, which copies, sums and picks table rows by an INT32 or INT64 index (`49951724`).
- 2026-11-16 — index_kernels: Added the internal module that holds the reference and SYCL kernels of `RowIndexOps` (`49951724`).
- 2026-11-16 — broadcast: Added `BroadcastOps` and `Dispatch::execute_broadcast`, which combine a matrix with up to four vectors and bound scalars (`49951724`).
- 2026-11-15 — tensor: Changed `StorageBlock::device_index()` and `Storage::device_index()` to return an optional, empty for a planned block in place of device 0 (GLOBAL_HANDLE R07).
- 2026-11-15 — tensor: Added `Storage::require_device_index()`, which throws `UnsupportedExecution` for a planned allocation (GLOBAL_HANDLE R07).
- 2026-11-15 — graph_primitives: Changed `HostBuffer::device_index()` to return `std::optional<DeviceIndex>`, as a runtime buffer does (GLOBAL_HANDLE R07).
- 2026-11-15 — graph_primitives: Changed every primitive to refuse an operand that states no device with InvalidGraph (GLOBAL_HANDLE R07).
- 2026-11-15 — engine: Changed an operation that needs a device to refuse a planned operand, naming the operation (GLOBAL_HANDLE R07).
- 2026-11-15 — tensor: Changed `Tensor::region()` to key on `Storage::global_handle()` in place of the allocation base, and a borrowed storage to keep its buffer's handle (`aaa927cc`, `borrow_runtime_storage`).
- 2026-11-15 — tensor: Added `TensorFactory::planned` and `from_file`, which make tensors over a `PlannedBlock`, a planned runtime buffer with a handle and no host address (`f4b56fdd`).
- 2026-11-15 — tensor: Gave host blocks a handle from one process-wide `LocalHandleSpace`, so a host tensor's region names its allocation (`12bd3f6a`).
- 2026-11-15 — engine: Recorded GEMM, copy and elementwise operands by handle, so those operations accept a planned tensor (`92a2638d`).
- 2026-11-15 — blas: Added `make_gemm_split`, which cuts a GEMM along the longer side of C and declares the split to the runtime (`4caa85ec`).
- 2026-11-15 — blas: Added `gemm_part_regions`, which states the bytes each GEMM split part reads and writes per batch and per panel (`e7f03b46`).
- 2026-11-15 — blas: Changed the GEMM split slice to take its dependency events and return its own event in place of waiting (`21fb5f66`).
- 2026-11-15 — graph_primitives: Added `RuntimePrimitiveRecorder::borrowed_region`, which names a scratch block as a region, and built the primitives' scratch regions from `GlobalHandle` (`aaa927cc`).
- 2026-11-15 — engine: Stated for each costed node the `NodeCost` operations, read and written bytes, kind and dtype, from `declared_read_bytes()` and `declared_written_bytes()` (`4caa85ec`).

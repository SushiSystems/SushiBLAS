# Known issues

Defects and unverified claims that are recorded and not yet settled. The estate audit of
2026-10-05 found the older ones; its text, with evidence, is
`docs/agent/2026_10_05_ESTATE_AUDIT/REPORT.md`. The v0.1.0 work of 2026-10-07 added the rest; its
record is `docs/agent/2026_10_07_V010/`, and a name such as `DESIGN_BLAS_L3` or `PLAN_EXT` below
is a heading of that folder's `SPEC.md` or `PLAN.md`. Work that is planned and is not a defect is
in [the backlog](../design/REMAINING_WORK.md).

## Verification state

| Issue | Where |
| --- | --- |
| The tree was configured, built and linked whole by `sb build`, test programs included, and `sb test` ran on 2026-10-12: 2 207 tests, none failed. That is one Windows machine, one card (an NVIDIA RTX 3080 Ti) and one toolchain (the pinned intel-llvm). The tree has not run on Linux, on an AMD or Intel GPU, on the OpenCL CPU device or on a second machine | the whole tree |
| Of the benchmark programs only `benchmark_band_packed` has run, on that card; no other design figure about speed is a measurement | `tests/<tier>/<module>/benchmark_*.cpp` |
| The record-time refusal of a device that cannot host the `syrk` or `syr2k` tile was never seen to throw: the one device the tree ran on hosts both tiles | `modules/operations/blas/include/SushiBLAS/blas/local_tile.hpp`, `require_local_tile` |
| With `SB_BUILD_TESTS` and `SB_BUILD_EXAMPLES` off, as under a consumer, the tree was never configured, and `tests/package/` was not built against an installed tree in the run of 2026-10-12 | `CMakeLists.txt`, `tests/package/` |
| The GPU evidence is functional runs on one card, an RTX 3080 Ti, the last on 2026-10-12. None recorded the device profile or which GPU-shaped paths it took, and nothing in sections 2 to 5 of the checklist is measured | [GPU_VALIDATION.md](../design/GPU_VALIDATION.md), section 0 |
| The kernels have been compiled by one toolchain, for the one card the tests ran on. No other ahead-of-time target has compiled them, and no SPIR-V device has. `Kernels::multiply`, `divide` and `conjugate`, which keep `std::complex` functions out of the SYCL BLAS kernels, are among them | `modules/foundation/arithmetic/include/SushiBLAS/arithmetic/complex_arithmetic.hpp`, `modules/kernels/` |
| Whether every backend returns memory on the 128-byte boundary `Storage` asks `Runtime::buffer` for is unverified; an address off it fails an `SB_ASSERT`, which a build with `NDEBUG` compiles out | `modules/data/tensor/include/SushiBLAS/tensor/storage.hpp` |
| The speed of the packed GEMM path is unmeasured, and its blocking defaults are derived from the cache size a device reports, which was never read on a real run | `modules/foundation/launch/include/SushiBLAS/launch/device_policy.hpp`, `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp` |
| The solve block widths, 256 on a CPU and 128 on a GPU, come from assumed launch and step costs | `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level3/triangular_plan.hpp` |
| The rotation bricks were transcribed from the reference BLAS (`drotg.f90`, `zrotg.f90`, `drotmg.f`) through a summarising fetch, without the source at hand. Whether `drotmg.f` writes the implied entries of H only for flags 0 and 1, as the brick does, is not settled | `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/givens.hpp`, `modified_givens.hpp` |
| `Core::IndexBase` carries 221 and 222, written from memory of the Sparse BLAS header | `modules/foundation/core/include/SushiBLAS/core/sparse_types.hpp` |
| The parity allowances of the random reference tests are estimates, and `LargeLambdaPoissonGivesTheSameCounts` asserts equality for one seed, which a device whose `log` or `lgamma` differs from the host's can break | `tests/operations/random/test_random/test_random_reference.cpp` |
| `ThePackedAndTheUnpackedKernelAgreeToTheBit` fails if the OpenCL CPU runtime contracts or reorders one GEMM kernel and not the other; whether it does was never observed | `tests/operations/blas/test_dispatch/test_packed_product.cpp` |
| The open and unverified items each design lists | `docs/agent/2026_10_07_V010/SPEC.md`: `DESIGN_RECORDER` section 15, `DESIGN_SUPPORT_MEMORY` 18, `DESIGN_BLAS` 12, `DESIGN_BLAS_L3` 13, `DESIGN_BLAS_BAND_PACKED` 14, `DESIGN_GEMM_CPU` 10.3 and 13, `DESIGN_SPARSE` 18, `DESIGN_YOLO26` 13; and `REPORT.md`, section 7 |

## Behaviour

| Issue | Where |
| --- | --- |
| The reference path does host-side work on tensor storage. On Windows with CUDA, SushiRuntime faults (`0xC0000006`) when the host touches a `Residency::Shared` buffer while a kernel runs on the same GPU, so `ForceReference`, a print and a save are exposed whenever another kernel is in flight. Untested here | `modules/engine/engine/include/SushiBLAS/engine/kernel_pair_dispatcher.hpp`, `modules/operations/io/source/io.cpp` |
| The reference path of the elementwise, nonlinear and logic seams runs the same functor as the SYCL kernel in a serial loop. It catches a wrong launch or a wrong index, not a wrong formula | `modules/foundation/reference/include/SushiBLAS/reference/serial_1d.hpp`, `modules/engine/functor_map/include/SushiBLAS/functor_map/map_task.hpp` |
| The toolchain's host `sycl::half` truncates a result below 2^-13 (about 1.2e-4) where a device rounds it: its conversion treats the lowest normal exponent as subnormal. `HALF` parity between the two paths is bitwise only above that magnitude, and one unit in the last place apart below it | `modules/foundation/reference/include/SushiBLAS/reference/serial_1d.hpp` |
| `poisson`, the normal family (`normal`, `he_normal`, `xavier_normal`, `log_normal`), `truncated_normal` and `orthogonal` are not bit-identical between the two paths: each path calls its own math library, and `orthogonal` also sums in another order | `modules/kernels/random_kernels/include/SushiBLAS/random_kernels/sycl/`, `modules/kernels/random_kernels/include/SushiBLAS/random_kernels/reference/orthonormalize.hpp` |
| The OpenCL CPU runtime of oneAPI 2026.1 (device runtime 2026.21.7.0.24_160000, on an AMD EPYC 7763) crashes in its vectorizer pass `vplan-vec` while JIT-compiling `TrmvRow<std::complex<float>>`, and every band and packed triangular test on that device then segfaults. Taking the diagonal out of the row loop did not avoid it. Every CI job sets `CL_CONFIG_CPU_VECTORIZER_MODE=1`, so kernels run unvectorized there; no other device or toolchain has shown the crash | `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level2/trmv.hpp`, `.github/workflows/ci.yml` |
| The permutation `shuffle` draws for a seed is not frozen; a parallel shuffle would change it | `modules/operations/random/source/shuffle.cpp` |
| `IO` takes an `Engine&`, so it is the one class outside the recorder shape, and every call that reads or writes a tensor runs the whole recorded graph first | `modules/operations/io/include/SushiBLAS/io/io.hpp`, `modules/operations/io/source/io.cpp` |
| `spmv` names two routines: the packed symmetric product on `BLASOps` and the CSR product on `SparseOps`, with kernels of that name in the same two namespaces | `modules/operations/blas/include/SushiBLAS/blas/level2.hpp`, `modules/operations/sparse/include/SushiBLAS/sparse/sparse.hpp` |
| "Packed" names two things: BLAS packed storage and the packed panels of the CPU GEMM | `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/packed_access.hpp`, `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/sycl/level3/packed_product.hpp` |
| `blas/sparse.hpp` is included by nothing and is installed; it declares a `SparseBLAS` class that holds an `Engine&` | `modules/operations/blas/include/SushiBLAS/blas/sparse.hpp` |
| A lease above 4 KiB is outside the scratch pool's size classes, so a recorded `trmv`, a packed GEMM (megabytes of panels) and a transposed or one-triangle sparse product each allocate and free device memory at every record | `modules/engine/engine/include/SushiBLAS/engine/scratch_shelf.hpp` |
| A tensor that is recorded and then dropped stays allocated until its graph node dies, and a caller that keeps a compiled plan keeps every operand and scratch block of that plan | `modules/engine/engine/include/SushiBLAS/engine/operand_set.hpp` |
| A recorder's identity comes from a `static` counter in an inline function, so two binaries that each link the static library can hand out one identity | `modules/engine/engine/include/SushiBLAS/engine/task_recorder.hpp` |
| `rule_task_dependencies` reads text: a kernel that accepts `deps` and never uses it passes. No test shows a kernel that drops its dependency events; `ARecordedTaskComputesItsResult` runs one operation of each seam alone | `tools/layering/check_engine_seams.py`, `tests/library/umbrella/test_SushiBLAS/test_task_dependencies.cpp` |
| A backward `tbsv` or `tpsv` and a backward `trsv` do not agree to the bit on inexact data, because the blocked dense solve subtracts from the top down | `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level3/trsm.hpp`, `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level2/triangular_solve.hpp` |
| `sptrsv` solves on one work-item; whether a long `single_task` trips a driver watchdog on a display GPU is unknown | `modules/kernels/sparse_kernels/include/SushiBLAS/sparse_kernels/sycl/sptrsv.hpp` |
| A SYCL sum that overflows its accumulation type midway gives an infinity where the host fold in double gives the finite value | `modules/foundation/reduction/include/SushiBLAS/reduction/reduce_1d.hpp`, `serial_reduce.hpp` |
| The eight older spatial operations do not check that a written tensor overlaps no read one; only the `window_dot` triple does | `modules/operations/spatial/source/` |
| A rank-0 tensor prints as `[]`, and `IO::to_vector` zero-fills its result before reading into it | `modules/operations/io/source/io.cpp` |
| The installed-package test calls no `SparseOps` routine, so it does not show that the sparse headers install | `tests/package/main.cpp` |
| On the pinned intel-llvm for Windows, a generic lambda handed to `with_element_type` that calls a task body holding a SYCL kernel left the `HostKernel` members of that kernel undefined, and the test executable failed to link; the dtype arms of `execute_random` and `execute_typed_task` are named classes for that reason. `all.cpp`, `any.cpp`, `shuffle.cpp` and `orthogonal.cpp` keep a generic lambda as their arm and linked in the build of 2026-10-12 | `modules/operations/random/include/SushiBLAS/random/dispatch.hpp`, `modules/engine/engine/include/SushiBLAS/engine/task.hpp` |
| `gelu.cpp` includes `<numbers>`, a C++20 header, in a library built at C++17 | `modules/operations/nonlinear/source/gelu.cpp` |
| `sb test --package` ends in a traceback when a DLL beside the consumer is locked | `cli/sushiblas/services/project.py` |
| `sb run --sort` keeps a flag name the sibling CLIs renamed to `--pick` | `cli/sushiblas/cli.py` |

## Comments and documents

| Issue | Where |
| --- | --- |

## Layout

| Issue | Where |
| --- | --- |
| The benchmark programs were not built on the module tree | `tests/<tier>/<module>/benchmark_*.cpp` |
| No test includes each header of a header-only module first and alone, so a header that forgot an include is caught only where a source of its own module includes it | `modules/foundation/`, `modules/kernels/`, `modules/engine/operands/`, `modules/engine/functor_map/` |
| `sushi-module.toml` spells the remote in lower case, the remote itself is `SushiSystems/SushiBLAS` | `sushi-module.toml` |

## Continuous integration

| Issue | Where |
| --- | --- |
| The build jobs check out SushiRuntime beside this repository and pins no tag of it, so it builds only while the runtime's default branch carries the interface this tree is written against (`Runtime::buffer` with an alignment, `Recorder::task`, `Graph::plan()`) | `.github/workflows/ci.yml`, `cmake/Runtime.cmake` |
| The Windows, sanitizer, ThreadSanitizer and SushiAI jobs, the `sb` calls on a runner and `nightly.yml` have never run | `.github/workflows/ci.yml`, `.github/workflows/nightly.yml` |
| The Linux build of the small and medium tests took 48 minutes without a compiler cache, above the pre-merge budget of 45; its timeout is 90 minutes until the ccache numbers are known | `.github/workflows/ci.yml` |
| Size is declared per test program, so a program whose tests mostly need no device is `medium` as a whole, and `small` covers seven programs | `cmake/SushiBLASTests.cmake` |
| `sb test` builds its own ctest command, because sushicore's `CMakeDriver` has no way to pass `-LE` to ctest | `cli/sushiblas/services/project.py` |
| The release workflow has never run | `.github/workflows/release.yml` |

## Release state

| Issue | Where |
| --- | --- |
| No release is tagged, and the Unreleased section holds every entry of the changelog | `docs/reference/CHANGELOG.md` |
| The build files pin no SushiRuntime version: `find_package(SushiRuntime QUIET)` and `find_dependency(SushiRuntime)` name none, and the sibling step takes the checkout as it is | `cmake/Runtime.cmake`, `cmake/SushiBLASConfig.cmake.in` |

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
| The whole suite ran once on this tree, on 2026-10-09, on the CUDA build with an RTX 3080 Ti: 2 047 of 2 047 tests passed, the ones CTest lists as skipped among them. The tree has not run on the OpenCL CPU device or on any other card | the whole tree |
| Of the benchmark programs only `benchmark_band_packed` has run, on that card; no other design figure about speed is a measurement | `tests/benchmark/` |
| The record-time refusal of a device that cannot host the `syrk` or `syr2k` tile was never seen to throw: the one device the tree ran on hosts both tiles | `include/SushiBLAS/engine/blas/utils.hpp`, `require_local_tile` |
| With `SB_BUILD_TESTS` and `SB_BUILD_EXAMPLES` off, as under a consumer, the tree was never configured | `CMakeLists.txt` |
| The last syntax sweep of the whole tree ran at commit `1746b03`, over 250 translation units. Every later commit was checked file by file only, so an error between two files that were each parsed alone is still possible | `docs/agent/2026_10_07_V010/SPEC.md`, `OWNER_DECISIONS_LATE` |
| The GPU evidence is two functional runs on one card, an RTX 3080 Ti. Neither recorded the device profile or which GPU-shaped paths it took, and nothing in sections 2 to 5 of the checklist is measured | [GPU_VALIDATION.md](../design/GPU_VALIDATION.md), section 0 |
| No kernel written since the break has been compiled for an ahead-of-time target. `Kernels::multiply`, `divide` and `conjugate`, which keep `std::complex` functions out of the SYCL BLAS kernels, are among them | `include/SushiBLAS/support/complex_arithmetic.hpp`, `source/kernels/sycl/` |
| Whether `sycl::aligned_alloc_device` returns 128-byte aligned memory on every backend is unverified; `Storage` throws when an address is not aligned | `include/SushiBLAS/storage.hpp` |
| The speed of the packed GEMM path is unmeasured, and its blocking defaults are derived from the cache size a device reports, which was never read on a real run | `include/SushiBLAS/support/device_policy.hpp`, `include/SushiBLAS/kernels/blas/level3/packed_product_plan.hpp` |
| The solve block widths, 256 on a CPU and 128 on a GPU, come from assumed launch and step costs | `source/kernels/sycl/blas/level3/triangular_plan.hpp` |
| The rotation bricks were transcribed from the reference BLAS (`drotg.f90`, `zrotg.f90`, `drotmg.f`) through a summarising fetch, without the source at hand. Whether `drotmg.f` writes the implied entries of H only for flags 0 and 1, as the brick does, is not settled | `include/SushiBLAS/kernels/blas/givens.hpp`, `modified_givens.hpp` |
| `Core::IndexBase` carries 221 and 222, written from memory of the Sparse BLAS header | `include/SushiBLAS/core/sparse_types.hpp` |
| The parity allowances of the random reference tests are estimates, and `LargeLambdaPoissonGivesTheSameCounts` asserts equality for one seed, which a device whose `log` or `lgamma` differs from the host's can break | `tests/unit/test_random_reference.cpp` |
| `ThePackedAndTheUnpackedKernelAgreeToTheBit` fails if the OpenCL CPU runtime contracts or reorders one GEMM kernel and not the other; whether it does was never observed | `tests/unit/test_packed_product.cpp` |
| The link of the new task bodies on the pinned intel-llvm is untried. `all.cpp`, `any.cpp`, `shuffle.cpp` and `orthogonal.cpp` keep a generic lambda as their dtype arm, the shape the row on `HostKernel` symbols below describes | `source/engine/logic/`, `source/engine/math/random/` |
| The open and unverified items each design lists | `docs/agent/2026_10_07_V010/SPEC.md`: `DESIGN_RECORDER` section 15, `DESIGN_SUPPORT_MEMORY` 18, `DESIGN_BLAS` 12, `DESIGN_BLAS_L3` 13, `DESIGN_BLAS_BAND_PACKED` 14, `DESIGN_GEMM_CPU` 10.3 and 13, `DESIGN_SPARSE` 18, `DESIGN_YOLO26` 13; and `REPORT.md`, section 7 |

## Behaviour

| Issue | Where |
| --- | --- |
| The reference path does host-side work on tensor storage. On Windows with CUDA, SushiRuntime faults (`0xC0000006`) when the host touches a `Residency::Shared` buffer while a kernel runs on the same GPU, so `ForceReference`, a print and a save are exposed whenever another kernel is in flight. Untested here | `include/SushiBLAS/engine/kernel_pair_dispatcher.hpp`, `source/io.cpp` |
| The reference path of the elementwise, nonlinear and logic seams runs the same functor as the SYCL kernel in a serial loop. It catches a wrong launch or a wrong index, not a wrong formula | `include/SushiBLAS/support/serial_1d.hpp`, `include/SushiBLAS/engine/math/elementwise/map_task.hpp` |
| The toolchain's host `sycl::half` truncates a result below 2^-13 (about 1.2e-4) where a device rounds it: its conversion treats the lowest normal exponent as subnormal. `HALF` parity between the two paths is bitwise only above that magnitude, and one unit in the last place apart below it | `include/SushiBLAS/support/serial_1d.hpp` |
| `poisson`, the normal family (`normal`, `he_normal`, `xavier_normal`, `log_normal`), `truncated_normal` and `orthogonal` are not bit-identical between the two paths: each path calls its own math library, and `orthogonal` also sums in another order | `source/kernels/sycl/random/`, `source/kernels/reference/random/orthonormalize.hpp` |
| The permutation `shuffle` draws for a seed is not frozen; a parallel shuffle would change it | `source/engine/math/random/shuffle.cpp` |
| `IO` still takes an `Engine&` and calls `engine_.execute()` at nine sites, so it is the one class outside the recorder shape, and every call that reads or writes a tensor runs the whole recorded graph first | `include/SushiBLAS/io.hpp`, `source/io.cpp` |
| `spmv` names two routines: the packed symmetric product on `BLASOps` and the CSR product on `SparseOps`, with kernels of that name in the same two namespaces | `include/SushiBLAS/engine/blas/level2.hpp`, `include/SushiBLAS/engine/sparse.hpp` |
| "Packed" names two things: BLAS packed storage and the packed panels of the CPU GEMM | `include/SushiBLAS/kernels/blas/packed_access.hpp`, `include/SushiBLAS/kernels/sycl/blas/level3/packed_product.hpp` |
| `engine/blas/sparse.hpp` is no longer included by anything and is still installed; it declares a `SparseBLAS` class that holds an `Engine&` | `include/SushiBLAS/engine/blas/sparse.hpp` |
| A lease above 4 KiB is outside the scratch pool's size classes, so a recorded `trmv`, a packed GEMM (megabytes of panels) and a transposed or one-triangle sparse product each allocate and free device memory at every record | `include/SushiBLAS/engine/scratch_shelf.hpp` |
| A tensor that is recorded and then dropped stays allocated until its graph node dies, and a caller that keeps a compiled plan keeps every operand and scratch block of that plan | `include/SushiBLAS/engine/operand_set.hpp` |
| A recorder's identity comes from a `static` counter in an inline function, so two binaries that each link the static library can hand out one identity | `include/SushiBLAS/engine/task_recorder.hpp` |
| `rule_task_dependencies` reads text: a kernel that accepts `deps` and never uses it passes, and the regression test behind it covers seven operations | `tools/layering/check_layering.py`, `tests/regression/test_task_dependencies.cpp` |
| A backward `tbsv` or `tpsv` and a backward `trsv` do not agree to the bit on inexact data, because the blocked dense solve subtracts from the top down | `source/kernels/sycl/blas/level3/trsm.hpp`, `source/kernels/sycl/blas/level2/triangular_solve.hpp` |
| `sptrsv` solves on one work-item; whether a long `single_task` trips a driver watchdog on a display GPU is unknown | `source/kernels/sycl/sparse/sptrsv.hpp` |
| A SYCL sum that overflows its accumulation type midway gives an infinity where the host fold in double gives the finite value | `include/SushiBLAS/support/reduce_1d.hpp`, `serial_reduce.hpp` |
| The eight older spatial operations do not check that a written tensor overlaps no read one; only the `window_dot` triple does | `source/engine/spatial/` |
| A rank-0 tensor prints as `[]`, and `IO::to_vector` zero-fills its result before reading into it | `source/io.cpp` |
| The installed-package test calls no `SparseOps` routine, so it does not show that the sparse headers install | `tests/package/main.cpp` |
| On the pinned intel-llvm for Windows, a generic lambda handed to `with_element_type` that calls a task body holding a SYCL kernel leaves the `HostKernel` members of that kernel undefined, and the test executable fails to link; the dtype arms of `execute_random` and `execute_typed_task` are named classes for that reason | `include/SushiBLAS/engine/math/random/dispatch.hpp`, `include/SushiBLAS/engine/task.hpp` |
| `gelu.cpp` includes `<numbers>`, a C++20 header, in a library built at C++17 | `source/engine/math/nonlinear/gelu.cpp` |
| `sb test --suite package` ends in a traceback when a DLL beside the consumer is locked | `cli/sushiblas/services/project.py` |
| The fixture `Regression_AxisSpanTest` sits in the unit folder, so its tests carry the label `regression` | `tests/unit/test_mean_spatial.cpp` |
| `sb run --sort` keeps a flag name the sibling CLIs renamed to `--pick` | `cli/sushiblas/cli.py` |

## Comments and documents

| Issue | Where |
| --- | --- |
| The manual and the module READMEs were read against the code once, on 2026-10-09, by nine agents, and about 150 sentences were corrected. Each report lists the sentences it could not check: measurements, history, and claims about another repository or a device | `docs/agent/2026_10_09_MANUAL_READ/REPORT.md` |

## Layout

| Issue | Where |
| --- | --- |
| `tests/` holds `benchmark/` and `package/`, two folders the layout does not name | `tests/` |
| `sushi-module.toml` spells the remote in lower case, the remote itself is `SushiSystems/SushiBLAS` | `sushi-module.toml` |

## Continuous integration

| Issue | Where |
| --- | --- |
| The `functional` job checks out SushiRuntime `main`. `Storage` now calls the aligned shared and device allocation of `RuntimeContext`, which was uncommitted in the runtime's working tree when it was written, so the job cannot build until the runtime carries it | `.github/workflows/ci.yml`, `include/SushiBLAS/storage.hpp` |
| The `functional` job runs on Linux only and calls cmake and ctest directly | `.github/workflows/ci.yml` |
| The release workflow has never run | `.github/workflows/release.yml` |
| The pyflakes step ends in `\|\| true` and cannot fail | `.github/workflows/ci.yml` |

## Release state

| Issue | Where |
| --- | --- |
| No release was ever tagged, and the Unreleased section holds 197 entries | `docs/reference/CHANGELOG.md` |
| The library builds only against a SushiRuntime with the aligned shared and device allocation; no runtime tag carries it, and the build files pin no runtime version | `cmake/Runtime.cmake`, `cmake/SushiBLASConfig.cmake.in` |

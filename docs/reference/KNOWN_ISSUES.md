# Known issues

Next id: KI-30

## Behaviour

### KI-1 A print, a save or a reference run can fault on Windows with CUDA
- Origin: sushiruntime
- Upstream: sushiruntime KI-1
- Where: `modules/engine/engine/include/SushiBLAS/engine/kernel_pair_dispatcher.hpp`,
  `modules/operations/io/source/io.cpp`
- Recognise: the process stops with `0xC0000006` while another kernel is in flight; `ForceReference`,
  a print and a save touch `Residency::Shared` storage from the host. Untested here
- Workaround: finish the queue before a print, a save or a `ForceReference` run

### KI-2 The reference path of the elementwise, nonlinear and logic seams checks launches, not formulas
- Origin: own
- Where: `modules/foundation/reference/include/SushiBLAS/reference/serial_1d.hpp`,
  `modules/engine/functor_map/include/SushiBLAS/functor_map/map_task.hpp`
- Cause: it runs the same functor as the SYCL kernel in a serial loop
- Recognise: a wrong formula passes parity; a wrong launch or index fails it
- Workaround: test a formula against an independent value
- Fix: none, the shared functor is what keeps the two paths one text

### KI-3 Host `HALF` truncates below 2^-13 where a device rounds
- Origin: toolchain
- Where: `modules/foundation/reference/include/SushiBLAS/reference/serial_1d.hpp`
- Cause: the toolchain's host `sycl::half` treats the lowest normal exponent as subnormal
- Recognise: `HALF` parity is one unit in the last place apart below about 1.2e-4, bitwise above
- Workaround: compare `HALF` below that magnitude to one unit in the last place

`RW-50` rounds the host conversion.

### KI-4 Some random families differ between the two paths
- Origin: own
- Where: `modules/kernels/random_kernels/include/SushiBLAS/random_kernels/sycl/`,
  `modules/kernels/random_kernels/include/SushiBLAS/random_kernels/reference/orthonormalize.hpp`
- Cause: `poisson`, the normal family, `truncated_normal` and `orthogonal` call each path's own
  math library, and `orthogonal` also sums in another order
- Recognise: those draws are not bit-identical between the SYCL and the reference path
- Workaround: compare them within the allowances of `test_random_reference.cpp`
- Fix: none, each path calls its own math library

### KI-5 The OpenCL CPU runtime crashes compiling `TrmvRow<std::complex<float>>`
- Origin: toolchain
- Where: `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level2/trmv.hpp`,
  `.github/workflows/ci.yml`
- Cause: the vectorizer pass `vplan-vec` of oneAPI 2026.1's OpenCL CPU runtime (device runtime
  2026.21.7.0.24_160000, AMD EPYC 7763) crashes in the JIT
- Recognise: every band and packed triangular test on that device then segfaults
- Workaround: none known; taking the diagonal out of the row loop did not avoid it

### KI-6 The permutation `shuffle` draws for a seed is not frozen
- Origin: own
- Where: `modules/operations/random/source/shuffle.cpp`
- Cause: no test pins it, and a parallel shuffle would change it
- Recognise: a seed's permutation may change between versions
- Workaround: none
- Fix: RW-48

### KI-7 Every `IO` call runs the whole recorded graph first
- Origin: own
- Where: `modules/operations/io/include/SushiBLAS/io/io.hpp`, `modules/operations/io/source/io.cpp`
- Cause: `IO` takes an `Engine&`, the one class outside the recorder shape
- Recognise: a print or a save waits for every recorded task
- Workaround: none
- Fix: RW-4

### KI-8 `spmv` and "packed" each name two things
- Origin: own
- Where: `modules/operations/blas/include/SushiBLAS/blas/level2.hpp`,
  `modules/operations/sparse/include/SushiBLAS/sparse/sparse.hpp`,
  `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/packed_access.hpp`
- Cause: `spmv` is the packed symmetric product on `BLASOps` and the CSR product on `SparseOps`;
  "packed" is BLAS packed storage and the CPU GEMM's packed panels
- Recognise: one name, two routines or two meanings
- Workaround: qualify the name by its class or module
- Fix: RW-1

### KI-9 `blas/sparse.hpp` is installed and used by nothing but the umbrella and one test
- Origin: own
- Where: `modules/operations/blas/include/SushiBLAS/blas/sparse.hpp`
- Cause: it declares a `SparseBLAS` class holding an `Engine&` that `SparseOps` replaced
- Recognise: the installed tree carries the header
- Workaround: use `SparseOps`
- Fix: RW-2

### KI-10 A lease above 4 KiB allocates device memory at every record
- Origin: own
- Where: `modules/engine/engine/include/SushiBLAS/engine/scratch_shelf.hpp`
- Cause: the scratch pool has no size class above 4 KiB
- Recognise: a recorded `trmv`, a packed GEMM and a transposed or one-triangle sparse product
  allocate and free at each record
- Workaround: none
- Fix: RW-21

### KI-11 A recorded tensor stays allocated until its graph node dies
- Origin: own
- Where: `modules/engine/engine/include/SushiBLAS/engine/operand_set.hpp`
- Cause: a recorded node owns its operands and scratch blocks
- Recognise: dropping a tensor frees nothing while its node lives; a kept plan keeps every operand
- Workaround: drop the compiled plan to release its operands
- Fix: none, a recorded node owns what it reads and writes

### KI-12 Two binaries that link the static library can share a recorder identity
- Origin: own
- Where: `modules/engine/engine/include/SushiBLAS/engine/task_recorder.hpp`
- Cause: the identity comes from a `static` counter in an inline function
- Recognise: two recorders in one process report one identity
- Workaround: link the shared library
- Fix: RW-67

### KI-13 The dependency rule reads text
- Origin: own
- Where: `tools/layering/check_engine_seams.py`,
  `tests/library/umbrella/test_SushiBLAS/test_task_dependencies.cpp`
- Cause: `rule_task_dependencies` passes a kernel that accepts `deps` and never uses it, and
  `ARecordedTaskComputesItsResult` runs one operation of each seam alone
- Recognise: no test shows a kernel that drops its dependency events
- Workaround: none
- Fix: RW-78

### KI-14 Backward banded and packed solves differ from the dense solve in the last bit
- Origin: own
- Where: `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level3/trsm.hpp`,
  `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level2/triangular_solve.hpp`
- Cause: the blocked dense solve subtracts from the top down
- Recognise: a backward `tbsv` or `tpsv` and a backward `trsv` disagree on inexact data
- Workaround: compare them within a tolerance
- Fix: none, the two solves sum in different orders

### KI-15 `sptrsv` solves on one work-item
- Origin: own
- Where: `modules/kernels/sparse_kernels/include/SushiBLAS/sparse_kernels/sycl/sptrsv.hpp`
- Cause: the solve is one `single_task`
- Recognise: a long solve on a display GPU may trip a driver watchdog; whether it does is unknown
- Workaround: run long solves on a device without a display
- Fix: RW-55

### KI-16 A SYCL sum that overflows midway gives an infinity
- Origin: own
- Where: `modules/foundation/reduction/include/SushiBLAS/reduction/reduce_1d.hpp`
- Cause: the device accumulates in the element type, the host fold in double
- Recognise: the host gives a finite sum where the device gives infinity
- Workaround: sum in a wider type
- Fix: none, the device accumulates in the element type

### KI-17 Eight spatial operations accept an output that overlaps an input
- Origin: own
- Where: `modules/operations/spatial/source/`
- Cause: only the `window_dot` triple checks the overlap
- Recognise: a wrong result when an output aliases an input
- Workaround: pass distinct tensors
- Fix: RW-73

### KI-18 A rank-0 tensor prints as `[]`
- Origin: own
- Where: `modules/operations/io/source/io.cpp`
- Cause: the printer walks no axis, and `IO::to_vector` zero-fills its result before reading
- Recognise: `[]` where a value was expected
- Workaround: read the value with `to_vector`
- Fix: RW-64

### KI-19 The package test does not show that the sparse headers install
- Origin: own
- Where: `tests/package/main.cpp`
- Cause: it calls no `SparseOps` routine
- Recognise: a broken sparse install passes the package test
- Workaround: none
- Fix: RW-61

### KI-20 A generic lambda calling a SYCL task body fails to link on Windows
- Origin: toolchain
- Where: `modules/operations/random/include/SushiBLAS/random/dispatch.hpp`,
  `modules/engine/engine/include/SushiBLAS/engine/task.hpp`
- Cause: the pinned intel-llvm for Windows leaves the `HostKernel` members of the kernel undefined
- Recognise: an undefined `HostKernel` symbol at link time
- Workaround: write the dtype arm as a named class, as `execute_random` and `execute_typed_task` do

### KI-21 `gelu.cpp` includes a C++20 header in a C++17 library
- Origin: own
- Where: `modules/operations/nonlinear/source/gelu.cpp`
- Cause: it includes `<numbers>`
- Recognise: a strict C++17 compiler refuses the file
- Workaround: none
- Fix: RW-3

### KI-22 `sb test --package` ends in a traceback when a DLL is locked
- Origin: own
- Where: `cli/sushiblas/services/project.py`
- Cause: the copy beside the consumer does not handle a locked file
- Recognise: a Python traceback in place of a console error
- Workaround: close the program holding the DLL
- Fix: RW-84

### KI-23 `sb run --sort` keeps a name the sibling command lines renamed
- Origin: own
- Where: `cli/sushiblas/cli.py`
- Cause: the rename to `--pick` did not reach this command line
- Recognise: `--pick` fails here and works in the siblings
- Workaround: spell `--sort`
- Fix: RW-80

## Layout

### KI-24 No test includes each header of a header-only module first and alone
- Origin: own
- Where: `modules/foundation/`, `modules/kernels/`, `modules/engine/operands/`,
  `modules/engine/functor_map/`
- Cause: no such test exists
- Recognise: a header that forgot an include builds until a source of another module includes it
  first
- Workaround: none
- Fix: RW-76

### KI-25 `sushi-module.toml` spells the remote in lower case
- Origin: own
- Where: `sushi-module.toml`
- Cause: the remote was written by hand
- Recognise: the file differs from `SushiSystems/SushiBLAS`
- Workaround: none needed on a case-blind host
- Fix: RW-77

## Continuous integration

### KI-26 The build pins no SushiRuntime version
- Origin: own
- Where: `cmake/Runtime.cmake`, `cmake/SushiBLASConfig.cmake.in`, `.github/workflows/ci.yml`
- Cause: `find_package(SushiRuntime QUIET)` and `find_dependency(SushiRuntime)` name none, and CI
  checks out the runtime's default branch
- Recognise: the build breaks when the runtime's default branch drops `Runtime::buffer` with an
  alignment, `Recorder::task` or `Graph::plan()`
- Workaround: check out a runtime commit that carries them
- Fix: RW-89

### KI-27 The Windows job and two nightly jobs fail
- Origin: own
- Where: `.github/workflows/ci.yml`, `.github/workflows/nightly.yml`
- Cause: no SYCL device passes the Windows job's hardware filter, and the nightly Benchmarks and
  Quarantined jobs fail at configure in `cmake/SushiBLASSushiLog.cmake`
- Recognise: those three jobs are red; the Linux, sanitizer, ThreadSanitizer and SushiAI jobs pass
- Workaround: none
- Fix: RW-85

### KI-28 The Linux job of the small and medium tests runs over its budget
- Origin: own
- Where: `.github/workflows/ci.yml`
- Cause: the build took 48 minutes without a compiler cache, against a pre-merge budget of 45
- Recognise: its timeout is 90 minutes until the ccache figures are known
- Workaround: none
- Fix: RW-86

### KI-29 The benchmark programs are not built on the module tree
- Origin: own
- Where: `cmake/SushiBLASSushiLog.cmake`
- Cause: the nightly Benchmarks job fails at configure
- Recognise: no benchmark binary exists after a nightly run
- Workaround: build them locally with `sb build --benchmarks`
- Fix: RW-85

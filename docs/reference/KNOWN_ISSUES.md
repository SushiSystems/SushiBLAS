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

### KI-20 A generic lambda calling a SYCL task body fails to link on Windows
- Origin: toolchain
- Where: `modules/operations/random/include/SushiBLAS/random/dispatch.hpp`,
  `modules/engine/engine/include/SushiBLAS/engine/task.hpp`
- Cause: the pinned intel-llvm for Windows leaves the `HostKernel` members of the kernel undefined
- Recognise: an undefined `HostKernel` symbol at link time
- Workaround: write the dtype arm as a named class, as `execute_random` and `execute_typed_task` do

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

### KI-29 The benchmark programs are not built on the module tree
- Origin: own
- Where: `cmake/SushiBLASSushiLog.cmake`
- Cause: the nightly Benchmarks job fails at configure
- Recognise: no benchmark binary exists after a nightly run
- Workaround: build them locally with `sb build --benchmarks`
- Fix: RW-85

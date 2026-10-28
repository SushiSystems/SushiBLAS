# SushiBLAS architecture

These pages describe how SushiBLAS is built, layer by layer, for a contributor who has to change
it. Every class and file they name exists under `modules/`. To
use the library, the [root README](../../README.md) and [the first program](../getting_started/FIRST_PROGRAM.md) are enough.

The pages describe the tree as it is. They give no figure for speed, and
[the known issues](../backlog/README.md) say what has run and on which device. A header
named alone, such as `engine/task.hpp`, is its include path after `SushiBLAS/`. A name in
`SushiBLAS::Internal` or a `Detail` namespace is not part of the interface a consumer may rely
on, and a header of a kernel module other than `blas_seam_kernels` is not installed.

A reference to "the architecture, section 3" elsewhere in the manual means the page numbered 3
here.

| Section | Page |
| --- | --- |
| 1 | [What SushiBLAS is](OVERVIEW.md) |
| 2 | [The layered architecture](LAYERS.md) |
| 3 | [Execution modes](EXECUTION_MODES.md) |
| 4 | [`Tensor`, `Storage` and memory](TENSOR_AND_MEMORY.md) |
| 5 | [`Engine`, the recorder and the views](ENGINE_AND_RECORDER.md) |
| 6 | [The dispatch seams and the fusion hooks](DISPATCH_SEAMS.md) |
| 7 | [Kernels: reference and SYCL](KERNELS.md) |
| 8 | [The SushiRuntime dependency](RUNTIME_DEPENDENCY.md) |
| 9 | [Quarantined surface (`modules/operations/experimental/include/SushiBLAS/experimental/`)](QUARANTINE.md) |
| 10 | [Packaging](PACKAGING.md) |
| 11 | [Exceptions and statuses](EXCEPTIONS.md) |
| 12 | [The fixed-order fold (`reduce_1d.hpp`)](FIXED_ORDER_FOLD.md) |
| 13 | [Profiling and the cost of a task](PROFILING.md) |

## Project layout

```
modules/                One tree per module: README.md, CMakeLists.txt, include/, source/, tests/
  foundation/
    core/               Element types, traits, the logger, errors, the version macros
    arithmetic/         The accumulator trait, complex arithmetic, safe math, Math
    launch/             The device policy and the per-element launch shapes
    reduction/          The fixed-order whole-tensor and segmented reductions
    reference/          The reference accumulator and the serial host loops and folds
    substitution/       RunningSolution, the unrounded solution of a substitution
    orthonormal/        The layout and constants of the orthonormalisation sweep
    sparse_assembly/    The host assembly of CSR arrays from triplets
    geometry/           Vector3T, QuaternionT, Matrix4T, transforms, the window geometry
    graph_primitives/   Reduce, segmented reduce, prefix scan and counting sort on a task graph
  data/
    tensor/             Storage, Tensor, TensorFactory
    sparse_storage/     The CSR matrix, the sparse vector and the builder of both
  kernels/
    blas_seam_kernels/  The GEMM kernels, the packed product and the access policies a
                         public seam instantiates; installed
    blas_kernels/, random_kernels/, sparse_kernels/, spatial_kernels/
                        The reference and SYCL kernels of each family; never installed
  engine/
    engine/             Engine, TaskRecorder, the execution mode, scratch, the random stream
    operands/           The operand checks and views the families share
    functor_map/        The per-element seam: the map task and its bindings
  operations/
    blas/, elementwise/, nonlinear/, reductions/, random/, logic/, sparse/, spatial/
                        One operation view each, with one <op>.cpp per operation
    io/                 Host transfer, save, load and print (native, .npy, raw binary)
    experimental/       The batched FFT, LU and GMRES; the other LAPACK declarations are undefined
  library/
    umbrella/           SushiBLAS.hpp, every public header in one line
    sushiblas/          The one library a consumer links
applications/examples/  Eight example programs, the adapter example among them
tests/                  Every test, mirroring modules/; see tests/README.md
  <tier>/<module>/      test_<header>.cpp per public header, and benchmark_*.cpp
                        built with sb build --benchmarks
  package/              Out-of-tree find_package(SushiBLAS) consumer
cli/                    The `sb` / `sushiblas` CLI package; see cli/README.md
cmake/                  CMake modules
tools/                  Documentation, layering and licence checkers; see tools/README.md
docs/                   The manual; see docs/README.md
.config/                Configuration a flag can point a tool at; see .config/README.md
sushi-module.toml       What `hub` reads about this module
LICENSE                 PolyForm Noncommercial License 1.0.0
COMMERCIAL.md           How to ask for a commercial licence
NOTICE.md               Third-party notices, including the portBLAS-derived tiling
third_party/licenses/   Licence texts of the third-party parts NOTICE.md lists
```

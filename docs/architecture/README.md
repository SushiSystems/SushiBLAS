# SushiBLAS architecture

These pages describe how SushiBLAS is built, layer by layer, for a contributor who has to change
it. Every class and file they name exists under `modules/`. To
use the library, the [root README](../../README.md) is enough; its
[project layout](../../README.md#project-layout) section holds the directory tree.

The pages describe the tree as it is. They give no figure for speed, and
[the known issues](../reference/KNOWN_ISSUES.md) say what has run and on which device. A header
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

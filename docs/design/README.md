# Design map

Each topic has one document. Look a topic up here before writing; extend the document that
exists.

| Topic | Document | Status |
| --- | --- | --- |
| What has not run on GPU hardware, and how to check it | [GPU_VALIDATION.md](GPU_VALIDATION.md) | Open |
| SYCL types in the interface, exceptions against statuses, and which kernels are installed | [PUBLIC_SEAM.md](PUBLIC_SEAM.md) | Shipped |
| The split of the library into modules: the measured include graph, the tree and what the move did | [MODULE_SPLIT.md](MODULE_SPLIT.md) | Shipped |
| SushiBLAS's share of the stack's layered execution: an engine without its own graph, the per-index seam, `convolve_1d` and `blend_transforms`, the cluster module and `ordered_sum()` (B1 to B13) | [LAYERED_EXECUTION.md](LAYERED_EXECUTION.md) | Open |
| The backlog | [REMAINING_WORK.md](REMAINING_WORK.md) | |

## Decisions without a document here

The manual cites three names whose record is outside this folder.
[The glossary](../reference/GLOSSARY.md) defines them.

| Name | Where it is recorded |
| --- | --- |
| D1, the quarantine of the FFT and LAPACK declarations | SushiAI's `docs/design/ROADMAP.md`; described in [the architecture, section 9](../architecture/QUARANTINE.md) |
| D3, the layered SushiRuntime resolution | [The architecture, section 8](../architecture/RUNTIME_DEPENDENCY.md) |
| SB-0 to SB-12, the build-out slices | SushiAI's `docs/design/ROADMAP.md` and this repository's commit subjects |

The reasons behind the changes made before 2026-09-07 are in the frozen
`docs/archive/CHANGELOG.md`, which held them as paragraphs.

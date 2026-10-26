# Design map

Each topic has one document. Look a topic up here before writing; extend the document that
exists.

## Topics

<!-- topics:begin -->
| Topic | State | About |
| --- | --- | --- |
| [GPU_VALIDATION](GPU_VALIDATION/README.md) | Open | Several changes in this repository and in SushiAI were made for a GPU, justified by argument and by arithmetic, and checked only as far as a CPU can check them. |
| [LAYERED_EXECUTION](LAYERED_EXECUTION/README.md) | Open | SushiBLAS's share of the stack's layered execution: the library becomes the one door to computation for SushiAI and SushiEngine. |
| [MODULE_SPLIT](MODULE_SPLIT/README.md) | Shipped | SushiBLAS was laid out as one module: include/SushiBLAS/ and source/ under the root, one CMake target, one test program. |
| [PROFILING](PROFILING/README.md) | Open | What SushiBLAS tells SushiProfiler about its work, and where it logs a decision it took. |
| [PUBLIC_SEAM](PUBLIC_SEAM/README.md) | Shipped | Three properties of the public interface depart from the house C++ style: SYCL types in the interface, exceptions where the style asks for statuses, and kernel headers that are installed. |
<!-- topics:end -->

## Decisions without a document here

The manual cites three names whose record is outside this folder.
[The glossary](../reference/GLOSSARY.md) defines them.


The reasons behind the changes made before 2026-09-07 are in the frozen
the changelog document in the version history, which held them as paragraphs.

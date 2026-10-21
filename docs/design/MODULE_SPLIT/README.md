# Splitting the library into modules

**Status:** Shipped — 2026-10-09
**About:** SushiBLAS was laid out as one module: include/SushiBLAS/ and source/ under the root, one CMake target, one test program.
**Modules:** none

## Problem

SushiBLAS was laid out as one module: `include/SushiBLAS/` and `source/` under the root, one CMake
target, one test program. The tree already had boundaries, and `tools/layering/check_layering.py`
enforced them by pattern matching on include lines. The repository layout names a module tree for
a project with more than one module; the build should enforce the boundaries and a file's path
should name its owner.

The include graph, measured before the move (the table is in
the evidence), showed four facts:

- **One cycle.** `sparse_builder.hpp` included `io.hpp`, and `io.hpp` included `tensor.hpp`.
- **No kernel includes the engine or a tensor.** Every kernel edge goes to `core`, `support` or
  another kernel family.
- **The engine includes no operation family.** `engine.hpp` reaches `tensor`, `core` and its
  own bricks only.
- **Three families lean on a sibling.** `sparse` takes six operand headers from `blas`
  (`argument_check`, `vector_operand`, `matrix_operand`, `scalar`, `blas_metadata`, `utils`), and
  `nonlinear` and `logic` take `map_task.hpp` from `elementwise`.

## Decision

The owner accepted breaking the public include paths on 2026-10-09, and answered the four open
decisions the same day:

1. The kernel modules that no seam instantiates are internal. A module has one include root, and
   a root is installed whole or not at all.
2. The shared operand headers leave `blas`, which removes the `sparse` to `blas` edge.
3. Each module has its own test program, registered under the same three CTest labels, so
   `sb test` and its suites behave as before.
4. The module names stand.

## Design

### The module tree

```
modules/
  foundation/   core  support
  data/         tensor  sparse_storage
  kernels/      blas_seam_kernels  blas_kernels  random_kernels  sparse_kernels  spatial_kernels
  engine/       engine  operands  functor_map
  operations/   blas  elementwise  nonlinear  reductions  random  logic  sparse  spatial  io
                experimental
  library/      sushiblas
```

A tier depends only on the tiers above it in this list. Inside a tier, an edge between two
modules is declared in `cmake/` and nothing else is allowed. After the move no operation family
includes another, and seven edges stay inside a tier; `cmake/SushiBLASLayers.cmake` lists them.

| Module | Owns | Target |
| --- | --- | --- |
| `core` | Types, traits, logging, formatting, errors, the version header | interface |
| `support` | Launch shapes, reductions, accumulators, device policy, geometry, `safe_math.hpp` | interface |
| `tensor` | `Storage`, `Tensor`, `TensorFactory`, `HostDataType` | interface |
| `sparse_storage` | The CSR matrix, the sparse vector and the builder of both | object |
| `blas_seam_kernels` | The BLAS kernels a public seam instantiates; installed | interface |
| `blas_kernels` | The other reference and SYCL kernels of the BLAS routines; internal | interface |
| `random_kernels` | Philox and the distribution kernels | interface |
| `sparse_kernels` | The sparse kernels and the structure checks | interface |
| `spatial_kernels` | The window, pooling and broadcast kernels | interface |
| `engine` | `Engine`, `TaskRecorder`, the execution mode, scratch and the random stream | object |
| `operands` | Operand checks and operand views every family shares, `scalar_result.hpp`, `scratch_lease.hpp` | interface |
| `functor_map` | The map task and its three bindings | interface |
| `blas` and the seven other families | One operation view and its operations | object |
| `io` | Host transfer, the three file formats, `IOStatus` | object |
| `experimental` | The quarantined LAPACK and signal declarations | interface |
| `sushiblas` | The umbrella header, the one linked library, the install and export rules | the library |

Each module has the shape the layout gives it: `README.md`, `CMakeLists.txt`,
`include/SushiBLAS/<module>/`, `source/` where it has one, and `tests/`. A module is an
interface target, and an object library when it has sources: a program that links an object
library gets its objects, so a test that linked the library and a module would get them twice.

`operands` names no kernel module. `utils.hpp` once included a BLAS kernel plan; it is now five
headers: `scalar_result.hpp` and `scratch_lease.hpp` in `operands`, and
`packed_product_scratch.hpp`, `local_tile.hpp` and `blas_task.hpp` in `blas`.

The include of `io.hpp` in `sparse_builder.hpp` was not dead: the builder uses `HostDataType`,
which `io.hpp` declared. The trait lives in `tensor/host_data_type.hpp`, both headers include it,
and the cycle is gone.

### What breaks for a consumer

The include path of a header becomes `<SushiBLAS/<module>/<name>.hpp>`.

| Before | After |
| --- | --- |
| `<SushiBLAS/tensor.hpp>`, `storage.hpp`, `tensor_factory.hpp` | `<SushiBLAS/tensor/...>` |
| `<SushiBLAS/sparse_matrix.hpp>`, `sparse_vector.hpp`, `sparse_builder.hpp` | `<SushiBLAS/sparse_storage/...>` |
| `<SushiBLAS/engine.hpp>` | `<SushiBLAS/engine/engine.hpp>` |
| `<SushiBLAS/engine/blas.hpp>`, `<SushiBLAS/engine/blas/...>` | `<SushiBLAS/blas/...>` |
| `<SushiBLAS/engine/math/elementwise/...>` and its three siblings | `<SushiBLAS/elementwise/...>`, and so on |
| `<SushiBLAS/io.hpp>` | `<SushiBLAS/io/io.hpp>` |
| `<SushiBLAS/kernels/...>` | `<SushiBLAS/blas_seam_kernels/...>` |
| `<SushiBLAS/SushiBLAS.h>` | unchanged |
| `find_package(SushiBLAS)` and `SushiBLAS::SushiBLAS` | unchanged |

No name, namespace or signature changes. A consumer that includes only the umbrella header
changes nothing. SushiAI names individual headers and has to follow;
`tools/naming/sushiblas_include_paths_0_1_0.tsv` lists the paths that changed.

### Assumptions

- Object libraries composed into one shared library, with the interface modules linked for the
  build only, export as the single target `SushiBLAS::SushiBLAS`. `tests/package/` is the proof.
- The export macro and the logging definitions reach every object library through one policy
  function, as they reach the one target before the split.

### Tests

Integration and regression tests, `benchmark/`, `package/` and `common/` stay under the root
`tests/`. A unit test goes to the module whose header it includes. Eight unit tests that include a
higher tier, and seven that test more than one family, stay under `tests/unit/`:
`check_layering.py` reports an upward include wherever it is.

## Phases

| Id | Phase | State |
| --- | --- | --- |
| P0 | `cmake/Module.cmake`, the tier table, `foundation/core`, the library composed from it and the old tree | built |
| P1 | `support`, `tensor`, `sparse_storage`; the `io` include of the builder cut | built |
| P2 | The kernel modules, with the internal mark | built |
| P3 | `engine` and `operands` | built |
| P4 | The nine operation modules and `experimental`, one module per worker | built |
| P5 | `library/sushiblas`: the umbrella, install and export; the old `include/` and `source/` removed | built |
| P6 | Tier order in `check_layering.py`, every module README, the architecture pages, the path table, the changelog | built |

### Work

<!-- work:begin -->
No open work.
<!-- work:end -->

## Open questions

None. The benchmark programs have not been built on the module tree.

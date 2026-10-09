# Splitting the library into modules

**Status:** Shipped on 2026-10-09

SushiBLAS is laid out as one module: `include/SushiBLAS/` and `source/` under the root, one CMake
target, one test program. The tree already has boundaries, and `tools/layering/check_layering.py`
enforces them by pattern matching on include lines. This page proposes the module tree the
repository layout names for a project with more than one module, so that the build enforces the
boundaries and a file's path names its owner. The owner accepted breaking the public include
paths for it on 2026-10-09.

Section 1 is measured on the tree before the move, and sections 2 to 6 are the proposal as the
owner read it. Section 7 states what the move did differently.

## 1. What the include graph says

Measured on commit `83aaacf` over the 343 C++ files under `include/` and `source/`, with each
file assigned to the candidate module of section 2. An edge is one `#include` line that crosses
two candidates.

| From | To | Lines |
| --- | --- | --- |
| `support` | `core` | 3 |
| `tensor` | `core`, `support` | 14, 1 |
| `tensor` | `io` | 1 |
| `io` | `tensor`, `core`, `engine` | 1, 1, 2 |
| kernels | `core`, `support` | 136, 184 |
| sparse kernels | BLAS kernels | 7 |
| `engine` | `core`, `support`, `tensor` | 10, 2, 7 |
| `blas` | `engine`, BLAS kernels, `tensor`, `core`, `support` | 72, 56, 12, 17, 10 |
| `sparse` | `engine`, `blas`, sparse kernels, BLAS kernels | 44, 33, 10, 2 |
| `spatial` | `engine`, spatial kernels | 24, 10 |
| `random` | `engine`, random kernels | 19, 14 |
| `elementwise` | `engine`, `safe_math.hpp` | 19, 2 |
| `nonlinear` | `engine`, `elementwise`, `safe_math.hpp` | 8, 5, 1 |
| `logic` | `engine`, `elementwise` | 16, 2 |
| `reductions` | `engine`, `support` | 11, 10 |
| `experimental` | `engine`, `tensor` | 4, 2 |

Four facts follow from the table.

- **One cycle.** `sparse_builder.hpp` includes `io.hpp`, and `io.hpp` includes `tensor.hpp`. No
  `IO` name appears in the builder's header, so the include may be dead; this is unread.
- **No kernel includes the engine or a tensor.** Every kernel edge goes to `core`, `support` or
  another kernel family.
- **The engine includes no operation family.** `engine.hpp` reaches `tensor`, `core` and its
  own bricks only.
- **Three families lean on a sibling.** `sparse` takes six operand headers from `blas`
  (`argument_check`, `vector_operand`, `matrix_operand`, `scalar`, `task_metadata`, `utils`), and
  `nonlinear` and `logic` take `map_task.hpp` from `elementwise`.

## 2. The module tree

```
modules/
  foundation/   core  support
  data/         tensor  sparse_storage
  kernels/      blas_kernels  random_kernels  sparse_kernels  spatial_kernels
  engine/       engine  operands
  operations/   blas  elementwise  nonlinear  reductions  random  logic  sparse  spatial  io
                experimental
  library/      sushiblas
```

A tier depends only on the tiers above it in this list. Inside a tier, an edge between two
modules is declared in `cmake/` and nothing else is allowed.

| Module | Owns | Comes from | Target |
| --- | --- | --- | --- |
| `core` | Types, traits, logging, formatting, errors, the version header | `include/SushiBLAS/core/`, `version.hpp` | interface |
| `support` | Launch shapes, reductions, accumulators, device policy, geometry | `include/SushiBLAS/support/` | interface |
| `tensor` | `Storage`, `Tensor`, `TensorFactory` | three root headers | interface |
| `sparse_storage` | The CSR matrix, the sparse vector and the builder of both | three root headers, `source/sparse_builder.cpp` | object |
| `blas_kernels` | The reference and SYCL kernels of the BLAS routines | `include/SushiBLAS/kernels/`, `source/kernels/{reference,sycl}/blas/` | interface |
| `random_kernels` | Philox and the distribution kernels | `source/kernels/{reference,sycl}/random/` | interface |
| `sparse_kernels` | The sparse kernels and the structure checks | `source/kernels/sparse/`, `{reference,sycl}/sparse/` | interface |
| `spatial_kernels` | The window, pooling and broadcast kernels | `source/kernels/{reference,sycl}/spatial/` | interface |
| `engine` | `Engine`, `TaskRecorder`, the execution mode, scratch and the random stream | `engine.hpp`, `source/engine.cpp`, the twelve headers directly under `include/SushiBLAS/engine/` | object |
| `operands` | Operand checks and operand views every family shares, the map task and `safe_math` | the six `blas` headers of section 1, `map_task.hpp`, `safe_math.hpp` | interface |
| `blas` and the seven other families | One operation view and its operations | `include/SushiBLAS/engine/<family>/`, `source/engine/<family>/` | object |
| `io` | Host transfer, the three file formats, `IoStatus` | `io.hpp`, `io_status.hpp`, `source/io.cpp` | object |
| `experimental` | The quarantined LAPACK and signal declarations | `include/SushiBLAS/experimental/` | interface |
| `sushiblas` | The umbrella header, the one linked library, the install and export rules | `SushiBLAS.h`, `source/CMakeLists.txt` | the library |

Each module has the shape the layout gives it: `README.md`, `CMakeLists.txt`,
`include/SushiBLAS/<module>/`, `source/` where it has one, and `tests/`.

## 3. What breaks for a consumer

The include path of a header becomes `<SushiBLAS/<module>/<name>.hpp>`.

| Today | After |
| --- | --- |
| `<SushiBLAS/tensor.hpp>`, `storage.hpp`, `tensor_factory.hpp` | `<SushiBLAS/tensor/...>` |
| `<SushiBLAS/sparse_matrix.hpp>`, `sparse_vector.hpp`, `sparse_builder.hpp` | `<SushiBLAS/sparse_storage/...>` |
| `<SushiBLAS/engine.hpp>` | `<SushiBLAS/engine/engine.hpp>` |
| `<SushiBLAS/engine/blas.hpp>`, `<SushiBLAS/engine/blas/...>` | `<SushiBLAS/blas/...>` |
| `<SushiBLAS/engine/math/elementwise/...>` and its three siblings | `<SushiBLAS/elementwise/...>`, and so on |
| `<SushiBLAS/io.hpp>` | `<SushiBLAS/io/io.hpp>` |
| `<SushiBLAS/kernels/...>` | `<SushiBLAS/blas_kernels/...>` |
| `<SushiBLAS/SushiBLAS.h>` | unchanged |
| `find_package(SushiBLAS)` and `SushiBLAS::SushiBLAS` | unchanged |

No name, namespace or signature changes. A consumer that includes only the umbrella header
changes nothing. SushiAI names individual headers and has to follow; a table of old and new
paths goes to `tools/naming/` with the change.

## 4. Decisions that are the owner's

1. **The private kernels become headers another module includes.** The layout forbids reaching
   into another module's `source/`, so the 96 kernel headers under `source/kernels/` move to the
   `include/` tree of their kernel module. Either all of them are installed, which makes them
   reachable by a consumer, or a module is marked internal and its headers stay out of the
   install. The proposal is the second: `blas_kernels` installs the headers a public seam
   instantiates, as today, and the rest are internal.
2. **The `operands` module.** Lifting the six headers out of `blas` removes the `sparse` to
   `blas` edge. The alternative is to declare that edge and leave the headers where they are.
   The six headers are unread for anything specific to BLAS.
3. **One test program per module, or one for all.** The proposal is one per module, each
   registered under the same three CTest labels, so `sb test` and its suites behave as now.
4. **Module names.** `sparse_storage` and the four `*_kernels` names are mine.

## 5. What is assumed and not tried

- Object libraries composed into one shared library, with the interface modules linked for the
  build only, export as the single target `SushiBLAS::SushiBLAS`. `tests/package/` is the proof.
- `cmake/Module.cmake` and `cmake/EngineLayers.cmake` of SushiEngine can be adapted. They are
  unread.
- The export macro and the logging definitions reach every object library through one policy
  function, as they reach the one target today.
- Build time does not get worse. One program per module links the library fifteen times.

## 6. Order of work

Each step ends with one build, and no step starts before the one above it is built. The moves
and the include rewrites are scripted; a step that says "workers" has disjoint file sets.

| Step | Work | Closes with |
| --- | --- | --- |
| 0 | `cmake/Module.cmake`, the tier table, `modules/foundation/core` as the first module, the library target composed from it and the old tree | Build, and `sb test --suite package` |
| 1 | `support`, `tensor`, `sparse_storage`; cut the `io` include of the builder | Build |
| 2 | The four kernel modules, with the internal mark of decision 1 | Build |
| 3 | `engine` and `operands` | Build |
| 4 | The nine operation modules and `experimental`; workers, one per module, each moving its sources and its unit tests | Build, then the full suite |
| 5 | `library/sushiblas`: the umbrella, install and export; remove the old `include/` and `source/` | Build, the package suite |
| 6 | Tier order in `check_layering.py`, every module README, the architecture pages, the old-to-new path table, the changelog | The six checks of `AGENTS.md` |

Integration and regression tests, `benchmark/`, `package/` and `common/` stay under the root
`tests/`. The 175 unit tests go to the module whose header each one includes; a test that
includes two operation modules stays in the root as an integration test.

## 7. What the move did

The owner answered the four decisions of section 4 on 2026-10-09: the kernel modules that no
seam instantiates are internal, the shared operand headers leave `blas`, each module has its own
test program, and the names stand. The tree was then moved in one pass, with no build between
the steps of section 6, by the owner's choice.

It departs from sections 2 and 3 in five places.

| Proposed | Done | Reason |
| --- | --- | --- |
| `blas_kernels` holds the installed and the internal BLAS kernels | `blas_seam_kernels` is installed and `blas_kernels` is internal | A module has one include root, and a root is installed whole or not at all |
| `operands` takes `map_task.hpp` and `safe_math.hpp` | `functor_map` holds the map task and its three bindings; `safe_math.hpp` is in `support` | The map task includes its bindings and one of them is named `scalar.hpp`, as an operand header is; `safe_math.hpp` includes only `support` |
| One target per module | An interface target, and an object library when the module has sources | A program that links an object library gets its objects, so a test that linked the library and a module would get them twice |
| Every unit test goes to a module | Eight unit tests that include a higher tier, and seven that test more than one family, stay under `tests/unit/` | `check_layering.py` reports an upward include wherever it is |
| `<SushiBLAS/kernels/...>` becomes `<SushiBLAS/blas_kernels/...>` | It becomes `<SushiBLAS/blas_seam_kernels/...>` | The first row |

The include of `io.hpp` in `sparse_builder.hpp` was not dead: the builder uses `HostDataType`,
which `io.hpp` declared. The trait moved to `tensor/host_data_type.hpp`, both headers include
it, and the one cycle of section 1 is gone. After the move no operation family includes another, and seven edges stay
inside a tier; `cmake/SushiBLASLayers.cmake` lists them.

`utils.hpp` of `operands` included a BLAS kernel plan, so the module `blas` and `sparse` share
depended on `blas_seam_kernels`. It is now five headers: `scalar_result.hpp` and
`scratch_lease.hpp` in `operands`, and `packed_product_scratch.hpp`, `local_tile.hpp` and
`blas_task.hpp` in `blas`. `operands` no longer names a kernel module.

What was checked on 2026-10-09, on one machine with an RTX 3080 Ti:

- Every translation unit compiled with `-fsyntax-only` under the include roots its own module
  declares: 330 units, no error.
- `sb build` configured the 23 modules, linked the library, 18 test programs and the examples.
- `sb test --suite package` installed the library and built and ran the out-of-tree consumer.
- `sb test` with a filter ran 463 tests, the same 463 the filter found before the move, and all
  passed.
- The six checks of `AGENTS.md` and the 79 tests of the command line pass.
- `sb test` ran the whole suite on the module tree: 2 049 tests, of which 2 014 passed, none
  failed and 35 were skipped.

What was not: the benchmark programs were not built.
`tools/naming/sushiblas_include_paths_0_1_0.tsv` lists the include paths that changed.

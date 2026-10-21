# Module split: evidence

**Status:** Shipped — 2026-10-21

The measurements and the verification of the split of the library into modules. The design
document [Splitting the library into modules](../../design/MODULE_SPLIT.md) took them out when it
took the design format.

## The include graph before the move

Measured on commit `83aaacf` over the 343 C++ files under `include/` and `source/`, with each
file assigned to the candidate module of the design. An edge is one `#include` line that crosses
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

## What was checked after the move

On 2026-10-09, on one machine with an RTX 3080 Ti:

- Every translation unit compiled with `-fsyntax-only` under the include roots its own module
  declares: 330 units, no error.
- `sb build` configured the 23 modules, linked the library, 18 test programs and the examples.
- `sb test --suite package` installed the library and built and ran the out-of-tree consumer.
- `sb test` with a filter ran 463 tests, the same 463 the filter found before the move, and all
  passed.
- The six checks of `AGENTS.md` and the 79 tests of the command line passed.
- `sb test` ran the whole suite on the module tree: 2 049 tests, of which 2 014 passed, none
  failed and 35 were skipped.

The benchmark programs were not built.
`tools/naming/sushiblas_include_paths_0_1_0.tsv` lists the include paths that changed.

The tree was moved in one pass, with no build between the steps of the work order, by the owner's
choice.

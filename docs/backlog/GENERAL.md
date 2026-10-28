# General

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-1 | P2 | decision | Decide the names of `spmv` and of "packed" | the owner has ruled whether the CSR `spmv` or the packed `spmv` is renamed, and what "packed" names besides BLAS packed storage |  |  | 2026-10-19 |
| RW-2 | P2 | decision | Delete `blas/sparse.hpp` or move it under `experimental/` | `modules/operations/blas/include/SushiBLAS/blas/sparse.hpp` is gone or under `experimental/` |  |  | 2026-10-19 |
| RW-3 | P2 | decision | Remove `core/export.hpp` and the `<numbers>` include | neither is in the tree (C13) |  |  | 2026-10-19 |
| RW-4 | P2 | decision | Split `IO` into one reader and writer per format | the owner has approved the boundary, and each format has a reader and a writer over a shared host-copy helper, with a load that returns a status (C4) |  |  | 2026-10-19 |
| RW-5 | P2 | decision | Decide whether ELU accepts complex input | ELU takes complex input, or the owner has ruled it real-only |  |  | 2026-10-19 |
| RW-7 | P2 | decision | Decide whether a green CI on the tagged commit is a release condition | the owner has ruled, and the build jobs build against the tagged runtime |  |  | 2026-10-19 |
| RW-8 | P2 | decision | Tag v0.1.0 | `## Unreleased` is renamed to the release heading `check_version.py` holds to `project()`, and v0.1.0 is tagged (L11) |  |  | 2026-10-19 |
| RW-9 | P2 | decision | Rule how a function-local `constexpr` is spelt | the about 124 local values not spelt `K_UPPER_SNAKE` are renamed, or the owner has ruled that a local keeps `snake_case` |  |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |

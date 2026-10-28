# Sparse

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-55 | P2 | open | Parallelise `sptrsv` by level scheduling | an analysis routine, a type for its result and a solve that takes it exist |  |  | 2026-10-19 |
| RW-56 | P2 | open | Add `sptrsm`, a symmetric `spmm` and a one-triangle builder entry | the three exist with tests |  |  | 2026-10-19 |
| RW-57 | P2 | open | Assemble on the device and convert a dense tensor in a recording | both run as recorded tasks |  |  | 2026-10-19 |
| RW-58 | P2 | open | Parallelise long rows and the transposed `spmm` | a long row gets a work-group and the transposed `spmm` is parallel inside a column |  |  | 2026-10-19 |
| RW-59 | P2 | open | Add CSC and COO operands, sparse-sparse products and mixed index widths | each exists with tests |  |  | 2026-10-19 |
| RW-60 | P2 | open | Add a recordable structure check and a one-device check | a caller can record a structure check, and one call refuses operands on two devices |  |  | 2026-10-19 |
| RW-61 | P2 | open | Call one `SparseOps` routine from the package test | `tests/package/main.cpp` calls one, so the sparse headers are shown to install |  |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |

# KI-9 `blas/sparse.hpp` is installed and used by nothing but the umbrella and one test

The row of this entry is in `docs/backlog/BEHAVIOUR.md`. This file holds what did not fit the row.

- Origin: own
- Where: `modules/operations/blas/include/SushiBLAS/blas/sparse.hpp`
- Cause: it declares a `SparseBLAS` class over a `TaskRecorder&` it never reads, which `SparseOps` replaced
- Recognise: the installed tree carries the header
- Workaround: use `SparseOps`
- Fix: RW-2

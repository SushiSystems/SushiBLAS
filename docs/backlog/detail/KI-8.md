# KI-8 `spmv` and "packed" each name two things

The row of this entry is in `docs/backlog/BEHAVIOUR.md`. This file holds what did not fit the row.

- Origin: own
- Where: `modules/operations/blas/include/SushiBLAS/blas/level2.hpp`, `modules/operations/sparse/include/SushiBLAS/sparse/sparse.hpp`, `modules/kernels/blas_seam_kernels/include/SushiBLAS/blas_seam_kernels/packed_access.hpp`
- Cause: `spmv` is the packed symmetric product on `BLASOps` and the CSR product on `SparseOps`; "packed" is BLAS packed storage and the CPU GEMM's packed panels
- Recognise: one name, two routines or two meanings
- Workaround: qualify the name by its class or module
- Fix: RW-1

# blas_kernels

The reference and SYCL kernels of the BLAS routines that no public seam instantiates. The module is
internal: its headers are not installed. `modules/kernels/README.md` holds the notes of each kernel.

It depends on `core`, `support` and `blas_seam_kernels`. Its headers are included as
`<SushiBLAS/blas_kernels/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `blas.hpp` | Includes every BLAS kernel, reference and SYCL |
| `reference/level1/level1.hpp` | The reference kernels of the BLAS level 1 routines |
| `reference/level1/rotg.hpp` | The reference kernel of ROTG |
| `reference/level1/rotm.hpp` | The reference kernel of ROTM |
| `reference/level1/rotmg.hpp` | The reference kernel of ROTMG |
| `reference/level2/gbmv.hpp` | The reference kernel of GBMV |
| `reference/level2/gemv.hpp` | The reference kernel of GEMV |
| `reference/level2/ger.hpp` | The reference kernel of GER, GERU and GERC |
| `reference/level2/sbmv.hpp` | The reference kernel of SBMV and HBMV |
| `reference/level2/spmv.hpp` | The reference kernel of SPMV and HPMV |
| `reference/level2/spr.hpp` | The reference kernel of SPR and HPR |
| `reference/level2/spr2.hpp` | The reference kernel of SPR2 and HPR2 |
| `reference/level2/symmetric_product.hpp` | The reference symmetric matrix-vector product SBMV and SPMV share |
| `reference/level2/symv.hpp` | The reference kernel of SYMV and HEMV |
| `reference/level2/syr.hpp` | The reference kernel of SYR and HER |
| `reference/level2/syr2.hpp` | The reference kernel of SYR2 and HER2 |
| `reference/level2/tbmv.hpp` | The reference kernel of TBMV |
| `reference/level2/tbsv.hpp` | The reference kernel of TBSV |
| `reference/level2/tpmv.hpp` | The reference kernel of TPMV |
| `reference/level2/tpsv.hpp` | The reference kernel of TPSV |
| `reference/level2/triangular_product.hpp` | The reference triangular matrix-vector product TBMV and TPMV share |
| `reference/level2/triangular_solve.hpp` | The reference triangular substitution TBSV and TPSV share |
| `reference/level2/trmv.hpp` | The reference kernel of TRMV |
| `reference/level3/syr2k.hpp` | The reference kernel of SYR2K and, under a conjugation policy, of HER2K |
| `reference/level3/syrk.hpp` | The reference kernels of SYRK, single and batched |
| `reference/level3/trsm.hpp` | The reference kernels of TRSM, single and batched |
| `sycl/level1/level1.hpp` | The SYCL kernels of the BLAS level 1 routines |
| `sycl/level1/rotg.hpp` | The SYCL kernel of ROTG |
| `sycl/level1/rotm.hpp` | The SYCL kernel of ROTM |
| `sycl/level1/rotmg.hpp` | The SYCL kernel of ROTMG |
| `sycl/level2/gbmv.hpp` | The SYCL kernel of GBMV |
| `sycl/level2/gemv.hpp` | The SYCL kernel of GEMV |
| `sycl/level2/ger.hpp` | The SYCL kernel of GER, GERU and GERC |
| `sycl/level2/row_reduction.hpp` | The work-group-per-row reduction GEMV, SYMV and TRMV share |
| `sycl/level2/sbmv.hpp` | The SYCL kernel of SBMV and HBMV |
| `sycl/level2/spmv.hpp` | The SYCL kernel of SPMV and HPMV |
| `sycl/level2/spr.hpp` | The SYCL kernel of SPR and HPR |
| `sycl/level2/spr2.hpp` | The SYCL kernel of SPR2 and HPR2 |
| `sycl/level2/symmetric_product.hpp` | The SYCL symmetric matrix-vector product SBMV and SPMV share |
| `sycl/level2/symv.hpp` | The SYCL kernel of SYMV and HEMV |
| `sycl/level2/syr.hpp` | The SYCL kernel of SYR and HER |
| `sycl/level2/syr2.hpp` | The SYCL kernel of SYR2 and HER2 |
| `sycl/level2/tbmv.hpp` | The SYCL kernel of TBMV |
| `sycl/level2/tbsv.hpp` | The SYCL kernel of TBSV |
| `sycl/level2/tpmv.hpp` | The SYCL kernel of TPMV |
| `sycl/level2/tpsv.hpp` | The SYCL kernel of TPSV |
| `sycl/level2/triangular_product.hpp` | The SYCL triangular matrix-vector product TBMV and TPMV share |
| `sycl/level2/triangular_solve.hpp` | The SYCL triangular substitution TBSV and TPSV share |
| `sycl/level2/trmv.hpp` | The SYCL kernel of TRMV: a snapshot of x, then one row product per element |
| `sycl/level2/vector_snapshot.hpp` | The copy of a strided vector the in-place triangular products read |
| `sycl/level3/syr2k.hpp` | The SYCL kernel of SYR2K and, under a conjugation policy, of HER2K |
| `sycl/level3/syrk.hpp` | The SYCL kernels of SYRK and HERK, their tile geometry and their packed store |
| `sycl/level3/triangular_plan.hpp` | TriangularSolvePlan, the diagonal block size of a triangular solve on a device |
| `sycl/level3/triangular_system.hpp` | TriangularSystem, the one substitution every side and flag of TRSM decodes to |
| `sycl/level3/trsm.hpp` | The SYCL kernel of TRSM: a blocked substitution with parallel updates |

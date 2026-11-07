# sparse_kernels

The reference and SYCL kernels of the sparse routines, the CSR pattern and index list views, the
structure checks, and the `*_index.hpp` headers that hold the index math the kernels share. Those
headers include no SYCL header and run on the host and in a kernel; the SYCL kernel and the reference
kernel of a family call the same function. The module is internal: its headers are not installed. `modules/kernels/README.md`
holds the notes of each kernel.

It depends on `core`, `support`, `blas_seam_kernels` and `blas_kernels`. Its headers are included as
`<SushiBLAS/sparse_kernels/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `csr_pattern.hpp` | CsrPattern, how a kernel reads the rows and columns of a CSR matrix |
| `index_list.hpp` | IndexList, how a kernel reads the stored positions of a sparse vector |
| `rhs_index.hpp` | The split of a flat product index into a matrix row and a right-hand side, shared by the host and the device |
| `row_block_index.hpp` | The row-block count, the block targets and boundaries, and the partial-sum index of the scattering products |
| `triangle_index.hpp` | The stored-triangle test and the substitution row order of the triangular kernels |
| `reference/level1.hpp` | The reference kernels of the sparse-dense level 1 routines |
| `reference/spmm.hpp` | The reference kernel of the product of a sparse matrix and a dense matrix |
| `reference/spmv.hpp` | The reference kernel of the sparse matrix-vector product |
| `reference/spsymv.hpp` | The reference kernel of the symmetric and Hermitian sparse matrix-vector product |
| `reference/sptrsv.hpp` | The reference kernel of the sparse triangular solve |
| `reference/structure.hpp` | The host checks of the structure a sparse kernel takes on trust |
| `sparse.hpp` | Includes every sparse kernel, reference and SYCL |
| `sycl/level1.hpp` | The SYCL kernels of the sparse-dense level 1 routines |
| `sycl/row_blocks.hpp` | The device shell of the row-block scatter that the transposed and the symmetric product share: the profile lookup and the fold launch |
| `sycl/spmm.hpp` | The SYCL kernel of the product of a sparse matrix and a dense matrix |
| `sycl/spmv.hpp` | The SYCL kernel of the sparse matrix-vector product |
| `sycl/spsymv.hpp` | The SYCL kernel of the symmetric and Hermitian sparse matrix-vector product |
| `sycl/sptrsv.hpp` | The SYCL kernel of the sparse triangular solve |

Kernels with nothing to move: `sycl/level1.hpp` keeps only a load and a store per stored entry, and the
position it reads is `IndexList::at`, tested in `test_index_list.cpp`.

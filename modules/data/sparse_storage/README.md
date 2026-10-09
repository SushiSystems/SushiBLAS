# sparse_storage

The sparse containers: the CSR matrix, the sparse vector and the builder that makes both from host
data.

It depends on `core`, `support` and `tensor`. Its headers are included as
`<SushiBLAS/sparse_storage/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `sparse_builder.hpp` | SparseBuilder, which builds sparse vectors and CSR matrices from host data |
| `sparse_matrix.hpp` | SparseMatrix, a matrix held in compressed sparse row form |
| `sparse_vector.hpp` | SparseVector, a vector held as the positions and values of its stored entries |

`source/` holds 1 translation unit.

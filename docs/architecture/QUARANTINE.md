# 9. Quarantined surface (`modules/operations/experimental/include/SushiBLAS/experimental/`)

`LinalgOps` (SVD, QR, eigendecomposition) and `TransformsOps` (FFT, DFT) are declared under
`modules/operations/experimental/include/SushiBLAS/experimental/` and have no implementation. The umbrella header does not
include them and no view reaches them. The glossary calls this decision D1, and
[the module's README](../../modules/operations/experimental/README.md) holds the notes of each
header.

`blas/sparse.hpp` declares a `SparseBLAS` class with no operation. No header includes it, and the
sparse family is `SparseOps` in `sparse/sparse.hpp`.

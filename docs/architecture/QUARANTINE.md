# 9. Quarantined surface (`include/SushiBLAS/experimental/`)

`LinalgOps` (SVD, QR, eigendecomposition) and `TransformsOps` (FFT, DFT) are declared under
`include/SushiBLAS/experimental/` and have no implementation. The umbrella header does not
include them and no view reaches them. This is decision D1.

`engine/blas/sparse.hpp` still declares a `SparseBLAS` class with no operation from before the sparse family
existed. No header includes it, and the sparse family is `SparseOps` in `engine/sparse.hpp`.

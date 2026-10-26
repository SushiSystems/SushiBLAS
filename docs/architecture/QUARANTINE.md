# 9. Quarantined surface (`modules/operations/experimental/include/SushiBLAS/experimental/`)

`LinalgOps` declares `inv`, `qr`, `svd`, `eigh`, `eig`, `det` and `solve`, which no source
defines. Its LU and GMRES operations and the FFT of `TransformsOps` are defined; the
[module's README](../../modules/operations/experimental/README.md) describes them and their
backend.

`blas/sparse.hpp` declares a `SparseBLAS` class with no operation. No header includes it, and the
sparse family is `SparseOps` in `sparse/sparse.hpp`.

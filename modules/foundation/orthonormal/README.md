# orthonormal

The vector layout and the constants the orthonormalisation kernels of both back ends read, so the reference and the SYCL kernel cannot disagree on them. Its headers are included as `<SushiBLAS/orthonormal/<name>.hpp>`.

It depends on no other module: its headers include SYCL and the standard library only.

| File | Declares |
| --- | --- |
| `orthonormal_sweep.hpp` | The vector layout and the constants the orthonormalisation kernels of both back ends share |

Each section below holds the reasoning behind a header, under the name of the file and of the symbol it belongs to. The header states the rule in one line and cites this page.

## `orthonormal_sweep.hpp`

`VectorSet` is the strided layout of the vectors `RandomOps::orthogonal` orthonormalises: a count,
a length, the step between elements of one vector and the step between vectors.
`K_ORTHONORMAL_SWEEPS` is 2, the number of modified Gram-Schmidt sweeps, and `tiny_norm_sq` is
the squared norm at or below which a vector counts as zero. The reference and the SYCL kernel
read all three from here, so the two back ends cannot disagree on them.

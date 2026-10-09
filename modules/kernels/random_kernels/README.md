# random_kernels

The Philox generator, the distribution kernels and the orthonormalisation sweep. The module is
internal: its headers are not installed. `modules/kernels/README.md` holds the notes of each kernel.

It depends on `core` and `support`. Its headers are included as
`<SushiBLAS/random_kernels/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `random.hpp` | Includes the random kernels: the SYCL ones and the reference orthonormalisation |
| `reference/orthonormalize.hpp` | The reference kernel of the orthonormalisation sweep |
| `sycl/orthonormalize.hpp` | The work-group parallel Gram-Schmidt sweep behind RandomOps::orthogonal |
| `sycl/philox.hpp` | Philox4x32-10 and the uniform, normal and index draws built on it |
| `sycl/poisson.hpp` | The Poisson draw: Knuth's product below lambda 10, Hoermann's PTRS above |
| `sycl/truncated_normal.hpp` | The truncated standard normal draw by bisection of the distribution function |

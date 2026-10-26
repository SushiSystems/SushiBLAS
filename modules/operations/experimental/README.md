# Experimental headers

`modules/operations/experimental/include/SushiBLAS/experimental` holds the dense linear algebra
and transform views. `TransformsOps` records the batched FFT, and `LinalgOps` records the batched
LU factorisation, its solve and restarted GMRES. Each is built from a `TaskRecorder`. Their
sources depend on `operands`, `signal_kernels` and `linalg_kernels`.

| File | Declares |
| --- | --- |
| `dense_operand.hpp` | The record-time operand checks the transforms and the dense solvers share |
| `lapack/linalg.hpp` | `LinalgOps` and `GmresOptions` |
| `signal/transforms.hpp` | `TransformsOps`, `FftShape` and `FftKind` |

## Backend

Every operation runs SushiBLAS's own SYCL kernels, with a reference kernel beside each. No vendor
library backs them: cuFFT and cuSOLVER are NVIDIA SDKs, which the stack does not take, and no
vendor library promises the same order of floating-point operations on every run, which the
engine's lanes need. The owner ruled this on 2026-10-26.

## `signal/transforms.hpp`

`fft` writes the unscaled forward transform of each of `batch` transforms of `rows` by `columns`
elements, and `inverse_fft` its inverse, scaled by `1 / (rows * columns)`. A complex operand is a
`COMPLEX32` or `COMPLEX64` tensor. Under `RealToComplex` the samples are `FLOAT32` or `FLOAT64`
and each spectrum row holds `columns / 2 + 1` elements.

A call cooks an `FftPlan` for its shape at record time: the bit-reversal and twiddle tables, and
for a length that is not a power of two the chirp and padded kernel of Bluestein's transform. The
task keeps the plan and a scratch lease. Its first run on the lease's device copies the tables
to the head of the lease, and every run reads them there and transforms in the rest of the lease
and in the output. The order of every operation is fixed
by the shape, so two runs on one device agree bit for bit.

## `lapack/linalg.hpp`

`lu_factor` factors `P A = L U` in place for each of `batch` row-major `n` by `n` matrices. The
pivot of a column is the row of largest magnitude, the lowest index on a tie. A matrix whose pivot
is not above `n * eps * max|a|` gets `1` in `singular`, the pivots from that step are the identity,
and its factoring stops there. `lu_solve` solves `A X = B` in place over what `lu_factor` wrote.

`gmres` solves one system `A x = b` by restarted GMRES with modified Gram-Schmidt run twice,
Givens rotations, right block-Jacobi preconditioning and the guess in `solution` as its start. It
writes the inner iterations taken and the final true relative residual `|b - A x| / |b|`. The
workspace is leased once at record time. On the device one work-group runs the solve. Every dot
product sums chunks of 256 elements in index order, then the chunk sums in order, so the result
is the same on every device and in the engine's native lane.

`LinalgOps` also declares `inv`, `qr`, `svd`, `eigh`, `eig`, `det` and `solve`, which no source
defines. The umbrella header includes both views.

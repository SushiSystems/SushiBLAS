# linalg_kernels

The reference and SYCL kernels of the dense solvers. The module is internal: its headers are not
installed.

It depends on `arithmetic` and `launch`. Its headers are included as `<SushiBLAS/linalg_kernels/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `chunked_dot.hpp` | `chunked_dot` and `K_DOT_CHUNK`, the one summation order of every GMRES dot product |
| `gmres_workspace.hpp` | `GmresSettings`, the stopping rule, and `GmresLayout`, where each array of the workspace starts |
| `linalg.hpp` | Includes every linear algebra kernel, reference and SYCL |
| `reference/gmres.hpp` | The reference restarted, block-Jacobi preconditioned GMRES solve |
| `reference/lu.hpp` | The reference batched LU factorisation and its solve |
| `sycl/gmres.hpp` | The SYCL GMRES solve, run by one work-group |
| `sycl/lu.hpp` | The SYCL batched LU, one work-group per matrix, and its solve, one work-item per right-hand side |

Every element of a factored matrix takes its updates in ascending step order on both sides, and
every GMRES dot product follows `chunked_dot`, so neither result depends on the work-group width.

## Bitwise agreement

The FFT kernels of `signal_kernels` and the LU and GMRES kernels here give the same bits on every
device, and the engine's native lane is held to the same bits, so a result depends on nothing but
its inputs. Three rules hold that, and the native kernels follow them too:

- None of them lets the compiler fuse a product into a sum. Each product is its own statement,
  rounded before it is added, and a block that computes opens with `SB_FP_CONTRACT_OFF`
  (`arithmetic/no_contraction.hpp`), so no build flag changes the result. A fused step a kernel
  wants is written as `sycl::fma` or `std::fma`, at the same point in both lanes; these three
  kernels have none.
- Every sum has one order, fixed by the extents. A GMRES dot product sums chunks of
  `K_DOT_CHUNK` (256) elements in index order, then the chunk sums in index order, whatever the
  work-group width.
- The FFT tables are cooked on the host, once per plan, and the device reads them rather than
  computing a sine or a cosine.

`lu_factor` flags a matrix singular on both lanes by one rule: a pivot not above
`n * eps * max|a|`, with `max|a|` taken over the input matrix before the first step and the
product formed left to right. An exact-zero test would pass a matrix singular only to rounding.

The rules assume IEEE division and square root, correctly rounded, on the device. A build that
lowers either to an approximation breaks the agreement.

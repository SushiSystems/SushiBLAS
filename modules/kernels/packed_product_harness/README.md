# packed_product_harness

The problem, the data and the blockings the packed product tests share. The module is internal and test-only: its headers are not installed, and it is added
only when `SB_BUILD_TESTS` is on.

It depends on `core`, `arithmetic`, `launch` and `blas_seam_kernels`. Its headers are included as `<SushiBLAS/packed_product_harness/packed_product_harness.hpp>`.

| File | Declares |
| --- | --- |
| `packed_product_harness.hpp` | `GemmProblem<T>` and the three blockings of the packed GEMM tests |

## packed_product_harness.hpp

`GemmProblem<T>` owns the three operands of one batched GEMM in shared USM, with leading
dimensions and batch strides larger than the extents, and runs the reference kernel, the packed
kernel under a blocking a test chooses, and the kernel that takes no scratch on the same data.
Each run returns the whole C buffer, padding included, so a test can check that no kernel wrote
outside its matrices. After `poison_output()` C starts each run as NaN, so an element a kernel
failed to write shows.

The three blockings are the edges of the plan: every block one register block and one term deep,
every block larger than any matrix, and a depth of 7 with two register blocks of rows and three
of columns. Integer-valued data makes every sum exact, so the kernels can be compared bit for
bit whatever order they add in.

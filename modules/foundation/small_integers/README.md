# small_integers

The integer-valued operands a kernel test fills its inputs with. The module is internal and
test-only: its headers are not installed, and it is added only when `SB_BUILD_TESTS` is on.

It depends on the standard library only. Its header is included as
`<SushiBLAS/small_integers/small_integers.hpp>`.

| File | Declares |
| --- | --- |
| `small_integers.hpp` | `small_integers`, values from -3 to 3, and `unit_diagonal_matrix` |

## small_integers.hpp

A reference kernel accumulates in double and a SYCL kernel in the element type, so on arbitrary
inputs the two differ in their last bits. On values from -3 to 3, every product and every partial
sum of a test-sized operand is an integer below 2^24, which FLOAT32 and FLOAT64 hold exactly. Both
kernels then compute without rounding, and a test compares them with exact equality.
`unit_diagonal_matrix` puts ones on the diagonal, so a triangle of it solves to integers as well.

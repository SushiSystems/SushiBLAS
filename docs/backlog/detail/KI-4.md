# KI-4 Some random families differ between the two paths

The row of this entry is in `docs/backlog/BEHAVIOUR.md`. This file holds what did not fit the row.

- Origin: own
- Where: `modules/kernels/random_kernels/include/SushiBLAS/random_kernels/sycl/`, `modules/kernels/random_kernels/include/SushiBLAS/random_kernels/reference/orthonormalize.hpp`
- Cause: `poisson`, the normal family, `truncated_normal` and `orthogonal` call each path's own math library, and `orthogonal` also sums in another order
- Recognise: those draws are not bit-identical between the SYCL and the reference path
- Workaround: compare them within the allowances of `test_random_reference.cpp`
- Fix: none, each path calls its own math library

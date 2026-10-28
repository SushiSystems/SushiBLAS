# KI-5 The OpenCL CPU runtime crashes compiling `TrmvRow<std::complex<float>>`

The row of this entry is in `docs/backlog/BEHAVIOUR.md`. This file holds what did not fit the row.

- Origin: toolchain
- Where: `modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level2/trmv.hpp`, `.github/workflows/ci.yml`
- Cause: the vectorizer pass `vplan-vec` of oneAPI 2026.1's OpenCL CPU runtime (device runtime 2026.21.7.0.24_160000, AMD EPYC 7763) crashes in the JIT
- Recognise: every band and packed triangular test on that device then segfaults
- Workaround: none known; taking the diagonal out of the row loop did not avoid it

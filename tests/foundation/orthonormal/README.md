# Tests of orthonormal

The compile check of the `orthonormal` module, listed under `COMPILE_CHECKS` of
`modules/foundation/orthonormal/CMakeLists.txt`: the build compiles it and no test runner lists
it. The sweep's behaviour is tested by `tests/kernels/random_kernels/reference/test_orthonormalize.cpp`.

| File | Tests |
| --- | --- |
| `compile_orthonormal_sweep.cpp` | Checks that the header stands alone and that the tiny-norm threshold has the element type |

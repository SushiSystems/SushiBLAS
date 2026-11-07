# Tests of orthonormal

The tests of the `orthonormal` module. `compile_orthonormal_sweep.cpp` is listed under `COMPILE_CHECKS` of
`modules/foundation/orthonormal/CMakeLists.txt`: the build compiles it and no test runner lists
it. The sweep's behaviour is tested by `tests/kernels/random_kernels/reference/test_orthonormalize.cpp`.

| File | Tests |
| --- | --- |
| `test_orthonormal_sweep.cpp` | Tests the projection coefficient, the normalising scale and the tiny-norm rule they share, on the host |
| `compile_orthonormal_sweep.cpp` | Checks that the header stands alone and that the tiny-norm threshold has the element type |

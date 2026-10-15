# Tests of nonlinear

The tests of the `nonlinear` module, built into `sushiblas_nonlinear_tests` by
`sushiblas_add_test_program()` in `cmake/SushiBLASTests.cmake`. The source comments state
what a test checks; a section below holds why a test is written the way it is.
[`tests/README.md`](../../README.md) holds the reasoning the tests share, and
`modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_dispatch.cpp` | Tests a forward task in place and a backward task with a bound scalar in both modes |
| `test_nonlinear/test_complex_nonlinear.cpp` | Tests complex sigmoid, tanh, sqrt and log against hand values, and elu's refusal |
| `test_nonlinear/test_elu.cpp` | Tests NonLinearOps::elu and elu_backward against fixed expected values |
| `test_nonlinear/test_gelu.cpp` | Tests NonLinearOps::gelu and gelu_backward against fixed expected values |
| `test_nonlinear/test_leaky_relu.cpp` | Tests NonLinearOps::leaky_relu and leaky_relu_backward against fixed expected values |
| `test_nonlinear/test_relu.cpp` | Tests NonLinearOps::relu and relu_backward against fixed expected values |
| `test_nonlinear/test_sigmoid.cpp` | Tests NonLinearOps::sigmoid and sigmoid_backward against fixed expected values |
| `test_nonlinear/test_silu.cpp` | Tests NonLinearOps::silu and silu_backward against fixed expected values |
| `test_nonlinear/test_softplus.cpp` | Tests NonLinearOps::softplus and softplus_backward against fixed expected values |
| `test_nonlinear/test_tanh.cpp` | Tests NonLinearOps::tanh and tanh_backward against fixed expected values |

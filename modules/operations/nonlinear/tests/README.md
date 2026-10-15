# Tests of nonlinear

The unit tests of the `nonlinear` module, built into `sushiblas_nonlinear_tests` by the
module's `CMakeLists.txt`. The source comments state what a test checks; a section below
holds why a test is written the way it is. `tests/unit/README.md` holds the reasoning the
tests share, and `modules/engine/engine_fixture/README.md` describes the fixture.

| File | Tests |
| --- | --- |
| `test_complex_nonlinear.cpp` | Tests complex sigmoid, tanh, sqrt and log against hand values, and elu's refusal |
| `test_elu.cpp` | Tests NonLinearOps::elu and elu_backward against fixed expected values |
| `test_gelu.cpp` | Tests NonLinearOps::gelu and gelu_backward against fixed expected values |
| `test_leaky_relu.cpp` | Tests NonLinearOps::leaky_relu and leaky_relu_backward against fixed expected values |
| `test_relu.cpp` | Tests NonLinearOps::relu and relu_backward against fixed expected values |
| `test_sigmoid.cpp` | Tests NonLinearOps::sigmoid and sigmoid_backward against fixed expected values |
| `test_silu.cpp` | Tests NonLinearOps::silu and silu_backward against fixed expected values |
| `test_softplus.cpp` | Tests NonLinearOps::softplus and softplus_backward against fixed expected values |
| `test_tanh.cpp` | Tests NonLinearOps::tanh and tanh_backward against fixed expected values |

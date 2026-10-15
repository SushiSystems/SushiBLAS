# tensor_expectations

Writing host values into a dense tensor, reading them back, and comparing a tensor with host
expectations. The module is internal and test-only: its headers are not
installed, and it is added only when `SB_BUILD_TESTS` is on.

It depends on `core`, `tensor`, `test_runner` and GoogleTest. Its headers are included as `<SushiBLAS/tensor_expectations/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `scalar_dtype.hpp` | `sushiblas_dtype`, the `DataType` of a C++ element type |
| `tensor_fill.hpp` | `fill_tensor`, `fill_random` and the generator `fill_random` draws from |
| `tensor_comparison.hpp` | `verify_tensor`, `assert_tensor_near`, `EXPECT_TENSOR_NEAR` and `SB_ASSERT_TENSOR_NEAR` |
| `tensor_readback.hpp` | `read_tensor`, which copies a tensor's elements into a host vector |

The fill, comparison and readback headers copy through `TestEnvironment::queue()` and wait, so a
call returns with the copy done.

## The complex-type trait

The helpers use the library's own trait, `SushiBLAS::Core::is_complex_v` from
`<SushiBLAS/core/traits.hpp>`, the one definition in the project.

## Dense tensors only

Every helper here reads or writes `num_elements` consecutive elements from `tensor.data()`. They
handle dense, contiguous tensors only. A strided view or a view at an offset needs its own
comparison, as `tests/library/umbrella/test_SushiBLAS/test_storage_offset.cpp` does.

## tensor_fill.hpp

### fill_tensor on integer tensors

The `INT32` and `INT64` arms truncate, like every other arm: the caller states the values it
wants and the tensor's dtype decides how they are stored. Filling an integer tensor from
`{1.0f, 0.0f}` is the natural way to write a comparison operand or an expected truth vector.

### fill_random

`fill_random` draws from one `std::mt19937` seeded with `K_FILL_RANDOM_SEED`. `SushiBLASTest`
restarts it in `SetUp`, so a test sees the same inputs whether ctest runs it alone or the test
binary runs the whole suite. Two calls in one test draw different values.

A complex element takes two draws, the real part first and the imaginary part second.

On a floating dtype the values lie in `[min_val, max_val]`, not `[min_val, max_val)`: the draw
is a `float` below `max_val`, and rounding it to `HALF` (or to `float` from a nearby double)
can land on `max_val` itself.

On an integer dtype each draw is truncated toward zero. With the default range `(-1, 1)` every
element is 0. A test that needs varied integers passes a wider range or uses `fill_tensor`.

`read_tensor<T>` copies a tensor's elements into a host vector for checks that `verify_tensor`
cannot express, such as sample statistics. It throws `std::invalid_argument` when `T` does not
map to the tensor's dtype, before it copies anything.

## tensor_comparison.hpp

### verify_tensor

`verify_tensor` takes a `std::vector<float>` or, through a template overload, a
`std::vector<double>`, `std::vector<std::complex<float>>` or `std::vector<std::complex<double>>`.
A braced list such as `{1.0f, 2.0f}` picks the `float` overload, because a braced list cannot
deduce the template's element type.

On a complex tensor both parts are compared. A real expectation asks for a zero imaginary part.

A NaN expectation passes only on a NaN element, and an infinite expectation only on the same
infinity. `EXPECT_NEAR` alone fails both, since the difference it computes is NaN.

Integer elements are compared with `EXPECT_EQ`. An integer element is the expected value or it
is not, and a tolerance would only hide an off-by-one.

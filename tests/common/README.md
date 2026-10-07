# Shared test support

Support code the functional tests include.

| File | What it holds |
| --- | --- |
| `reference_reductions.hpp` | Serial reductions over a host vector, in namespace `ReferenceKernels` |
| `test_environment.hpp` | The GoogleTest environment that owns the one `RuntimeContext` |
| `test_main.cpp` | `main`: synchronous logging, the environment, then `RUN_ALL_TESTS` |
| `test_utils.hpp` | Tensor fill and comparison helpers, the `SushiBLASTest` fixture, the suite macros |

## test_utils.hpp

### The complex-type trait

The helpers use the library's own trait, `SushiBLAS::Core::is_complex_v` from
`<SushiBLAS/core/traits.hpp>`. This file once carried its own copy, the fourth in the project.
The project now has exactly one.

### fill_tensor on integer tensors

The `INT32` and `INT64` arms truncate, like every other arm: the caller states the values it
wants and the tensor's dtype decides how they are stored. Filling an integer tensor from
`{1.0f, 0.0f}` is the natural way to write a comparison operand or an expected truth vector.

### Dense tensors only

Every helper here reads or writes `num_elements` consecutive elements from `tensor.data()`. They
handle dense, contiguous tensors only. A strided view or a view at an offset needs its own
comparison, as `tests/functional/unit/test_storage_offset.cpp` does.

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

### DefaultTypeNames and SB_TYPED_TEST_SUITE

`TYPED_TEST_SUITE(Suite, Types)` looks like a two-argument macro and is declared
`TYPED_TEST_SUITE(CaseName, Types, ...)`. The `...` is an optional name generator. Passing no
argument at all for a `...` is allowed from C++20 and is a diagnosed extension before it. On this
project's C++17 baseline the two-argument spelling is an error under `-pedantic -Werror`, raised
inside a third-party header the project does not own and cannot fix.

Naming a generator explicitly settles it. `DefaultTypeNames` restates GoogleTest's default
against its public API, without reaching into its `internal` namespace. The suffix is the type's
index, as before, so every generated test name is unchanged and the fix renames no test.

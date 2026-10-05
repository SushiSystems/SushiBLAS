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

### verify_tensor on integer tensors

Integer elements are compared with `EXPECT_EQ`. An integer element is the expected value or it
is not, and a tolerance would only hide an off-by-one.

### DefaultTypeNames and SB_TYPED_TEST_SUITE

`TYPED_TEST_SUITE(Suite, Types)` looks like a two-argument macro and is declared
`TYPED_TEST_SUITE(CaseName, Types, ...)`. The `...` is an optional name generator. Passing no
argument at all for a `...` is allowed from C++20 and is a diagnosed extension before it. On this
project's C++17 baseline the two-argument spelling is an error under `-pedantic -Werror`, raised
inside a third-party header the project does not own and cannot fix.

Naming a generator explicitly settles it. `DefaultTypeNames` restates GoogleTest's default
against its public API, without reaching into its `internal` namespace. The suffix is the type's
index, as before, so every generated test name is unchanged and the fix renames no test.

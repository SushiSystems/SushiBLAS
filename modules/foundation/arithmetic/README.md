# arithmetic

The scalar arithmetic a kernel and the host share: the type a BLAS kernel accumulates in, the complex conjugation, product and quotient a SYCL device can run, the real and complex safe math, and `Math`, the scalar functions that are `std::` on the host and `sycl::` on the device. Its headers are included as `<SushiBLAS/arithmetic/<name>.hpp>`.

It depends on no other module: its headers include SYCL and the standard library only.

| File | Declares |
| --- | --- |
| `complex_arithmetic.hpp` | The conjugation, product and quotient a SYCL device can run on any element type |
| `safe_math.hpp` | `safe_exp`, `safe_tanh`, `safe_log`, `safe_sqrt`, `safe_erf` and `safe_abs`, each with a real and a complex overload |
| `device_math.hpp` | `Math`, the scalar functions a kernel may call: `std::` on the host, `sycl::` on the device |
| `accumulator.hpp` | `AccumulatorOf`, the type a BLAS kernel accumulates an element type in |

Each section below holds the reasoning behind a header, under the name of the file and of the symbol it belongs to. The header states the rule in one line and cites this page.

## `complex_arithmetic.hpp`

`std::complex` multiplication and division are host library calls on some toolchains, and a SYCL
device cannot call them. This header spells both on the real and imaginary parts: the product by
the four-product formula, the quotient by Smith's algorithm, which returns NaN components for a
zero divisor. Each has a real overload that is the plain operator, so a kernel template calls one
name for every element type. `RealType<T>` names the real component type, and the conjugation
helpers return a real value unchanged.

## `safe_math.hpp`

Scalar math that stays well-formed for both real and complex element types.

The problem these solve: an op functor handed to a Dispatch executor is a generic lambda,
instantiated once per dtype the executor's Domain admits. A body written as `sycl::exp(x)` is fine
for HALF/FLOAT32/FLOAT64 and a hard compile error for `std::complex`, because the SYCL math builtins
only accept device scalar types. Writing `safe_exp(x)` instead picks `sycl::exp` for real scalars
and the complex overload for complex ones by overload resolution, so one functor covers
Domain::RealAndComplex without any `if constexpr` at the call site.

The complex overloads of `safe_exp`, `safe_tanh`, `safe_log`, `safe_sqrt`, `safe_erf` and
`safe_abs` compute on the real and imaginary parts with `sycl::` real math only. They never call
the `std::` complex functions, because MSVC implements those through host-only helpers
(`_FExp`, `_FSinh`) that a SYCL device cannot call. `safe_exp` and `safe_tanh` return a zero
imaginary part unchanged, sign included, when the input's imaginary part is zero.

Only operations with an unambiguous complex meaning appear here. There is deliberately no `safe_max`
/ `safe_floor` / `safe_clamp`: those need a total order on the element type, complex numbers do not
have one, and inventing a lexicographic stand-in would be a silently-wrong answer rather than a
missing feature. Ops built on them declare Domain::Real or Domain::RealAndIntegral and
throw for complex inputs.

Lived in src/engine/math/nonlinear/nonlinear_internal.hpp before the dispatch seam went public;
hoisted here so the elementwise and nonlinear families share one definition instead of growing a
second copy each.

### `safe_erf` for a complex value

The complex error function is not elementary and no activation in this library needs it; GELU only
ever reaches `safe_erf` with a real-valued argument.

### `safe_abs` for a complex value

The result is real-valued, but elementwise `abs` is an in-place op writing back into the tensor's
own complex storage, so the magnitude is returned with a zero imaginary part rather than as a bare
scalar.

## `device_math.hpp`

`SushiBLAS::Math` names the scalar functions that code shared by the host and a kernel may call:
`abs`, `asin`, `atan2`, `cbrt`, `copysign`, `cos`, `exp`, `floor`, `fmod`, `isfinite`, `log`,
`pow`, `sin` and `sqrt`. On the host each is the `std::` function. In the device pass of a SYCL
compiler each is the `sycl::` function, and `abs` is a template over `sycl::fabs`. A function
written once against `Math::sqrt` therefore runs in both places.

The switch is `__SYCL_DEVICE_ONLY__`, which the compiler defines only while it compiles device
code. Every host translation unit sees the `std::` names, whether it is a SYCL one or not, so the
header gives one definition across a program's host objects.

`<sycl/sycl.hpp>` is included only when `SYCL_LANGUAGE_VERSION` is defined. A translation unit
built without SYCL includes `<cmath>` and nothing else, which is what lets
`modules/foundation/geometry` compile in a consumer's build lane that has no SYCL.

## `accumulator.hpp`

### `AccumulatorOf`

`AccumulatorOf<T>::type`, shortened to `Accumulator<T>`, names the type a BLAS kernel sums `T`
values in. It is `float` for `sycl::half` and `T` for every other element type. HALF has 11
significant bits, so a HALF running sum of ones stops growing at 2048: from there the spacing
between HALF values is 2, and 2048 + 1 rounds back to 2048.

The SYCL kernels of GEMM, SYRK, TRSM (and TRSV through it), GEMV, SYMV, TRMV, GER, SYR and SYR2,
and the reference kernels of GER, SYR and SYR2, widen each operand to `Accumulator<T>`, form the
sum in it and round to `T` once, when the element is stored. GEMM's epilogue still receives a `T`.
TRSM and TRSV read the solution elements they already solved through `RunningSolution`, which
hands a later row or column the unrounded value.
For every type but HALF the widening is the identity and the kernels run the same operations in the
same order as before. `Reduce::Accumulation` and `Reduce::AccumulationOf` alias this trait.

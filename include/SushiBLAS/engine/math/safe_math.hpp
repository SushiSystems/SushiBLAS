/**************************************************************************/
/* safe_math.hpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

#include <cmath>
#include <complex>
#include <sycl/sycl.hpp>

// Scalar math that stays well-formed for both real and complex element types.
//
// The problem these solve: an op functor handed to a Dispatch executor is a
// generic lambda, instantiated once per dtype the executor's Domain admits. A
// body written as `sycl::exp(x)` is fine for HALF/FLOAT32/FLOAT64 and a hard
// compile error for `std::complex`, because the SYCL math builtins only accept
// device scalar types. Writing `safe_exp(x)` instead picks `sycl::exp` for real
// scalars and `std::exp` for complex ones by overload resolution, so one functor
// covers Domain::REAL_AND_COMPLEX without any `if constexpr` at the call site.
//
// Only operations with an unambiguous complex meaning appear here. There is
// deliberately no `safe_max` / `safe_floor` / `safe_clamp`: those need a total
// order on the element type, complex numbers do not have one, and inventing a
// lexicographic stand-in would be a silently-wrong answer rather than a missing
// feature. Ops built on them declare Domain::REAL and throw for complex inputs.
//
// Lived in src/engine/math/nonlinear/nonlinear_internal.hpp before the dispatch
// seam went public; hoisted here so the elementwise and nonlinear families share
// one definition instead of growing a second copy each.

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Identity for a real scalar; the counterpart to the complex overload.
         *
         * @param x The value to project.
         * @return `x` unchanged.
         */
        template<typename T>
        inline T real_part(T x) { return x; }

        /**
         * @brief Drops a complex value's imaginary component, keeping the complex type.
         *
         * @param x The value to project.
         * @return `x.real()` re-wrapped as a complex number with zero imaginary part.
         */
        template<typename T>
        inline std::complex<T> real_part(const std::complex<T>& x) { return std::complex<T>(x.real(), 0); }

        /**
         * @brief Widens a real scalar to the element type in use; identity for real types.
         *
         * @param val The value to widen.
         * @return `val` unchanged.
         */
        template<typename T>
        inline T make_complex_safe(T val, T) { return val; }

        /**
         * @brief Widens a real scalar into a complex element type.
         *
         * @param val The real value to widen.
         * @return `val` as a complex number with zero imaginary part.
         */
        template<typename T, typename U>
        inline std::complex<T> make_complex_safe(U val, const std::complex<T>&) { return std::complex<T>(val, 0); }

        /**
         * @brief Sign test for a real scalar.
         *
         * @param x The value to test.
         * @return True if `x > 0`.
         */
        template<typename T>
        inline bool greater_than_zero(T x) { return x > T(0); }

        /**
         * @brief Sign test for a complex value, taken on the real part.
         *
         * @param x The value to test.
         * @return True if `x.real() > 0`.
         */
        template<typename T>
        inline bool greater_than_zero(const std::complex<T>& x) { return x.real() > T(0); }

        /**
         * @brief Sign test for a real scalar.
         *
         * @param x The value to test.
         * @return True if `x < 0`.
         */
        template<typename T>
        inline bool less_than_zero(T x) { return x < T(0); }

        /**
         * @brief Sign test for a complex value, taken on the real part.
         *
         * @param x The value to test.
         * @return True if `x.real() < 0`.
         */
        template<typename T>
        inline bool less_than_zero(const std::complex<T>& x) { return x.real() < T(0); }

        /**
         * @brief Exponential of a real scalar.
         *
         * @param x The exponent.
         * @return `e**x`.
         */
        template<typename T>
        inline T safe_exp(T x) { return sycl::exp(x); }

        /**
         * @brief Exponential of a complex value.
         *
         * @param x The exponent.
         * @return `e**x`.
         */
        template<typename T>
        inline std::complex<T> safe_exp(const std::complex<T>& x) { return std::exp(x); }

        /**
         * @brief Hyperbolic tangent of a real scalar.
         *
         * @param x The argument.
         * @return `tanh(x)`.
         */
        template<typename T>
        inline T safe_tanh(T x) { return sycl::tanh(x); }

        /**
         * @brief Hyperbolic tangent of a complex value.
         *
         * @param x The argument.
         * @return `tanh(x)`.
         */
        template<typename T>
        inline std::complex<T> safe_tanh(const std::complex<T>& x) { return std::tanh(x); }

        /**
         * @brief Error function of a real scalar.
         *
         * @param x The argument.
         * @return `erf(x)`.
         */
        template<typename T>
        inline T safe_erf(T x) { return sycl::erf(x); }

        /**
         * @brief Error function of a complex value, evaluated on the real part only.
         *
         * The complex error function is not elementary and no activation in this library
         * needs it; GELU only ever reaches this overload with a real-valued argument.
         *
         * @param x The argument.
         * @return `erf(x.real())` with zero imaginary part.
         */
        template<typename T>
        inline std::complex<T> safe_erf(const std::complex<T>& x) { return std::complex<T>(std::erf(x.real()), 0); }

        /**
         * @brief Natural logarithm of a real scalar.
         *
         * @param x The argument.
         * @return `log(x)`.
         */
        template<typename T>
        inline T safe_log(T x) { return sycl::log(x); }

        /**
         * @brief Principal natural logarithm of a complex value.
         *
         * @param x The argument.
         * @return `log(x)`, branch cut along the negative real axis.
         */
        template<typename T>
        inline std::complex<T> safe_log(const std::complex<T>& x) { return std::log(x); }

        /**
         * @brief Square root of a real scalar.
         *
         * @param x The argument.
         * @return `sqrt(x)`.
         */
        template<typename T>
        inline T safe_sqrt(T x) { return sycl::sqrt(x); }

        /**
         * @brief Principal square root of a complex value.
         *
         * @param x The argument.
         * @return `sqrt(x)`, the root with non-negative real part.
         */
        template<typename T>
        inline std::complex<T> safe_sqrt(const std::complex<T>& x) { return std::sqrt(x); }

        /**
         * @brief Absolute value of a real scalar.
         *
         * @param x The argument.
         * @return `|x|`.
         */
        template<typename T>
        inline T safe_abs(T x) { return sycl::fabs(x); }

        /**
         * @brief Modulus of a complex value, kept in the complex element type.
         *
         * The result is real-valued, but elementwise `abs` is an in-place op writing back
         * into the tensor's own complex storage, so the magnitude is returned with a zero
         * imaginary part rather than as a bare scalar.
         *
         * @param x The argument.
         * @return `(|x|, 0)`.
         */
        template<typename T>
        inline std::complex<T> safe_abs(const std::complex<T>& x)
        {
            return std::complex<T>(sycl::sqrt(x.real() * x.real() + x.imag() * x.imag()), 0);
        }

    } // namespace Dispatch
} // namespace SushiBLAS

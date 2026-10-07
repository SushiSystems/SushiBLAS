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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file safe_math.hpp
 * @brief Declares scalar math helpers that stay well-formed for real and complex element types.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <limits>
#include <complex>
#include <sycl/sycl.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>

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
        inline std::complex<T> safe_exp(const std::complex<T>& x)
        {
            const T re = x.real();
            const T im = x.imag();
            if (im == T(0))
                return std::complex<T>(sycl::exp(re), im);
            const T magnitude = sycl::exp(re);
            return std::complex<T>(magnitude * sycl::cos(im), magnitude * sycl::sin(im));
        }

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
        inline std::complex<T> safe_tanh(const std::complex<T>& x)
        {
            const T re = x.real();
            const T im = x.imag();
            if (im == T(0))
                return std::complex<T>(sycl::tanh(re), im);
            // Beyond this |re|, tanh(re) rounds to +-1 in double and cosh(re)**2 is still finite.
            constexpr T K_SATURATION = T(22);
            if (sycl::fabs(re) > K_SATURATION)
            {
                const T decay = sycl::exp(T(-2) * sycl::fabs(re));
                return std::complex<T>(sycl::copysign(T(1), re), T(2) * sycl::sin(T(2) * im) * decay);
            }
            const T tangent = sycl::tan(im);
            const T secant_squared = T(1) + tangent * tangent;
            const T hyperbolic_sine = sycl::sinh(re);
            const T hyperbolic_cosine = sycl::sqrt(T(1) + hyperbolic_sine * hyperbolic_sine);
            const T denominator = T(1) + secant_squared * hyperbolic_sine * hyperbolic_sine;
            return std::complex<T>(secant_squared * hyperbolic_cosine * hyperbolic_sine / denominator,
                                   tangent / denominator);
        }

        /**
         * @brief Error function of a real scalar.
         *
         * @param x The argument.
         * @return `erf(x)`.
         */
        template<typename T>
        inline T safe_erf(T x) { return sycl::erf(x); }

        /**
         * @brief Returns the error function of the real part of a complex value, with zero imaginary part.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename T>
        inline std::complex<T> safe_erf(const std::complex<T>& x) { return std::complex<T>(sycl::erf(x.real()), 0); }

        /**
         * @brief Natural logarithm of a real scalar.
         *
         * @param x The argument.
         * @return `log(x)`.
         */
        template<typename T>
        inline T safe_log(T x) { return sycl::log(x); }

        /**
         * @brief Returns the principal natural logarithm of x, branch cut along the negative
         * real axis.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename T>
        inline std::complex<T> safe_log(const std::complex<T>& x)
        {
            const T re = x.real();
            const T im = x.imag();
            const T modulus = sycl::hypot(re, im);
            const T log_modulus = (sycl::fabs(modulus - T(1)) < T(0.5))
                                      ? T(0.5) * sycl::log1p((re - T(1)) * (re + T(1)) + im * im)
                                      : sycl::log(modulus);
            return std::complex<T>(log_modulus, sycl::atan2(im, re));
        }

        /**
         * @brief Square root of a real scalar.
         *
         * @param x The argument.
         * @return `sqrt(x)`.
         */
        template<typename T>
        inline T safe_sqrt(T x) { return sycl::sqrt(x); }

        /**
         * @brief Returns the principal square root of x, the root with non-negative real part.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename T>
        inline std::complex<T> safe_sqrt(const std::complex<T>& x)
        {
            const T re = x.real();
            const T im = x.imag();
            if (sycl::isinf(im))
                return std::complex<T>(std::numeric_limits<T>::infinity(), im);
            if (re == T(0) && im == T(0))
                return std::complex<T>(T(0), im);
            const T root = sycl::sqrt((sycl::fabs(re) + sycl::hypot(re, im)) * T(0.5));
            if (re >= T(0))
                return std::complex<T>(root, im / (T(2) * root));
            return std::complex<T>(sycl::fabs(im) / (T(2) * root), sycl::copysign(root, im));
        }

        /**
         * @brief Absolute value of a real scalar.
         *
         * @param x The argument.
         * @return `|x|`.
         */
        template<typename T>
        inline T safe_abs(T x) { return sycl::fabs(x); }

        /**
         * @brief Returns the modulus of a complex value as `(|x|, 0)` in the complex element type.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename T>
        inline std::complex<T> safe_abs(const std::complex<T>& x)
        {
            return std::complex<T>(sycl::hypot(x.real(), x.imag()), 0);
        }

        /**
         * @brief Returns the complex conjugate of x; a real x is returned unchanged.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename T>
        [[nodiscard]] inline T safe_conj(T x) noexcept
        {
            return Kernels::conjugate(x);
        }

        /**
         * @brief Returns `a * b` without calling a std::complex operator.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename T>
        [[nodiscard]] inline T safe_mul(T a, T b) noexcept
        {
            return Kernels::multiply(a, b);
        }

        /**
         * @brief Returns `a / b` without calling a std::complex operator.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename T>
        [[nodiscard]] inline T safe_div(T a, T b) noexcept
        {
            return Kernels::divide(a, b);
        }

    } // namespace Dispatch
} // namespace SushiBLAS

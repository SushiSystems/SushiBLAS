/****************************************************************/
/* safe_math.hpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file safe_math.hpp
 * @brief Declares scalar math helpers that stay well-formed for real and complex element types.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <complex>
#include <sycl/sycl.hpp>

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
         * @brief Returns the error function of the real part of a complex value, with zero imaginary part.
         * @see include/SushiBLAS/engine/math/README.md
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
         * @brief Returns the modulus of a complex value as `(|x|, 0)` in the complex element type.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename T>
        inline std::complex<T> safe_abs(const std::complex<T>& x)
        {
            return std::complex<T>(sycl::sqrt(x.real() * x.real() + x.imag() * x.imag()), 0);
        }

    } // namespace Dispatch
} // namespace SushiBLAS

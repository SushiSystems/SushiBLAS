/**************************************************************************/
/* complex_arithmetic.hpp                                                 */
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
 * @file complex_arithmetic.hpp
 * @brief Declares the conjugation, product and quotient a SYCL device can run on any element.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Names the real component type of T: T itself for a real type.
         * @see modules/foundation/arithmetic/README.md
         */
        template<typename T>
        struct RealOf
        {
            /** @brief The real component type. */
            using type = T;
        };

        /** @brief Names R as the real component type of std::complex<R>. */
        template<typename R>
        struct RealOf<std::complex<R>>
        {
            /** @brief The real component type. */
            using type = R;
        };

        /** @brief Shorthand for `RealOf<T>::type`. */
        template<typename T>
        using RealType = typename RealOf<T>::type;

        /** @brief Returns a real value unchanged. */
        template<typename T>
        [[nodiscard]] constexpr T conjugate(T value) noexcept
        {
            return value;
        }

        /** @brief Returns the complex conjugate of value. */
        template<typename R>
        [[nodiscard]] constexpr std::complex<R> conjugate(const std::complex<R>& value) noexcept
        {
            return std::complex<R>(value.real(), -value.imag());
        }

        /** @brief Returns a real value unchanged, whatever enabled is. */
        template<typename T>
        [[nodiscard]] constexpr T conjugate_if([[maybe_unused]] bool enabled, T value) noexcept
        {
            return value;
        }

        /** @brief Returns the conjugate of value when enabled is true, value otherwise. */
        template<typename R>
        [[nodiscard]] constexpr std::complex<R> conjugate_if(bool enabled,
                                                             const std::complex<R>& value) noexcept
        {
            return enabled ? conjugate(value) : value;
        }

        /** @brief Returns a real value unchanged. */
        template<typename T>
        [[nodiscard]] constexpr T real_part(T value) noexcept
        {
            return value;
        }

        /** @brief Returns value with its imaginary part set to zero, in the complex type. */
        template<typename R>
        [[nodiscard]] constexpr std::complex<R> real_part(const std::complex<R>& value) noexcept
        {
            return std::complex<R>(value.real(), R(0));
        }

        /** @brief Returns `a * b` for two real values. */
        template<typename T>
        [[nodiscard]] constexpr T multiply(T a, T b) noexcept
        {
            return a * b;
        }

        /**
         * @brief Returns the complex product `a * b` by the four-product formula.
         * @see modules/foundation/arithmetic/README.md
         */
        template<typename R>
        [[nodiscard]] constexpr std::complex<R> multiply(const std::complex<R>& a,
                                                         const std::complex<R>& b) noexcept
        {
            return std::complex<R>(a.real() * b.real() - a.imag() * b.imag(),
                                   a.real() * b.imag() + a.imag() * b.real());
        }

        /** @brief Returns a complex value scaled by a real factor, component by component. */
        template<typename R>
        [[nodiscard]] constexpr std::complex<R> multiply(R a, const std::complex<R>& b) noexcept
        {
            return std::complex<R>(a * b.real(), a * b.imag());
        }

        /** @brief Returns `a / b` for two real values. */
        template<typename T>
        [[nodiscard]] constexpr T divide(T a, T b) noexcept
        {
            return a / b;
        }

        /**
         * @brief Returns the complex quotient `a / b` by Smith's algorithm.
         * @return NaN components when b is zero.
         * @see modules/foundation/arithmetic/README.md
         */
        template<typename R>
        [[nodiscard]] constexpr std::complex<R> divide(const std::complex<R>& a,
                                                       const std::complex<R>& b) noexcept
        {
            const R c = b.real();
            const R d = b.imag();
            const R magnitude_c = (c < R(0)) ? -c : c;
            const R magnitude_d = (d < R(0)) ? -d : d;

            if (magnitude_c >= magnitude_d)
            {
                const R ratio = d / c;
                const R denominator = c + d * ratio;
                return std::complex<R>((a.real() + a.imag() * ratio) / denominator,
                                       (a.imag() - a.real() * ratio) / denominator);
            }

            const R ratio = c / d;
            const R denominator = c * ratio + d;
            return std::complex<R>((a.real() * ratio + a.imag()) / denominator,
                                   (a.imag() * ratio - a.real()) / denominator);
        }
    } // namespace Kernels
} // namespace SushiBLAS

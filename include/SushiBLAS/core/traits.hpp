/****************************************************************/
/* traits.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file traits.hpp
 * @brief Declares the compile-time type traits the library shares.
 * @author Mustafa Garip
 */

#pragma once

#include <complex>
#include <type_traits>

namespace SushiBLAS
{
    namespace Core
    {
        /**
         * @brief Detects whether a type is a `std::complex` specialization.
         *
         * @tparam T The type to inspect.
         */
        template<typename T> struct is_complex : std::false_type {};

        /**
         * @brief Specialization matching any `std::complex<T>`.
         *
         * @tparam T The complex number's value type.
         */
        template<typename T> struct is_complex<std::complex<T>> : std::true_type {};

        /**
         * @brief Convenience variable template for @ref is_complex.
         *
         * @tparam T The type to inspect.
         */
        template<typename T> inline constexpr bool is_complex_v = is_complex<T>::value;

    } // namespace Core
} // namespace SushiBLAS

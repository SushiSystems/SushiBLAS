/**************************************************************************/
/* traits.hpp                                                             */
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

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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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

/****************************************************************/
/* traits.hpp                                                   */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

#include <complex>
#include <type_traits>

// The project's compile-time type vocabulary.
//
// `is_complex` used to be written out three separate times, all three in the
// same `SushiBLAS::Internal` namespace: engine/blas/utils.hpp, the elementwise
// private helper, and the random one. Three identical primary templates in one
// namespace is only harmless while no translation unit sees more than one of
// them - and promoting the op-dispatch helpers to public headers (which a
// downstream consumer will include side by side) made that guarantee
// disappear. Collapsed to this single definition; utils.hpp now includes this
// file so anything that reached the trait through it still compiles.

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

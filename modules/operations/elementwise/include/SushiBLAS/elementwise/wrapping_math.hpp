/**************************************************************************/
/* wrapping_math.hpp                                                      */
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
 * @file wrapping_math.hpp
 * @brief Declares the arithmetic functors that wrap modulo 2^N on integer element types.
 * @author Mustafa Garip
 */

#pragma once

#include <type_traits>
#include <SushiBLAS/support/safe_math.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Returns `a + b`, wrapping modulo 2^N when T is an integer type.
         * @see modules/operations/MATH_README.md
         */
        template<typename T>
        inline T wrapping_add(T a, T b)
        {
            if constexpr (std::is_integral_v<T>)
            {
                using U = std::make_unsigned_t<T>;
                return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
            }
            else
            {
                return a + b;
            }
        }

        /**
         * @brief Returns `a - b`, wrapping modulo 2^N when T is an integer type.
         * @see modules/operations/MATH_README.md
         */
        template<typename T>
        inline T wrapping_sub(T a, T b)
        {
            if constexpr (std::is_integral_v<T>)
            {
                using U = std::make_unsigned_t<T>;
                return static_cast<T>(static_cast<U>(a) - static_cast<U>(b));
            }
            else
            {
                return a - b;
            }
        }

        /**
         * @brief Returns `a * b`, wrapping modulo 2^N when T is an integer type.
         * @see modules/operations/MATH_README.md
         */
        template<typename T>
        inline T wrapping_mul(T a, T b)
        {
            if constexpr (std::is_integral_v<T>)
            {
                using U = std::make_unsigned_t<T>;
                return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
            }
            else
            {
                return a * b;
            }
        }

        /**
         * @brief Returns `-x`, wrapping modulo 2^N when T is an integer type.
         * @see modules/operations/MATH_README.md
         */
        template<typename T>
        inline T wrapping_neg(T x)
        {
            if constexpr (std::is_integral_v<T>)
            {
                using U = std::make_unsigned_t<T>;
                return static_cast<T>(U(0) - static_cast<U>(x));
            }
            else
            {
                return -x;
            }
        }

        /**
         * @brief Returns `|x|`; the most negative integer maps to itself.
         * @see modules/operations/MATH_README.md
         */
        template<typename T>
        inline T wrapping_abs(T x)
        {
            if constexpr (std::is_integral_v<T>)
                return x < T(0) ? wrapping_neg(x) : x;
            else
                return safe_abs(x);
        }
    } // namespace Dispatch
} // namespace SushiBLAS

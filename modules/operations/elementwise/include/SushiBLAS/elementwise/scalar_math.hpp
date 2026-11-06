/**************************************************************************/
/* scalar_math.hpp                                                        */
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
 * @file scalar_math.hpp
 * @brief Declares the scalar functions the elementwise operations apply to one element, callable on the host and in a device kernel.
 * @author Mustafa Garip
 */

#pragma once

#include <type_traits>
#include <sycl/sycl.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Returns `acos(x)` through `sycl::acos`, in the type `sycl::acos` returns for T.
         */
        template<typename T>
        inline auto scalar_acos(T x)
        {
            return sycl::acos(x);
        }

        /**
         * @brief Returns `acosh(x)` through `sycl::acosh`, in the type `sycl::acosh` returns for T.
         */
        template<typename T>
        inline auto scalar_acosh(T x)
        {
            return sycl::acosh(x);
        }

        /**
         * @brief Returns `asin(x)` through `sycl::asin`, in the type `sycl::asin` returns for T.
         */
        template<typename T>
        inline auto scalar_asin(T x)
        {
            return sycl::asin(x);
        }

        /**
         * @brief Returns `asinh(x)` through `sycl::asinh`, in the type `sycl::asinh` returns for T.
         */
        template<typename T>
        inline auto scalar_asinh(T x)
        {
            return sycl::asinh(x);
        }

        /**
         * @brief Returns `atan(x)` through `sycl::atan`, in the type `sycl::atan` returns for T.
         */
        template<typename T>
        inline auto scalar_atan(T x)
        {
            return sycl::atan(x);
        }

        /**
         * @brief Returns `atanh(x)` through `sycl::atanh`, in the type `sycl::atanh` returns for T.
         */
        template<typename T>
        inline auto scalar_atanh(T x)
        {
            return sycl::atanh(x);
        }

        /**
         * @brief Returns `ceil(x)` through `sycl::ceil`, in the type `sycl::ceil` returns for T.
         */
        template<typename T>
        inline auto scalar_ceil(T x)
        {
            return sycl::ceil(x);
        }

        /**
         * @brief Returns `cos(x)` through `sycl::cos`, in the type `sycl::cos` returns for T.
         */
        template<typename T>
        inline auto scalar_cos(T x)
        {
            return sycl::cos(x);
        }

        /**
         * @brief Returns `cosh(x)` through `sycl::cosh`, in the type `sycl::cosh` returns for T.
         */
        template<typename T>
        inline auto scalar_cosh(T x)
        {
            return sycl::cosh(x);
        }

        /**
         * @brief Returns `floor(x)` through `sycl::floor`, in the type `sycl::floor` returns for T.
         */
        template<typename T>
        inline auto scalar_floor(T x)
        {
            return sycl::floor(x);
        }

        /**
         * @brief Returns `x / y` in the type the built-in division gives for the two operands.
         */
        template<typename A, typename B>
        inline auto scalar_div(A x, B y)
        {
            return x / y;
        }

        /**
         * @brief Returns `fmod(x, y)` through `sycl::fmod`.
         */
        template<typename A, typename B>
        inline auto scalar_fmod(A x, B y)
        {
            return sycl::fmod(x, y);
        }

        /**
         * @brief Returns `clamp(x, lo, hi)` through `sycl::clamp`.
         */
        template<typename X, typename Lo, typename Hi>
        inline auto scalar_clamp(X x, Lo lo, Hi hi)
        {
            return sycl::clamp(x, lo, hi);
        }

        /**
         * @brief Returns the max of `x` and `y`; on a floating type a NaN operand wins, the first one first.
         */
        template<typename A, typename B>
        inline auto scalar_max(A x, B y)
        {
            if constexpr (std::is_integral_v<A>)
                return sycl::max(x, y);
            else
                return (x != x) ? x : ((y != y) ? y : sycl::max(x, y));
        }

        /**
         * @brief Returns the min of `x` and `y`; on a floating type a NaN operand wins, the first one first.
         */
        template<typename A, typename B>
        inline auto scalar_min(A x, B y)
        {
            if constexpr (std::is_integral_v<A>)
                return sycl::min(x, y);
            else
                return (x != x) ? x : ((y != y) ? y : sycl::min(x, y));
        }
    } // namespace Dispatch
} // namespace SushiBLAS

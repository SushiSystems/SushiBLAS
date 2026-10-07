/**************************************************************************/
/* scalar.hpp                                                             */
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
 * @file scalar.hpp
 * @brief Declares ElementScalar, a double operation parameter narrowed once to every element type.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <limits>
#include <cstdint>
#include <type_traits>
#include <sycl/sycl.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds one double parameter as a value of each real and integer element type.
         * @see include/SushiBLAS/engine/math/README.md
         */
        class ElementScalar final
        {
            public:
                /**
                 * @brief Narrows value to every element type on the host.
                 * @param value Narrows to an integer type by saturating to its range, then
                 * truncating toward zero; NaN becomes 0.
                 */
                explicit ElementScalar(double value) noexcept
                    : as_half_(static_cast<sycl::half>(value)),
                      as_float_(static_cast<float>(value)),
                      as_double_(value),
                      as_int32_(to_integer<int32_t>(value)),
                      as_int64_(to_integer<int64_t>(value))
                {

                }

                /**
                 * @brief Returns the parameter in element type T.
                 * @tparam T sycl::half, float, double, int32_t or int64_t.
                 */
                template<typename T>
                [[nodiscard]] T as() const noexcept
                {
                    if constexpr (std::is_same_v<T, sycl::half>)
                        return as_half_;
                    else if constexpr (std::is_same_v<T, float>)
                        return as_float_;
                    else if constexpr (std::is_same_v<T, double>)
                        return as_double_;
                    else if constexpr (std::is_same_v<T, int32_t>)
                        return as_int32_;
                    else
                    {
                        static_assert(std::is_same_v<T, int64_t>,
                                      "ElementScalar serves HALF, FLOAT32, FLOAT64, INT32 and INT64.");
                        return as_int64_;
                    }
                }

                /** @brief Returns the parameter as given, for the task's metadata. */
                [[nodiscard]] double value() const noexcept
                {
                    return as_double_;
                }

            private:
                /**
                 * @brief Saturates value to the range of Int and truncates it toward zero.
                 * @return 0 for NaN.
                 */
                template<typename Int>
                [[nodiscard]] static Int to_integer(double value) noexcept
                {
                    if (std::isnan(value))
                        return Int(0);

                    if (value <= static_cast<double>(std::numeric_limits<Int>::min()))
                        return std::numeric_limits<Int>::min();

                    // The double nearest to max() may exceed it, so the comparison is >=.
                    if (value >= static_cast<double>(std::numeric_limits<Int>::max()))
                        return std::numeric_limits<Int>::max();

                    return static_cast<Int>(value);
                }

                /// @brief The parameter rounded to binary32, then to binary16.
                sycl::half as_half_;

                /// @brief The parameter rounded to binary32.
                float as_float_;

                /// @brief The parameter as given.
                double as_double_;

                /// @brief The parameter saturated and truncated to INT32.
                int32_t as_int32_;

                /// @brief The parameter saturated and truncated to INT64.
                int64_t as_int64_;
        };
    } // namespace Dispatch
} // namespace SushiBLAS

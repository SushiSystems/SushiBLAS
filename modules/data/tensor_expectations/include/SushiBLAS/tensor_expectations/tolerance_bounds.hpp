/**************************************************************************/
/* tolerance_bounds.hpp                                                   */
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
 * @file tolerance_bounds.hpp
 * @brief Declares the absolute tolerances a test compares with, each derived from a rounding bound.
 * @author Mustafa Garip
 * @see modules/data/tensor_expectations/README.md
 */

#pragma once

#include <cstddef>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief The unit roundoff of IEEE 754 binary16, 2^-11, which sycl::half carries.
         */
        inline constexpr double K_HALF_UNIT_ROUNDOFF = 0x1p-11;

        /**
         * @brief Returns the absolute bound on a value made by a chain of rounded operations.
         * @tparam Real The type the value is computed in: float or double.
         * @param operations The number of rounded operations on the longest path to the value.
         * @param magnitude The largest absolute value on that path.
         * @return threshold · operations · u · magnitude, with the threshold of tolerance.hpp.
         */
        template <typename Real>
        [[nodiscard]] constexpr double absolute_tolerance(std::size_t operations, double magnitude = 1.0) noexcept
        {
            return TestSupport::K_DEFAULT_THRESHOLD * static_cast<double>(operations)
                 * TestSupport::unit_roundoff<Real>() * magnitude;
        }

        /**
         * @brief Returns the absolute bound on a sycl::half value made by a chain of rounded operations.
         * @param operations The number of rounded operations on the longest path to the value.
         * @param magnitude The largest absolute value on that path.
         * @return threshold · operations · 2^-11 · magnitude.
         */
        [[nodiscard]] constexpr double half_absolute_tolerance(std::size_t operations, double magnitude = 1.0) noexcept
        {
            return TestSupport::K_DEFAULT_THRESHOLD * static_cast<double>(operations) * K_HALF_UNIT_ROUNDOFF * magnitude;
        }

        /**
         * @brief The number of rounded float operations the default comparison allows on a value of order one.
         */
        inline constexpr std::size_t K_DEFAULT_OPERATIONS = 128;

        /**
         * @brief The default absolute tolerance of expect_tensor: 128 float roundings of a value of order one.
         */
        inline constexpr double K_DEFAULT_TOLERANCE = absolute_tolerance<float>(K_DEFAULT_OPERATIONS);
    } // namespace Test
} // namespace SushiBLAS

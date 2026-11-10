/**************************************************************************/
/* rotg_table.hpp                                                          */
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
 * @file rotg_table.hpp
 * @brief Declares the real rotation table of the design and the values a row expects.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace RotgTable
        {
            /** @brief Names how z of a real rotation is rebuilt from its other values. */
            enum class RebuildValue : std::uint8_t
            {
                Zero,          /**< z is 0. */
                One,           /**< z is 1. */
                Sine,          /**< z is s. */
                InverseCosine /**< z is 1 / c. */
            };

            /** @brief Holds one row of the real table of the design, section 10.2. */
            struct RealRotationRow
            {
                double a;             /**< The first operand. */
                double b;             /**< The second operand. */
                double r;             /**< The expected r, which is exact. */
                double c_numerator;   /**< The expected c is this over c_denominator. */
                double c_denominator; /**< The divisor of the expected c. */
                double s_numerator;   /**< The expected s is this over s_denominator. */
                double s_denominator; /**< The divisor of the expected s. */
                RebuildValue z;       /**< The expected z. */
            };

            /// @brief The real table of the design, section 10.2.
            inline constexpr RealRotationRow K_REAL_ROWS[] = {
                {3.0, 4.0, 5.0, 3.0, 5.0, 4.0, 5.0, RebuildValue::InverseCosine},
                {4.0, 3.0, 5.0, 4.0, 5.0, 3.0, 5.0, RebuildValue::Sine},
                {-4.0, 3.0, -5.0, -4.0, -5.0, 3.0, -5.0, RebuildValue::Sine},
                {3.0, -4.0, -5.0, 3.0, -5.0, -4.0, -5.0, RebuildValue::InverseCosine},
                {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, RebuildValue::Zero},
                {0.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0, RebuildValue::One},
                {2.0, 0.0, 2.0, 1.0, 1.0, 0.0, 1.0, RebuildValue::Zero}};

            /** @brief Holds what one row of the real table expects, rounded to T once. */
            struct Expected
            {
                double r; /**< The expected r. */
                double z; /**< The expected z. */
                double c; /**< The expected cosine. */
                double s; /**< The expected sine. */
            };

            /** @brief Returns the expectation of a row, each quotient formed in the accumulation type of T. */
            template<typename T>
            [[nodiscard]] Expected expected_of(const RealRotationRow& row)
            {
                using Wide = Kernels::Accumulator<T>;
                const auto stored = [](Wide value)
                {
                    return Detail::to_double(static_cast<T>(value));
                };

                const Wide c = static_cast<Wide>(row.c_numerator) /
                               static_cast<Wide>(row.c_denominator);
                const Wide s = static_cast<Wide>(row.s_numerator) /
                               static_cast<Wide>(row.s_denominator);

                Wide z = Wide(0);
                switch (row.z)
                {
                    case RebuildValue::Zero:
                        z = Wide(0);
                        break;
                    case RebuildValue::One:
                        z = Wide(1);
                        break;
                    case RebuildValue::Sine:
                        z = s;
                        break;
                    case RebuildValue::InverseCosine:
                        z = Wide(1) / c;
                        break;
                }

                return Expected{stored(static_cast<Wide>(row.r)), stored(z), stored(c), stored(s)};
            }
        } // namespace RotgTable
    } // namespace Test
} // namespace SushiBLAS

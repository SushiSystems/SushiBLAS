/**************************************************************************/
/* test_cost.cpp                                                          */
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
 * @file test_cost.cpp
 * @brief Tests that Cost separates a stated cost from an absent one, and names the real class of a dtype.
 * @author Mustafa Garip
 * @see tests/foundation/cost/README.md
 */

#include <SushiBLAS/cost/cost.hpp>

#include <vector>

#include <gtest/gtest.h>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds a dtype and the real class the cost model judges it in. */
            struct ClassRow
            {
                Core::DataType dtype;     /**< The element type of the operands. */
                Core::DataType expected;  /**< The real class it is counted in. */
            };

            /** @brief Returns the real and the complex dtypes with the class each is judged in. */
            [[nodiscard]] std::vector<ClassRow> class_rows()
            {
                return {{Core::DataType::Half, Core::DataType::Half},
                        {Core::DataType::Float32, Core::DataType::Float32},
                        {Core::DataType::Float64, Core::DataType::Float64},
                        {Core::DataType::Int32, Core::DataType::Int32},
                        {Core::DataType::Complex32, Core::DataType::Float32},
                        {Core::DataType::Complex64, Core::DataType::Float64}};
            }
        } // namespace

        /** @brief Names the fixture of the class rows. */
        class Small_Cost : public ::testing::TestWithParam<ClassRow>
        {
        };

        TEST_P(Small_Cost, TheClassFollowsTheDtypeAndAComplexTypeIsCountedInItsRealClass)
        {
            const ClassRow& row = GetParam();
            EXPECT_EQ(real_class_of(row.dtype), row.expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_Cost, ::testing::ValuesIn(class_rows()));

        TEST(Small_CostStated, AnUncostedCostKeepsItsReasonAndStatesNothing)
        {
            const Cost cost = Cost::uncosted(UncostedReason::DataDependent);
            EXPECT_FALSE(cost.is_costed());
            EXPECT_EQ(cost.reason, UncostedReason::DataDependent);
            EXPECT_EQ(cost.flops, 0U);
        }

        TEST(Small_CostStated, WithBytesSetsTheBytesOfACostedCost)
        {
            Cost costed;
            costed.flops = 7;
            EXPECT_EQ(costed.with_bytes(64).bytes, 64U);
            EXPECT_EQ(costed.with_bytes(64).flops, 7U);
        }

        TEST(Small_CostStated, WithBytesLeavesAnUncostedCostAtZeroBytes)
        {
            EXPECT_EQ(Cost::uncosted(UncostedReason::NoRule).with_bytes(64).bytes, 0U);
        }
    } // namespace Test
} // namespace SushiBLAS

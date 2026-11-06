/**************************************************************************/
/* test_combiners.cpp                                                     */
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
 * @file test_combiners.cpp
 * @brief Tests what each combiner returns and its identity.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <SushiBLAS/graph_primitives/combiners.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using GraphPrimitives::Maximum;
            using GraphPrimitives::Minimum;
            using GraphPrimitives::Product;
            using GraphPrimitives::Sum;

            /// @brief Holds the value every neutral row combines with the identity.
            constexpr std::int32_t K_VALUE = -17;

            /// @brief Names a combiner and the result of combining its identity with K_VALUE.
            struct NeutralRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the combiner's identity combined with K_VALUE.
                std::int32_t combined;
            };

            /// @brief Names a combiner, the identity it reports and the constant that must be.
            struct IdentityRow
            {
                /// @brief Names the row in the test report.
                const char* name;
                /// @brief Holds the identity the combiner reports.
                double identity;
                /// @brief Holds the constant of the type the identity must equal.
                double expected;
            };

            /** @brief Returns the row of one combiner on std::int32_t. */
            template <typename Combiner>
            NeutralRow neutral_row_of(const char* name)
            {
                return {name, Combiner{}(Combiner::identity(), K_VALUE)};
            }

            /** @brief Returns the neutral rows of the four combiners. */
            std::vector<NeutralRow> neutral_rows()
            {
                return {neutral_row_of<Sum<std::int32_t>>("Sum"),
                        neutral_row_of<Product<std::int32_t>>("Product"),
                        neutral_row_of<Minimum<std::int32_t>>("Minimum"),
                        neutral_row_of<Maximum<std::int32_t>>("Maximum")};
            }

            /** @brief Returns the identity rows of the four combiners on double and float. */
            std::vector<IdentityRow> identity_rows()
            {
                return {
                    {"SumDouble", Sum<double>::identity(), 0.0},
                    {"ProductDouble", Product<double>::identity(), 1.0},
                    {"MinimumDouble", Minimum<double>::identity(),
                     std::numeric_limits<double>::max()},
                    {"MaximumDouble", Maximum<double>::identity(),
                     std::numeric_limits<double>::lowest()},
                    {"SumFloat", static_cast<double>(Sum<float>::identity()), 0.0},
                    {"ProductFloat", static_cast<double>(Product<float>::identity()), 1.0},
                    {"MinimumFloat", static_cast<double>(Minimum<float>::identity()),
                     static_cast<double>(std::numeric_limits<float>::max())},
                    {"MaximumFloat", static_cast<double>(Maximum<float>::identity()),
                     static_cast<double>(std::numeric_limits<float>::lowest())},
                };
            }

            /** @brief Names a test case after its row. */
            template <typename Row>
            std::string row_name(const ::testing::TestParamInfo<Row>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Tests that Sum adds its operands. */
        TEST(Small_GraphPrimitivesCombiners, SumAddsTwoValues)
        {
            EXPECT_EQ(Sum<std::int32_t>{}(3, 4), 7);
        }

        /** @brief Tests that Product multiplies its operands. */
        TEST(Small_GraphPrimitivesCombiners, ProductMultipliesTwoValues)
        {
            EXPECT_EQ(Product<std::int32_t>{}(3, 4), 12);
        }

        /** @brief Tests that Minimum returns the smaller operand in both orders. */
        TEST(Small_GraphPrimitivesCombiners, MinimumReturnsTheSmallerOfTwoValues)
        {
            EXPECT_EQ(Minimum<std::int32_t>{}(3, 4), 3);
            EXPECT_EQ(Minimum<std::int32_t>{}(4, 3), 3);
        }

        /** @brief Tests that Maximum returns the larger operand in both orders. */
        TEST(Small_GraphPrimitivesCombiners, MaximumReturnsTheLargerOfTwoValues)
        {
            EXPECT_EQ(Maximum<std::int32_t>{}(3, 4), 4);
            EXPECT_EQ(Maximum<std::int32_t>{}(4, 3), 4);
        }

        /// @brief Runs one combiner's neutral row.
        class Small_GraphPrimitivesCombinerNeutral : public ::testing::TestWithParam<NeutralRow>
        {
        };

        /** @brief Tests that combining a value with the identity returns the value. */
        TEST_P(Small_GraphPrimitivesCombinerNeutral, IdentityLeavesAValueUnchanged)
        {
            EXPECT_EQ(GetParam().combined, K_VALUE);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesCombinerNeutral,
                                 ::testing::ValuesIn(neutral_rows()), row_name<NeutralRow>);

        /// @brief Runs one combiner's identity row on a floating-point type.
        class Small_GraphPrimitivesCombinerIdentity : public ::testing::TestWithParam<IdentityRow>
        {
        };

        /** @brief Tests that an identity is the constant of its element type, exactly. */
        TEST_P(Small_GraphPrimitivesCombinerIdentity, TheIdentityIsTheConstantOfTheType)
        {
            EXPECT_EQ(GetParam().identity, GetParam().expected);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_GraphPrimitivesCombinerIdentity,
                                 ::testing::ValuesIn(identity_rows()), row_name<IdentityRow>);

    } // namespace Test
} // namespace SushiBLAS
